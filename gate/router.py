"""LLM router: a provider fallback chain with retries, tool calling and a cost guard.

Providers are tried in order. Each gets up to MAX_RETRIES attempts with backoff;
auth, billing and unknown-model errors (401/402/403/404) skip straight to the next
provider, because retrying them cannot help. An empty answer counts as a failure.
Tool calls are executed and fed back to the model until it answers in plain text
or the round limit is reached.
"""
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI

from gate.config import config
from gate.tools import TOOLS

logger = logging.getLogger("gate.router")

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]
FAIL_FAST_CODES = (401, 402, 403, 404)

# Text a model writes alongside tool calls is kept for the final answer only when it is
# substantial; short "calling the search tool..." chatter is dropped.
PREFACE_MIN_CHARS = int(os.getenv("VIKA_PREFACE_MIN_CHARS", "200"))
MAX_TOKENS = int(os.getenv("VIKA_MAX_TOKENS", "6000"))

DEEPSEEK_MODEL = os.getenv("VIKA_DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE = os.getenv("VIKA_DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_EFFORT = os.getenv("VIKA_DEEPSEEK_EFFORT", "high")  # low / high / max
OPENROUTER_MODEL = os.getenv("VIKA_MODEL", "deepseek/deepseek-v4-pro")
OR_PROVIDER_IGNORE = [x.strip() for x in os.getenv("VIKA_OR_PROVIDER_IGNORE", "StreamLake").split(",")
                      if x.strip()]
DO_MODEL = os.getenv("VIKA_DO_MODEL", "openai-gpt-oss-120b")
GROQ_MODEL = os.getenv("VIKA_GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("VIKA_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Cost guard. During DeepSeek peak windows (weekdays, UTC) the pro model and heavy requests
# are routed around DeepSeek to cheaper providers. Outside the windows the chain is unchanged.
PEAK_HOURS_UTC = [(1, 4), (6, 10)]
PEAK_GUARD_ON = os.getenv("VIKA_DEEPSEEK_PEAK_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
PEAK_HEAVY_CHARS = int(os.getenv("VIKA_PEAK_HEAVY_CHARS", "6000"))

# Per-turn reasoning effort (low / med / high) mapped to each provider's own scale.
GROQ_EFFORT_MAP = {"low": "low", "med": "medium", "high": "high"}
DS_EFFORT_MAP = {"low": "low", "med": "high", "high": "max"}

MSG_STOPPED = "⏹ Зупинено за /stop."
MSG_ALL_DOWN = "❌ Усі провайдери недоступні. Спробуй пізніше."
MSG_CAP = ("⏸ Досягнуто ліміту кроків на одне повідомлення ({n}). "
           "Напиши «продовжуй» — доведу до кінця.")
CAP_NOTE = ("Internal step limit for ONE message reached ({n} tool calls). Do not call tools anymore. "
            "In the user's language, briefly (up to 6 points) summarize what was done and with what actual "
            "result, and what remains. Say honestly that you hit the per-message step limit and the user can "
            "reply 'continue'. Do not invent external platform or API limits.")
TRUNCATED_ARGS = ("ERROR: tool call arguments were truncated (too long for the response limit). "
                  "Retry with shorter arguments: split large text across several calls.")


def peak_now_utc(now: Optional[datetime] = None) -> bool:
    """True during DeepSeek peak hours (weekdays, UTC). `now` can be injected in tests."""
    dt = now or datetime.now(timezone.utc)
    if dt.weekday() >= 5:  # Saturday and Sunday are off-peak all day
        return False
    return any(a <= dt.hour < b for a, b in PEAK_HOURS_UTC)


def peak_skip_deepseek(model: str, n_chars: int = 0, now: Optional[datetime] = None) -> bool:
    """True when DeepSeek must not be called right now: peak hours and (pro model or a heavy request)."""
    if not PEAK_GUARD_ON or not peak_now_utc(now):
        return False
    if "pro" in (model or "").lower():
        return True
    return n_chars >= PEAK_HEAVY_CHARS


class LLMRouter:
    """Routes chat requests across LLM providers with retries, fallback and tool calling."""

    def __init__(self):
        self.deepseek_client: Optional[OpenAI] = None
        self.deepseek_model = DEEPSEEK_MODEL
        self.openrouter_client: Optional[OpenAI] = None
        self.openrouter_model = OPENROUTER_MODEL
        self.do_client: Optional[OpenAI] = None
        self.groq_client: Optional[OpenAI] = None
        self.gemini_client: Optional[OpenAI] = None
        # Per-turn settings live in thread-local storage: turns are served by threads, so the
        # model choice and cancellation of one chat must not leak into another.
        self._tl = threading.local()

        self._init_deepseek()
        self._init_openrouter()
        self._init_do()
        self._init_groq()
        self._init_gemini()

        # Providers without a key return None and are skipped.
        self.chain = [
            ("deepseek", self._call_deepseek),
            ("openrouter", self._call_openrouter),
            ("do", self._call_do),
            ("groq", self._call_groq),
            ("gemini", self._call_gemini),
        ]

    # ── per-turn settings ──
    def set_turn(self, provider: Optional[str] = None, deepseek_model: Optional[str] = None,
                 openrouter_model: Optional[str] = None, cancel_check=None,
                 effort: Optional[str] = None) -> None:
        """`provider` goes first in the chain for this turn; the rest stay as fallback."""
        self._tl.provider = provider
        self._tl.deepseek_model = deepseek_model
        self._tl.openrouter_model = openrouter_model
        self._tl.cancel_check = cancel_check
        self._tl.effort = effort

    def clear_turn(self) -> None:
        self.set_turn()

    def _effort(self) -> Optional[str]:
        return getattr(self._tl, "effort", None)

    def _ds_model(self) -> str:
        return getattr(self._tl, "deepseek_model", None) or self.deepseek_model

    def _or_model(self) -> str:
        return getattr(self._tl, "openrouter_model", None) or self.openrouter_model

    def _last_chars(self) -> int:
        return int(getattr(self._tl, "last_chars", 0) or 0)

    def _ordered_chain(self) -> list:
        pref = getattr(self._tl, "provider", None)
        chain = list(self.chain)
        # At peak the pro model is skipped both directly and via OpenRouter (same model there).
        if peak_skip_deepseek(self._ds_model(), self._last_chars()):
            chain = [c for c in chain if c[0] not in ("deepseek", "openrouter")]
            chain = [c for c in chain if c[0] == "groq"] + [c for c in chain if c[0] != "groq"]
            logger.warning(f"DeepSeek peak guard: model={self._ds_model()!r} chars={self._last_chars()} "
                           f"— routing around DeepSeek")
        if not pref:
            return chain
        first = [c for c in chain if c[0] == pref]
        return first + [c for c in chain if c[0] != pref]

    def _cancelled(self) -> bool:
        chk = getattr(self._tl, "cancel_check", None)
        try:
            return bool(chk and chk())
        except Exception:
            return False

    # ── providers ──
    def _init_deepseek(self):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return
        try:
            self.deepseek_client = OpenAI(base_url=DEEPSEEK_BASE, api_key=key)
            logger.info(f"DeepSeek initialized ({self.deepseek_model})")
        except Exception as e:
            logger.warning(f"DeepSeek init failed: {e}")

    def _call_deepseek(self, messages: list[dict], tools=None) -> Optional[dict]:
        """DeepSeek direct. In thinking mode with tools DeepSeek requires the assistant's
        reasoning_content to be sent back, so it is returned up and kept in the turn history."""
        if not self.deepseek_client:
            return None
        ds_model = self._ds_model()
        if peak_skip_deepseek(ds_model, self._last_chars()):
            logger.warning(f"DeepSeek peak guard (in call): skip model={ds_model!r}")
            return None
        kwargs = {"model": ds_model, "messages": messages, "max_tokens": MAX_TOKENS, "timeout": 180,
                  "reasoning_effort": DS_EFFORT_MAP.get(self._effort() or "", DEEPSEEK_EFFORT)}
        if tools:
            kwargs["tools"] = tools
        resp = self.deepseek_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        tool_calls = getattr(choice.message, "tool_calls", None)
        if not content.strip() and not tool_calls:
            raise ValueError(f"empty content (finish={choice.finish_reason})")
        result = {"content": content, "_usage": getattr(resp, "usage", None), "_model": ds_model}
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            result["reasoning_content"] = reasoning
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def _init_openrouter(self):
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            return
        try:
            self.openrouter_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                                            default_headers={"X-Title": "vika_relay"})
            logger.info(f"OpenRouter initialized ({self.openrouter_model})")
        except Exception as e:
            logger.warning(f"OpenRouter init failed: {e}")

    def _call_openrouter(self, messages: list[dict], tools=None) -> Optional[dict]:
        if not self.openrouter_client:
            return None
        kwargs = {"model": self._or_model(), "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if OR_PROVIDER_IGNORE:
            kwargs["extra_body"] = {"provider": {"ignore": OR_PROVIDER_IGNORE}}
        # Without max_tokens OpenRouter reserves the model's full output window and,
        # on a low balance, answers 402.
        kwargs["max_tokens"] = MAX_TOKENS
        resp = self.openrouter_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        # Empty content without tool calls is a provider failure: retry rather than
        # surface the model's reasoning to the user.
        if not content.strip() and not getattr(choice.message, "tool_calls", None):
            raise ValueError(f"empty content (finish={choice.finish_reason}, "
                             f"provider={getattr(resp, 'provider', '?')})")
        result = {"content": content, "_model": kwargs["model"]}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _init_do(self):
        if not (config.do_api_key and config.do_base_url):
            return
        try:
            self.do_client = OpenAI(base_url=config.do_base_url, api_key=config.do_api_key)
            logger.info("DO endpoint initialized")
        except Exception as e:
            logger.warning(f"DO init failed: {e}")

    def _call_do(self, messages: list[dict], tools=None) -> Optional[dict]:
        if not self.do_client:
            return None
        kwargs = {"model": DO_MODEL, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self.do_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or ""}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _init_groq(self):
        if not config.groq_api_key:
            return
        try:
            self.groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=config.groq_api_key)
            logger.info("Groq initialized")
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    def _call_groq(self, messages: list[dict], tools=None) -> Optional[dict]:
        if not self.groq_client:
            return None
        kwargs = {"model": GROQ_MODEL, "messages": messages, "max_tokens": MAX_TOKENS}
        eff = GROQ_EFFORT_MAP.get(self._effort() or "")
        if eff and "gpt-oss" in GROQ_MODEL:
            kwargs["reasoning_effort"] = eff
        if tools:
            kwargs["tools"] = tools
        resp = self.groq_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or "", "_usage": getattr(resp, "usage", None)}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _init_gemini(self):
        if not config.gemini_api_key:
            return
        try:
            self.gemini_client = OpenAI(base_url=GEMINI_OPENAI_BASE, api_key=config.gemini_api_key)
            logger.info(f"Gemini initialized ({GEMINI_MODEL}, OpenAI-compatible)")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    def _call_gemini(self, messages: list[dict], tools=None) -> Optional[dict]:
        if not self.gemini_client:
            return None
        kwargs = {"model": GEMINI_MODEL, "messages": messages, "max_tokens": MAX_TOKENS, "timeout": 120}
        if tools:
            kwargs["tools"] = tools
        resp = self.gemini_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or ""}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    # ── main loop ──
    @staticmethod
    def _strip_reasoning(messages: list[dict]) -> list[dict]:
        """Drop reasoning_content for non-DeepSeek providers (some reject the unknown field)."""
        if not any(isinstance(m, dict) and "reasoning_content" in m for m in messages):
            return messages
        return [{k: v for k, v in m.items() if k != "reasoning_content"} if isinstance(m, dict) else m
                for m in messages]

    def _log_ok(self, name: str, result: dict, round_num: int):
        usage = result.get("_usage")
        model = result.get("_model") or {"openrouter": self._or_model(), "do": DO_MODEL,
                                         "groq": GROQ_MODEL, "gemini": GEMINI_MODEL}.get(name, "?")
        u = ""
        if usage is not None:
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            u = (f" in={getattr(usage, 'prompt_tokens', '?')} out={getattr(usage, 'completion_tokens', '?')}"
                 + (f" cache_hit={hit}" if hit is not None else ""))
        logger.info(f"LLM ok provider={name} model={model} round={round_num} "
                    f"tool_calls={len(result.get('tool_calls') or [])}{u}")

    def ask(self, messages: list[dict], tool_executor=None, tools=None, max_rounds: int = 10) -> str:
        """Run one turn: try providers in order, execute tool calls, loop until a text answer.

        `messages` is extended in place with the assistant tool calls and tool results.
        """
        # Input size feeds the peak guard: heavy requests skip DeepSeek at peak.
        self._tl.last_chars = sum(len(m.get("content") or "") for m in messages if isinstance(m, dict))
        preface: list[str] = []

        def _join(final):
            parts = preface + ([final.strip()] if final and final.strip() else [])
            return "\n\n".join(parts) if parts else (final or "")

        chain = self._ordered_chain()
        for round_num in range(max_rounds):
            if self._cancelled():
                logger.info(f"LLM cancelled at round={round_num}")
                return _join(MSG_STOPPED)
            result = None
            for name, caller in chain:
                for attempt in range(MAX_RETRIES):
                    try:
                        msgs = messages if name == "deepseek" else self._strip_reasoning(messages)
                        candidate = caller(msgs, tools=((tools if tools is not None else TOOLS)
                                                        if tool_executor else None))
                        if (candidate and not candidate.get("tool_calls")
                                and not str(candidate.get("content") or "").strip()):
                            raise ValueError("empty response")
                        # Only a validated answer is kept: an empty one must not survive the
                        # retries and stop the chain.
                        if candidate:
                            result = candidate
                            break
                    except Exception as e:
                        delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                        logger.warning(f"[{name}] attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")
                        # No key, no balance, no access or no such model: retrying cannot help.
                        if getattr(e, "status_code", None) in FAIL_FAST_CODES:
                            break
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(delay)
                if result:
                    self._log_ok(name, result, round_num)
                    break

            if not result:
                return MSG_ALL_DOWN

            if not result.get("tool_calls") or not tool_executor:
                return _join(result["content"])

            pre = str(result.get("content") or "").strip()
            if len(pre) >= PREFACE_MIN_CHARS:
                preface.append(pre)

            asst = {
                "role": "assistant",
                "content": result["content"] or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in result["tool_calls"]
                ],
            }
            if result.get("reasoning_content"):
                asst["reasoning_content"] = result["reasoning_content"]
            messages.append(asst)

            for tc in result["tool_calls"]:
                if self._cancelled():
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "cancelled by /stop"})
                    continue
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    # The model hit its output limit mid-arguments: ask it to retry in parts.
                    logger.warning(f"Tool args JSON broken for {fn_name}: {e}")
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": TRUNCATED_ARGS})
                    continue
                logger.info(f"Tool call: {fn_name}({fn_args})")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(tool_executor(fn_name, fn_args))})

        return self._cap_summary(messages, max_rounds, preface)

    def _cap_summary(self, messages: list[dict], max_rounds: int, preface: list[str]) -> str:
        """Round limit reached: ask for an honest summary without tools instead of cutting off."""
        msgs = messages + [{"role": "user", "content": f"[system note] {CAP_NOTE.format(n=max_rounds)}"}]
        for name, caller in self._ordered_chain():
            try:
                r = caller(msgs if name == "deepseek" else self._strip_reasoning(msgs), tools=None)
                if r and (r.get("content") or "").strip():
                    return r["content"].strip()
            except Exception as e:
                logger.warning(f"[{name}] cap summary failed: {e}")
        done = "\n\n".join(preface)[-1500:]
        return ((done + "\n\n") if done else "") + MSG_CAP.format(n=max_rounds)

    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Speech to text via Groq Whisper."""
        if not self.groq_client:
            return None
        with open(file_path, "rb") as f:
            result = self.groq_client.audio.transcriptions.create(
                model="whisper-large-v3", file=f, response_format="text")
        return result.text.strip() if hasattr(result, "text") else str(result).strip()
