"""LLM provider clients with retry, fallback chain, and function calling."""
import logging
import threading
import time
import json
import os
from typing import Optional

from openai import OpenAI

from src.core.config import config
from src.services.tool_defs import TOOLS

logger = logging.getLogger("vika.llm")

MAX_RETRIES = 3
RETRY_DELAYS = [1, 3, 5]


PREFACE_MIN_CHARS = int(os.getenv("VIKA_PREFACE_MIN_CHARS", "200"))
GROQ_MODEL = os.getenv("VIKA_GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("VIKA_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_MODEL = os.getenv("VIKA_DEEPSEEK_MODEL", "deepseek-flash")
DEEPSEEK_BASE = os.getenv("VIKA_DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_EFFORT = os.getenv("VIKA_DEEPSEEK_EFFORT", "high")  # low/high/max
OR_MAX_TOKENS = int(os.getenv("VIKA_MAX_TOKENS", "6000"))
OR_PROVIDER_IGNORE = [x.strip() for x in os.getenv("VIKA_OR_PROVIDER_IGNORE", "").split(",") if x.strip()]


PEAK_HOURS_UTC = [(1, 4), (6, 10)]
GROQ_EFFORT_MAP = {"low": "low", "med": "medium", "high": "high"}
DS_EFFORT_MAP = {"low": "low", "med": "high", "high": "max"}
PEAK_GUARD_ON = os.getenv("VIKA_DEEPSEEK_PEAK_GUARD", "1").strip().lower() not in ("0", "false", "no", "off")
PEAK_HEAVY_CHARS = int(os.getenv("VIKA_PEAK_HEAVY_CHARS", "6000"))


def peak_now_utc(now=None) -> bool:
    """True during DeepSeek peak-price hours (UTC, weekdays). `now` is injectable for tests."""
    from datetime import datetime, timezone
    dt = now or datetime.now(timezone.utc)
    if dt.weekday() >= 5:
        return False
    return any(a <= dt.hour < b for a, b in PEAK_HOURS_UTC)


def peak_skip_deepseek(model: str, n_chars: int = 0, now=None) -> bool:
    """True if DeepSeek should be skipped now: peak hours and (pro model or a heavy request)."""
    if not PEAK_GUARD_ON or not peak_now_utc(now):
        return False
    if "pro" in (model or "").lower():
        return True
    return n_chars >= PEAK_HEAVY_CHARS


class LLMProvider:
    """Manages LLM clients with automatic retry, fallback, and tool calling."""

    def __init__(self):
        self.do_client: Optional[OpenAI] = None
        self.groq_client: Optional[OpenAI] = None
        self.gemini_model = None
        self.claude_client = None
        self.openrouter_client: Optional[OpenAI] = None
        self.openrouter_model = None
        self.deepseek_client: Optional[OpenAI] = None
        self.deepseek_model = DEEPSEEK_MODEL
        self._tl = threading.local()

        self._init_openrouter()
        self._init_deepseek()
        self._init_claude()
        self._init_do()
        self._init_groq()
        self._init_gemini()

        self.chain = [
            ("deepseek", self._call_deepseek),
            ("openrouter", self._call_openrouter),
            ("claude", self._call_claude),
            ("do", self._call_do),
            ("groq", self._call_groq),
            ("gemini", self._call_gemini),
        ]

    def set_turn(self, provider: Optional[str] = None, deepseek_model: Optional[str] = None,
                 openrouter_model: Optional[str] = None, cancel_check=None,
                 effort: Optional[str] = None) -> None:
        """provider: which backend goes first in the chain; the rest stay as fallback."""
        self._tl.provider = provider
        self._tl.deepseek_model = deepseek_model
        self._tl.openrouter_model = openrouter_model
        self._tl.cancel_check = cancel_check
        self._tl.effort = effort

    def _effort(self) -> Optional[str]:
        return getattr(self._tl, "effort", None)

    def clear_turn(self) -> None:
        self.set_turn()

    def _ds_model(self) -> str:
        return getattr(self._tl, "deepseek_model", None) or self.deepseek_model

    def _or_model(self) -> str:
        return getattr(self._tl, "openrouter_model", None) or self.openrouter_model

    def _last_chars(self) -> int:
        return int(getattr(self._tl, "last_chars", 0) or 0)

    def _ordered_chain(self) -> list:
        pref = getattr(self._tl, "provider", None)
        chain = list(self.chain)
        if peak_skip_deepseek(self._ds_model(), self._last_chars()):
            chain = [c for c in chain if c[0] not in ("deepseek", "openrouter")]
            chain = [c for c in chain if c[0] == "groq"] + [c for c in chain if c[0] != "groq"]
            logger.warning(f"DeepSeek peak guard: model={self._ds_model()!r} "
                           f"chars={self._last_chars()} - skipping DeepSeek, using groq/local")
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

    def _init_openrouter(self):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return
        try:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
                default_headers={"X-Title": "vika_relay"},
            )
            self.openrouter_model = os.getenv("VIKA_MODEL", "deepseek/deepseek-v4-pro")
            logger.info(f"OpenRouter initialized ({self.openrouter_model})")
        except Exception as e:
            logger.warning(f"OpenRouter init failed: {e}")

    def _call_openrouter(self, messages: list[dict], tools=None) -> dict:
        """DeepSeek models via OpenRouter."""
        if not self.openrouter_client:
            return None
        kwargs = {"model": self._or_model(), "messages": messages}
        if tools:
            kwargs["tools"] = tools
        if OR_PROVIDER_IGNORE:
            kwargs["extra_body"] = {"provider": {"ignore": OR_PROVIDER_IGNORE}}
        kwargs["max_tokens"] = OR_MAX_TOKENS
        resp = self.openrouter_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        if not content.strip() and not getattr(choice.message, "tool_calls", None):
            raise ValueError(f"empty content (finish={choice.finish_reason}, "
                             f"provider={getattr(resp, 'provider', '?')})")
        result = {"content": content, "_model": kwargs["model"]}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _init_claude(self):
        if not config.anthropic_api_key:
            return
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
            logger.info("Claude initialized")
        except Exception as e:
            logger.warning(f"Claude init failed: {e}")

    def _init_do(self):
        if not config.do_api_key:
            return
        try:
            self.do_client = OpenAI(
                base_url=config.do_base_url,
                api_key=config.do_api_key,
            )
            logger.info("DO Agent initialized")
        except Exception as e:
            logger.warning(f"DO init failed: {e}")

    def _init_groq(self):
        if not config.groq_api_key:
            return
        try:
            self.groq_client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=config.groq_api_key,
            )
            logger.info("Groq initialized")
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    def _init_gemini(self):
        if not config.gemini_api_key:
            return
        try:
            self.gemini_model = OpenAI(base_url=GEMINI_OPENAI_BASE, api_key=config.gemini_api_key)
            logger.info(f"Gemini initialized ({GEMINI_MODEL}, OpenAI-compatible)")
        except Exception as e:
            logger.warning(f"Gemini init failed: {e}")

    def _call_claude(self, messages: list[dict], tools=None) -> dict:
        if not self.claude_client:
            return None
        
        system_msg = ""
        filtered_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg += m["content"] + "\n"
            else:
                filtered_messages.append({"role": m["role"], "content": m["content"]})

        try:
            resp = self.claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4096,
                system=system_msg,
                messages=filtered_messages,
                tools=tools,
            )
            
            content = ""
            tool_calls = []
            if resp.content:
                for block in resp.content:
                    if block.type == "text":
                        content += block.text
                    elif block.type == "tool_use":
                        tool_calls.append({
                            "id": block.id,
                            "function": {
                                "name": block.name,
                                "arguments": block.input
                            }
                        })
            
            return {"content": content.strip(), "tool_calls": tool_calls} if tool_calls or content else None
        except Exception as e:
            logger.warning(f"Claude call failed: {e}")
            return None

    def _call_do(self, messages: list[dict], tools=None) -> dict:
        """Returns dict with 'content' and optional 'tool_calls'."""
        if not self.do_client:
            return None
        kwargs = {"model": "openai-gpt-oss-120b", "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = self.do_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or ""}
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _call_groq(self, messages: list[dict], tools=None) -> dict:
        if not self.groq_client:
            return None
        kwargs = {"model": GROQ_MODEL, "messages": messages, "max_tokens": OR_MAX_TOKENS}
        eff = GROQ_EFFORT_MAP.get(self._effort() or "")
        if eff and "gpt-oss" in GROQ_MODEL:
            kwargs["reasoning_effort"] = eff
        if tools:
            kwargs["tools"] = tools
        resp = self.groq_client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or "", "_usage": getattr(resp, "usage", None)}
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _call_gemini(self, messages: list[dict], tools=None) -> dict:
        if not self.gemini_model:
            return None
        kwargs = {"model": GEMINI_MODEL, "messages": messages, "max_tokens": OR_MAX_TOKENS, "timeout": 120}
        if tools:
            kwargs["tools"] = tools
        resp = self.gemini_model.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        result = {"content": choice.message.content or ""}
        if getattr(choice.message, "tool_calls", None):
            result["tool_calls"] = choice.message.tool_calls
        return result

    def _init_deepseek(self):
        key = os.getenv("DEEPSEEK_API_KEY")
        if not key:
            return
        try:
            self.deepseek_client = OpenAI(base_url=DEEPSEEK_BASE, api_key=key)
            logger.info(f"DeepSeek direct initialized ({self.deepseek_model})")
        except Exception as e:
            logger.warning(f"DeepSeek init failed: {e}")

    def _call_deepseek(self, messages: list[dict], tools=None) -> dict:
        """DeepSeek direct API. In thinking mode with tools, reasoning_content is passed back up so ask() can keep it in the turn history."""
        if not self.deepseek_client:
            return None
        ds_model = self._ds_model()
        if peak_skip_deepseek(ds_model, self._last_chars()):
            logger.warning(f"DeepSeek peak guard (in-call): skip model={ds_model!r} (peak UTC)")
            return None
        kwargs = {"model": ds_model, "messages": messages, "max_tokens": OR_MAX_TOKENS,
                  "timeout": 180,
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

    @staticmethod
    def _strip_reasoning(messages: list[dict]) -> list[dict]:
        """Strip reasoning_content for non-DeepSeek providers."""
        if not any(isinstance(m, dict) and "reasoning_content" in m for m in messages):
            return messages
        return [{k: v for k, v in m.items() if k != "reasoning_content"} if isinstance(m, dict) else m
                for m in messages]

    def _log_ok(self, name: str, result: dict, round_num: int):
        usage = result.get("_usage")
        model = result.get("_model") or {"openrouter": self._or_model(), "groq": GROQ_MODEL,
                                         "gemini": GEMINI_MODEL}.get(name, "?")
        u = ""
        if usage is not None:
            hit = getattr(usage, "prompt_cache_hit_tokens", None)
            u = (f" in={getattr(usage, 'prompt_tokens', '?')} out={getattr(usage, 'completion_tokens', '?')}"
                 + (f" cache_hit={hit}" if hit is not None else ""))
        logger.info(f"LLM ok provider={name} model={model} round={round_num} "
                    f"tool_calls={len(result.get('tool_calls') or [])}{u}")

    def ask(self, messages: list[dict], tool_executor=None, tools=None,
            max_rounds: int = 10) -> str:
        """
        Try each provider with retries. If tool_calls returned,
        execute them and loop back to LLM for final answer.
        Max 10 tool call rounds to prevent infinite loops.
        """
        self._tl.last_chars = sum(len(m.get("content") or "") for m in messages if isinstance(m, dict))
        preface: list[str] = []

        def _join(final):
            parts = preface + ([final.strip()] if final and final.strip() else [])
            return "\n\n".join(parts) if parts else (final or "")

        chain = self._ordered_chain()
        for round_num in range(max_rounds):
            if self._cancelled():
                logger.info(f"LLM cancelled by /stop at round={round_num}")
                return _join("⏹ Stopped by /stop.")
            result = None
            for name, caller in chain:
                for attempt in range(MAX_RETRIES):
                    try:
                        msgs = messages if name == "deepseek" else self._strip_reasoning(messages)
                        result = caller(msgs, tools=((tools if tools is not None else TOOLS) if tool_executor else None))
                        if result and not result.get("tool_calls") and not str(result.get("content") or "").strip():
                            raise ValueError("empty response")
                        if result:
                            break
                    except Exception as e:
                        delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                        logger.warning(f"[{name}] attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
                        if getattr(e, "status_code", None) in (401, 402, 403, 404):
                            break
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(delay)
                if result:
                    self._log_ok(name, result, round_num)
                    break

            if not result:
                return "❌ All providers are unavailable. Try again later."

            # No tool calls — return content
            if "tool_calls" not in result or not result["tool_calls"]:
                return _join(result["content"])

            # Execute tool calls
            if not tool_executor:
                return _join(result["content"])

            _pre = str(result.get("content") or "").strip()
            if len(_pre) >= PREFACE_MIN_CHARS:
                preface.append(_pre)

            asst = {
                "role": "assistant",
                "content": result["content"] or None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in result["tool_calls"]
                ]
            }
            if result.get("reasoning_content"):
                asst["reasoning_content"] = result["reasoning_content"]
            messages.append(asst)

            # Execute each tool call and add results
            for tc in result["tool_calls"]:
                if self._cancelled():
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": "cancelled by /stop"})
                    continue
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Tool args JSON broken for {fn_name}: {e}; len={len(tc.function.arguments or chr(32))}")
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                        "content": "ERROR: tool-call arguments were truncated (too long for the output limit). Retry with shorter arguments, splitting large content into several calls."})
                    continue
                logger.info(f"Tool call: {fn_name}({fn_args})")

                tool_result = tool_executor(fn_name, fn_args)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(tool_result),
                })

            # Loop back to LLM with tool results

        return self._cap_summary(messages, max_rounds, preface)

    def _cap_summary(self, messages: list[dict], max_rounds: int, preface: list[str]) -> str:
        """Tool-round limit reached: ask for an honest summary without tools."""
        note = (f"Tool-call limit for a single message reached ({max_rounds} rounds). "
                "Do not call tools anymore. Briefly summarize what was done, with real results, and what is left. "
                "Say that the step limit was reached and the user can reply 'continue'.")
        msgs = messages + [{"role": "user", "content": f"[system note] {note}"}]
        for name, caller in self._ordered_chain():
            try:
                r = caller(msgs if name == "deepseek" else self._strip_reasoning(msgs), tools=None)
                if r and (r.get("content") or "").strip():
                    return r["content"].strip()
            except Exception as e:
                logger.warning(f"[{name}] cap summary failed: {e}")
        done = "\n\n".join(preface)[-1500:]
        return ((done + "\n\n") if done else "") + (
            f"⏸ Step limit reached ({max_rounds}). Reply 'continue' to resume.")

    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio via Groq Whisper."""
        if not self.groq_client:
            return None
        with open(file_path, "rb") as f:
            result = self.groq_client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=f,
                response_format="text",
            )
        return result.text.strip() if hasattr(result, "text") else str(result)
