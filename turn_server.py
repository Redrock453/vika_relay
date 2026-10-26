"""HTTP front for the router: POST /turn {"chat_id", "text"} -> {"text"}; GET /health.

Slash commands: /help, /model [key], /new, /status. The "local" model runs on Ollama
and never leaves the host.
"""
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import requests

from src.core.agent import Agent
from src.core.config import config
from src.core.llm import peak_now_utc
from src.core.sessions import SessionStore

logger = logging.getLogger("vika.server")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
LOCAL_MODEL = os.environ.get("VIKA_LOCAL_MODEL", "hermes3:8b")
DS_FLASH = os.environ.get("VIKA_DEEPSEEK_MODEL_FLASH", "deepseek-flash")
DS_PRO = os.environ.get("VIKA_DEEPSEEK_MODEL_PRO", "deepseek-v4-pro")
OR_FLASH = os.environ.get("VIKA_MODEL_FLASH", "deepseek/deepseek-v4-flash")
OR_PRO = os.environ.get("VIKA_MODEL_PRO", "deepseek/deepseek-v4-pro")

CHOICES = {
    "flash": {"provider": "deepseek", "ds": DS_FLASH, "or": OR_FLASH, "label": f"{DS_FLASH} - fast and cheap"},
    "pro": {"provider": "deepseek", "ds": DS_PRO, "or": OR_PRO, "label": f"{DS_PRO} - strongest, slower"},
    "groq": {"provider": "groq", "ds": DS_FLASH, "or": OR_FLASH,
             "label": os.environ.get("VIKA_GROQ_MODEL", "openai/gpt-oss-120b") + " (Groq)"},
    "gemini": {"provider": "gemini", "ds": DS_FLASH, "or": OR_FLASH,
               "label": os.environ.get("VIKA_GEMINI_MODEL", "gemini-2.5-flash") + " (Google)"},
    "claude": {"provider": "claude", "ds": DS_FLASH, "or": OR_FLASH, "label": "Claude (Anthropic)"},
    "local": {"provider": "local", "label": f"{LOCAL_MODEL} - local Ollama, offline, no tools"},
}
ALIASES = {"deepseek-flash": "flash", "deepseek-v4-pro": "pro", "gpt-oss": "groq",
           "hermes": "local", "ollama": "local"}
DEFAULT_MODEL_KEY = ALIASES.get(os.environ.get("VIKA_DEFAULT_MODEL", "flash").strip().lower(),
                                os.environ.get("VIKA_DEFAULT_MODEL", "flash").strip().lower())
if DEFAULT_MODEL_KEY not in CHOICES:
    DEFAULT_MODEL_KEY = "flash"

HELP_TEXT = ("Commands:\n/model - list models, /model <key> - switch\n"
             "/new - start a new session\n/status - current model and uptime\n/help - this text")
STARTED = time.time()


class Brain:
    def __init__(self, agent=None, sessions=None, state_path=None) -> None:
        self.agent = agent or Agent()
        config.data_dir.mkdir(parents=True, exist_ok=True)
        self.sessions = sessions or SessionStore(config.data_dir / "sessions.json", self.agent.history)
        self.state_path = state_path or (config.data_dir / "state.json")
        self._lock = threading.Lock()
        try:
            self.models = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            self.models = {}

    def _save_state(self) -> None:
        try:
            self.state_path.write_text(json.dumps(self.models), encoding="utf-8")
        except Exception as e:
            logger.warning(f"state save failed: {e}")

    def is_owner(self, chat_id: str) -> bool:
        return str(chat_id) in config.allowed_ids

    def model_key(self, chat_id: str) -> str:
        return self.models.get(str(chat_id), DEFAULT_MODEL_KEY)

    def _model_cmd(self, chat_id: str, arg: str) -> str:
        if not arg:
            cur = self.model_key(chat_id)
            return "\n".join(f"{'*' if k == cur else ' '} {k}: {v['label']}" for k, v in CHOICES.items())
        key = ALIASES.get(arg, arg)
        if key not in CHOICES:
            return f"Unknown model: {arg}"
        with self._lock:
            self.models[str(chat_id)] = key
            self._save_state()
        return f"Model: {CHOICES[key]['label']}"

    def _ask_local(self, hkey: str, text: str) -> str:
        msgs = [{"role": "system", "content": self.agent.system_prompt("default")}]
        msgs += self.agent.history.recent(hkey) + [{"role": "user", "content": text}]
        r = requests.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": LOCAL_MODEL, "messages": msgs, "stream": False}, timeout=300)
        r.raise_for_status()
        reply = (r.json().get("message", {}).get("content") or "").strip()
        self.agent.history.add(hkey, "user", text)
        self.agent.history.add(hkey, "assistant", reply)
        return reply

    def handle(self, req: dict) -> dict:
        chat_id = str(req.get("chat_id", "default"))
        text = str(req.get("text", "")).strip()
        low = text.lower()
        owner = self.is_owner(chat_id)
        if low in ("/help", "/start", "help"):
            return {"text": HELP_TEXT}
        if low.startswith("/model"):
            if not owner:
                return {"text": "This command is available to the owner only."}
            return {"text": self._model_cmd(chat_id, low[len("/model"):].strip())}
        if low == "/new":
            sid, _ = self.sessions.new(chat_id)
            return {"text": f"New session {sid}."}
        if low == "/status":
            up = int(time.time() - STARTED)
            return {"text": f"Model: {CHOICES[self.model_key(chat_id) if owner else DEFAULT_MODEL_KEY]['label']}\n"
                            f"Uptime: {up // 3600}h {up % 3600 // 60}m\nDeepSeek peak: {'yes' if peak_now_utc() else 'no'}"}
        if not text:
            return {"text": ""}
        key = self.model_key(chat_id) if owner else DEFAULT_MODEL_KEY
        choice = CHOICES[key]
        sid, _ = self.sessions.current(chat_id)
        hkey = self.sessions.hist_key(chat_id, sid)
        if choice["provider"] == "local":
            reply = self._ask_local(hkey, text)
        else:
            self.agent.llm.set_turn(provider=choice["provider"], deepseek_model=choice.get("ds"),
                                    openrouter_model=choice.get("or"))
            try:
                reply = self.agent.ask(text, user_id=hkey, mode="owner" if owner else "guest")
            finally:
                self.agent.llm.clear_turn()
        self.sessions.touch(chat_id, sid, text)
        return {"text": reply}


def make_handler(brain: Brain):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/turn":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                self._send(200, brain.handle(req))
            except Exception as e:
                logger.exception("turn failed")
                self._send(500, {"error": str(e)})

        def log_message(self, fmt, *args):
            pass
    return Handler


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    brain = Brain()
    srv = ThreadingHTTPServer((config.host, config.port), make_handler(brain))
    logger.info(f"vika_relay listening on {config.host}:{config.port}")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
