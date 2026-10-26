"""Per-chat sessions: each chat has a list of sessions, each with its own history key.

Key format: {chat_id} for the first session, {chat_id}#{sid} for later ones.
Metadata (title, timestamps, counters) is stored in a JSON file.
"""
import json
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

LEGACY_SID = "s0"


class SessionStore:
    def __init__(self, path: Path, history) -> None:
        self.path = Path(path)
        self.history = history
        self._lock = threading.RLock()
        self._data: dict = {}
        self._load()

    # ── persistence ──
    def _load(self) -> None:
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    # ── helpers ──
    @staticmethod
    def hist_key(chat_id: str, sid: str) -> str:
        return f"{chat_id}" if sid == LEGACY_SID else f"{chat_id}#{sid}"

    def _chat(self, chat_id: str) -> dict:
        """Chat state; on first access existing history becomes the first session."""
        ch = self._data.get(chat_id)
        if ch is None:
            now = time.time()
            n = len(self.history.get(f"wa:{chat_id}"))
            ch = {"current": LEGACY_SID, "sessions": {LEGACY_SID: {
                "title": "Earlier conversation" if n else None, "titled": bool(n),
                "created": now, "updated": now, "count": n}}}
            self._data[chat_id] = ch
            self._save()
        return ch

    def current(self, chat_id: str) -> tuple[str, dict]:
        with self._lock:
            ch = self._chat(chat_id)
            sid = ch["current"]
            if sid not in ch["sessions"]:
                sid = self._new_locked(chat_id)
            return sid, dict(ch["sessions"][sid])

    def current_key(self, chat_id: str) -> str:
        sid, _ = self.current(chat_id)
        return self.hist_key(chat_id, sid)

    def messages(self, chat_id: str, sid: str) -> list[dict]:
        """Messages of a session from HistoryManager."""
        return self.history.get(self.hist_key(chat_id, sid))

    def ordered(self, chat_id: str) -> list[tuple[str, dict]]:
        """Sessions ordered by last activity (newest first)."""
        with self._lock:
            ch = self._chat(chat_id)
            return sorted(((s, dict(m)) for s, m in ch["sessions"].items()),
                          key=lambda x: x[1].get("updated", 0), reverse=True)

    def _new_locked(self, chat_id: str) -> str:
        ch = self._chat(chat_id)
        sid = "s" + secrets.token_hex(3)
        now = time.time()
        ch["sessions"][sid] = {"title": None, "titled": False, "created": now, "updated": now, "count": 0}
        ch["current"] = sid
        self._save()
        return sid

    def new(self, chat_id: str) -> tuple[str, Optional[dict]]:
        """Create a new session; returns (sid, metadata of the previous current one)."""
        with self._lock:
            ch = self._chat(chat_id)
            prev = ch["sessions"].get(ch["current"])
            prev = dict(prev) if prev else None
            if prev and prev.get("count", 0) == 0:
                return ch["current"], None
            return self._new_locked(chat_id), prev

    def switch(self, chat_id: str, sid: str) -> dict:
        with self._lock:
            ch = self._chat(chat_id)
            ch["current"] = sid
            ch["sessions"][sid]["updated"] = time.time()
            self._save()
            return dict(ch["sessions"][sid])

    def rename(self, chat_id: str, title: str) -> None:
        with self._lock:
            ch = self._chat(chat_id)
            m = ch["sessions"][ch["current"]]
            m["title"], m["titled"] = title, True
            self._save()

    def set_title_if_untitled(self, chat_id: str, sid: str, title: str) -> None:
        with self._lock:
            m = self._chat(chat_id)["sessions"].get(sid)
            if m is not None and not m.get("titled"):
                m["title"], m["titled"] = title, True
                self._save()

    def delete(self, chat_id: str, sid: str) -> dict:
        with self._lock:
            ch = self._chat(chat_id)
            meta = ch["sessions"].pop(sid)
            self._save()
        self.history.clear(self.hist_key(chat_id, sid))
        return meta

    def reset_current(self, chat_id: str) -> dict:
        with self._lock:
            ch = self._chat(chat_id)
            sid = ch["current"]
            m = ch["sessions"][sid]
            old = dict(m)
            m.update({"title": None, "titled": False, "count": 0, "updated": time.time()})
            self._save()
        self.history.clear(self.hist_key(chat_id, sid))
        return old

    def touch(self, chat_id: str, sid: str, first_text: str = "") -> bool:
        """After a turn: +2 messages. Returns True on the first exchange (time to name the session)."""
        with self._lock:
            m = self._chat(chat_id)["sessions"].get(sid)
            if m is None:
                return False
            m["count"] = int(m.get("count", 0)) + 2
            m["updated"] = time.time()
            need = not m.get("titled")
            if need and not m.get("title") and first_text:
                m["title"] = _short(first_text)
            self._save()
            return need


def _short(text: str, n: int = 40) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"
