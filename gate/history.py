"""Per-user chat history with JSON persistence."""
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from gate.config import config

logger = logging.getLogger("gate.history")


class HistoryManager:
    """Thread-safe chat history, persisted to a JSON file after every change."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.history_file)
        self._histories: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            self._histories = json.loads(self.path.read_text(encoding="utf-8"))
            logger.info(f"Loaded history for {len(self._histories)} users")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"History load error: {e}")

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._histories, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"History save error: {e}")

    def get(self, user_id: str) -> list[dict]:
        with self._lock:
            return list(self._histories.get(str(user_id), []))

    def add(self, user_id: str, role: str, content: str):
        with self._lock:
            uid = str(user_id)
            self._histories.setdefault(uid, []).append({"role": role, "content": content})
            # Keep at most twice the context window on disk.
            if len(self._histories[uid]) > config.max_history * 2:
                self._histories[uid] = self._histories[uid][-config.max_history * 2:]
            self._save()

    def clear(self, user_id: str):
        with self._lock:
            self._histories.pop(str(user_id), None)
            self._save()

    def recent(self, user_id: str, limit: Optional[int] = None) -> list[dict]:
        return self.get(user_id)[-(limit or config.max_history):]
