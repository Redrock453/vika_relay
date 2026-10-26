"""Chat history management with persistence."""
import json
import logging
import threading
from typing import Optional

from src.core.config import config

logger = logging.getLogger("vika.history")


class HistoryManager:
    """Thread-safe chat history with JSON persistence."""

    def __init__(self):
        self._histories: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            with open(config.history_file, "r") as f:
                self._histories = json.load(f)
            logger.info(f"Loaded history for {len(self._histories)} users")
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"History load error: {e}")

    def _save(self):
        try:
            with open(config.history_file, "w") as f:
                json.dump(self._histories, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"History save error: {e}")

    def get(self, user_id: str) -> list[dict]:
        with self._lock:
            return list(self._histories.get(str(user_id), []))

    def add(self, user_id: str, role: str, content: str):
        with self._lock:
            uid = str(user_id)
            if uid not in self._histories:
                self._histories[uid] = []
            self._histories[uid].append({"role": role, "content": content})
            # Trim to 2x max
            if len(self._histories[uid]) > config.max_history * 2:
                self._histories[uid] = self._histories[uid][-config.max_history * 2:]
            self._save()

    def clear(self, user_id: str) -> Optional[str]:
        """Clear history for user, return last context summary or None."""
        with self._lock:
            uid = str(user_id)
            removed = self._histories.pop(uid, [])
            self._save()
            return None

    def recent(self, user_id: str, limit: Optional[int] = None) -> list[dict]:
        limit = limit or config.max_history
        return self.get(user_id)[-limit:]
