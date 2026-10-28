"""Configuration from environment variables (.env in the repo root)."""
import os
from pathlib import Path

from dotenv import load_dotenv


class Config:
    """Centralized configuration with defaults."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent
        load_dotenv(self.base_dir / ".env")

        # Telegram
        self.telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.allowed_ids: list[int] = self._parse_ids(os.getenv("ALLOWED_IDS", ""))

        # LLM providers (DeepSeek and OpenRouter keys are read by the router directly)
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")

        # Qdrant
        self.qdrant_host: str = os.getenv("QDRANT_HOST", "127.0.0.1")
        self.qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

        # Storage and limits
        self.history_file: str = os.getenv("HISTORY_FILE", str(self.base_dir / "data" / "history.json"))
        self.max_history: int = int(os.getenv("MAX_HISTORY", "30"))
        self.web_search_timeout: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))

    @staticmethod
    def _parse_ids(raw: str) -> list[int]:
        return [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]


config = Config()
