"""Configuration loader: every setting comes from environment variables (or .env)."""
import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Centralized configuration with defaults."""

    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        load_dotenv(self.base_dir / ".env")

        # Access control: comma-separated chat IDs allowed to use tools (empty = everyone is a guest)
        self.allowed_ids: list[str] = [x.strip() for x in os.getenv("ALLOWED_IDS", "").split(",") if x.strip()]

        # HTTP server
        self.host: str = os.getenv("RELAY_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("RELAY_PORT", "8080"))
        self.data_dir: Path = Path(os.getenv("RELAY_DATA_DIR", str(self.base_dir / "data")))

        # LLM providers
        self.do_api_key: str = os.getenv("DO_AI_API_KEY", "")
        self.do_base_url: str = os.getenv("DO_AI_BASE_URL", "https://api.do-ai.run/api/v1")
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
        self.anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

        # Qdrant
        self.qdrant_host: str = os.getenv("QDRANT_HOST", "127.0.0.1")
        self.qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

        # Paths
        self.history_file: str = os.getenv("HISTORY_FILE", str(Path(os.getenv("RELAY_DATA_DIR", str(self.base_dir / "data"))) / "history.json"))

        # Limits
        self.max_history: int = int(os.getenv("MAX_HISTORY", "30"))
        self.web_search_timeout: int = int(os.getenv("WEB_SEARCH_TIMEOUT", "15"))


config = Config()
