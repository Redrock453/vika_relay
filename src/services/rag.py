"""RAG: Ollama embeddings (bge-m3) + Qdrant vector search.

Any embedder/Qdrant error returns an empty string, so the agent keeps working without RAG.
"""
import logging
import os

import requests

from src.core.config import config

logger = logging.getLogger("vika.rag")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("VIKA_EMBED_MODEL", "bge-m3")
QDRANT_URL = os.getenv("QDRANT_URL", f"http://{config.qdrant_host}:{config.qdrant_port}")
COLLECTION = os.getenv("VIKA_RAG_COLLECTION", "knowledge")
TOP_K = int(os.getenv("VIKA_RAG_TOPK", "3"))
SCORE_THRESHOLD = float(os.getenv("VIKA_RAG_THRESHOLD", "0.50"))
SNIPPET_CHARS = int(os.getenv("VIKA_RAG_SNIPPET", "500"))


class RAGService:
    """Semantic search over a Qdrant collection (embeddings via local Ollama)."""

    def __init__(self):
        self._ready = True

    def _embed(self, text: str):
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text, "keep_alive": 0},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embeddings"][0]

    def search(self, query: str, limit: int = 3) -> str:
        if len(query.strip()) < 20:
            return ""
        try:
            vec = self._embed(query)
        except Exception as e:
            logger.warning(f"embedding failed: {e}")
            return ""
        try:
            r = requests.post(
                f"{QDRANT_URL}/collections/{COLLECTION}/points/search",
                json={"vector": vec, "limit": limit, "with_payload": True,
                      "score_threshold": SCORE_THRESHOLD},
                timeout=10,
            )
            r.raise_for_status()
            hits = r.json().get("result", [])
        except Exception as e:
            logger.warning(f"qdrant search failed: {e}")
            return ""
        parts = [
            "Relevant knowledge-base snippets below. Use them only if they relate to the question; do not invent facts beyond them."
        ]
        for h in hits:
            text = (h.get("payload", {}).get("text") or "").strip()
            if text:
                src = h.get("payload", {}).get("source", "?")
                parts.append(f"[{src} score={float(h.get('score', 0)):.3f}] {text[:SNIPPET_CHARS]}")
        return "\n\n".join(parts) if len(parts) > 1 else ""
