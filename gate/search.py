"""Web search via a SearXNG instance (JSON API)."""
import logging
import os

import requests

from gate.config import config

logger = logging.getLogger("gate.search")

SEARXNG_URL = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
SEARCH_LANG = os.getenv("VIKA_SEARCH_LANG", "ru-RU")
MAX_RESULTS = 5


def web_search(query: str) -> str:
    """Return the top results as plain text for the model."""
    try:
        response = requests.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "engines": "google,duckduckgo,bing",
                    "language": SEARCH_LANG},
            timeout=config.web_search_timeout,
        )
        if response.status_code != 200:
            logger.error(f"SearXNG error: {response.status_code}")
            return f"Search error: HTTP {response.status_code}"
        results = response.json().get("results", [])
        if not results:
            return "No results found."
        return "\n\n".join(
            f"- {r.get('title', 'No title')}\n{r.get('content', '')}\nURL: {r.get('url', '')}"
            for r in results[:MAX_RESULTS]
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search error: {e}"
