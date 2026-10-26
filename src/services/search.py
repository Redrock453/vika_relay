"""Web search via local SearXNG instance."""
import logging
import os
import requests
from src.core.config import config

logger = logging.getLogger("vika.search")

def web_search(query: str) -> str:
    """Search local SearXNG and return combined results."""
    searx_url = os.getenv("SEARXNG_URL", "http://127.0.0.1:8888")
    try:
        # SearXNG API request
        response = requests.get(
            f"{searx_url}/search",
            params={
                "q": query,
                "format": "json",
                "engines": os.getenv("SEARXNG_ENGINES", "google,duckduckgo,bing"),
                "language": os.getenv("SEARXNG_LANGUAGE", "en-US")
            },
            timeout=config.web_search_timeout
        )
        
        if response.status_code != 200:
            logger.error(f"SearXNG error: {response.status_code}")
            return f"❌ SearXNG error: {response.status_code}"
            
        results = response.json().get("results", [])
        if not results:
            return "🔍 No results."
            
        formatted_results = []
        for res in results[:5]:  # Top 5 results
            title = res.get("title", "No title")
            content = res.get("content", "No content")
            url = res.get("url", "")
            formatted_results.append(f"🔹 {title}\n{content}\nURL: {url}")
            
        return "\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"❌ Search error: {e}"
