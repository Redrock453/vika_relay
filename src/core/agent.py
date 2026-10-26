"""Core agent: system prompt + RAG context + LLM fallback chain with function calling."""
import functools
import logging
import os
from pathlib import Path

from src.core.history import HistoryManager
from src.core.llm import LLMProvider
from src.services.rag import RAGService
from src.services.search import web_search
from src.services.tool_defs import TOOLS

logger = logging.getLogger("vika.agent")

# Access mode defines the tool allowlist (deny by default for anything but "owner").
GUEST_TOOLS = {"web_search"}
DEFAULT_PROMPT = ("You are a helpful assistant. Answer concisely in the user's language. "
                  "Use tools when they help.")
OWNER_MAX_ROUNDS = int(os.getenv("VIKA_MAX_TOOL_ROUNDS", "20"))
GUEST_MAX_ROUNDS = int(os.getenv("VIKA_GUEST_TOOL_ROUNDS", "5"))
PROMPTS_DIR = Path(os.getenv("RELAY_PROMPTS_DIR", "prompts"))


class Agent:
    """Wraps LLMProvider with history, RAG and tools."""

    def __init__(self, llm=None, history=None, rag=None):
        self.llm = llm or LLMProvider()
        self.history = history or HistoryManager()
        self.rag = rag or RAGService()
        self.prompts = {}
        if PROMPTS_DIR.is_dir():
            for p in PROMPTS_DIR.glob("*.txt"):
                self.prompts[p.stem] = p.read_text(encoding="utf-8")

    def system_prompt(self, mode: str) -> str:
        return self.prompts.get(mode) or self.prompts.get("default") or DEFAULT_PROMPT

    def _execute_tool(self, name: str, args: dict, mode: str = "owner") -> str:
        if mode != "owner" and name not in GUEST_TOOLS:
            return f"Tool {name} is not available in this mode."
        if name == "web_search":
            return web_search(args.get("query", ""))
        if name == "rag_search":
            return self.rag.search(args.get("query", "")) or "Nothing relevant found."
        return f"Unknown tool: {name}"

    def build_messages(self, query: str, user_id: str, mode: str) -> list[dict]:
        guest = mode != "owner"
        messages = [{"role": "system", "content": self.system_prompt("guest" if guest else "default")}]
        messages.extend(self.history.recent(user_id))
        # KB context goes at the end of the user turn to keep the prompt-cache prefix stable.
        rag_context = "" if guest else self.rag.search(query)
        user_content = query
        if rag_context:
            user_content = (
                "KNOWLEDGE BASE CONTEXT (may be irrelevant; not part of the dialogue). "
                "Use only facts related to the question; do not quote it verbatim.\n"
                f"<kb>\n{rag_context}\n</kb>\n\n{query}")
        messages.append({"role": "user", "content": user_content})
        return messages

    def ask(self, query: str, user_id: str = "default", mode: str = "guest") -> str:
        mode = "owner" if mode == "owner" else "guest"
        messages = self.build_messages(query, user_id, mode)
        tools = TOOLS if mode == "owner" else [t for t in TOOLS if t["function"]["name"] in GUEST_TOOLS]
        executor = functools.partial(self._execute_tool, mode=mode)
        max_rounds = OWNER_MAX_ROUNDS if mode == "owner" else GUEST_MAX_ROUNDS
        response = self.llm.ask(messages, tool_executor=executor, tools=tools, max_rounds=max_rounds)
        self.history.add(user_id, "user", query)
        self.history.add(user_id, "assistant", response)
        return response

    def new_chat(self, user_id: str):
        self.history.clear(user_id)
