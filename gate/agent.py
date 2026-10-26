"""Agent: system prompt + knowledge-base context + history + tools, routed through LLMRouter.

Access is set by mode, not by the prompt. Each mode has a tool allowlist enforced twice:
in the tool list the model sees and again in the executor. Unknown modes get guest rights
(deny by default). Guests get web search only and never see the knowledge base.
"""
import functools
import logging
import os
from pathlib import Path
from typing import Optional

from gate.config import config
from gate.history import HistoryManager
from gate.rag import RAGService
from gate.router import LLMRouter
from gate.search import web_search
from gate.tools import TOOLS

logger = logging.getLogger("gate.agent")

OWNER_TOOLS = {"web_search", "rag_search"}
GUEST_TOOLS = {"web_search"}
MODE_TOOLS = {"owner": OWNER_TOOLS, "guest": GUEST_TOOLS}
OWNER_MAX_ROUNDS = int(os.getenv("VIKA_MAX_TOOL_ROUNDS", "60"))
GUEST_MAX_ROUNDS = int(os.getenv("VIKA_GUEST_TOOL_ROUNDS", "20"))

GUEST_PROMPT_FALLBACK = ("You are Vika, a friendly assistant. Answer briefly, in the user's language. "
                         "You can search the web. Never share anything about the owner, servers or files.")
GUEST_NOTE = ("GUEST MODE: only web_search is available to you. You have no files, commands or servers — "
              "do not promise them and do not name internal tools. Decline such requests in one sentence.")
DENIED = "⛔ Ця дія недоступна в гостьовому режимі."
# Knowledge-base context goes at the end of the user turn, not right after the system prompt:
# that keeps the prompt prefix (system + tools + history) stable, so provider-side prompt
# caching keeps hitting even though the retrieved context changes every turn.
KB_WRAPPER = ("KNOWLEDGE BASE NOTES (may be irrelevant; these are NOT dialogue turns). Use only facts that "
              "relate to the question. Never quote, continue or retell these fragments in full, and do not "
              "output file paths or JSON from them.\n<kb>\n{kb}\n</kb>\n\n{query}")


class Agent:
    """One entry point for a chat turn: ask(query, user_id, mode)."""

    def __init__(self, router: Optional[LLMRouter] = None, rag: Optional[RAGService] = None,
                 history: Optional[HistoryManager] = None, prompts_dir: Optional[str] = None):
        self.router = router or LLMRouter()
        self.rag = rag or RAGService()
        self.history = history or HistoryManager()
        pdir = Path(prompts_dir) if prompts_dir else config.base_dir / "prompts"
        self.prompts = {p.stem: p.read_text(encoding="utf-8") for p in pdir.glob("*.txt")}

    def _execute_tool(self, name: str, args: dict, mode: str = "guest") -> str:
        """Run a tool call. The mode's allowlist is checked here, not only in the visible tool list."""
        if name not in MODE_TOOLS.get(mode, GUEST_TOOLS):
            logger.warning(f"Tool {name} denied for mode={mode}")
            return DENIED
        try:
            if name == "web_search":
                return web_search(args["query"])
            if name == "rag_search":
                return self.rag.search(args["query"])
            return f"❌ Unknown tool: {name}"
        except Exception as e:
            logger.error(f"Tool {name} error: {e}")
            return f"❌ Error: {e}"

    def ask(self, query: str, user_id: str = "default", mode: str = "owner") -> str:
        mode = mode if mode in MODE_TOOLS else "guest"
        guest = mode == "guest"
        allowed = MODE_TOOLS[mode]

        if guest:
            system_prompt = self.prompts.get("guest") or GUEST_PROMPT_FALLBACK
        else:
            system_prompt = self.prompts.get("owner", "")
        messages = [{"role": "system", "content": system_prompt}]
        if guest:
            messages.append({"role": "system", "content": GUEST_NOTE})
        messages.extend(self.history.recent(user_id))

        rag_context = "" if guest else self.rag.search(query)
        user_content = KB_WRAPPER.format(kb=rag_context, query=query) if rag_context else query
        messages.append({"role": "user", "content": user_content})

        response = self.router.ask(
            messages,
            tool_executor=functools.partial(self._execute_tool, mode=mode),
            tools=[t for t in TOOLS if t["function"]["name"] in allowed],
            max_rounds=GUEST_MAX_ROUNDS if guest else OWNER_MAX_ROUNDS,
        )

        self.history.add(user_id, "user", query)
        self.history.add(user_id, "assistant", response)
        return response

    def transcribe(self, file_path: str) -> Optional[str]:
        return self.router.transcribe_audio(file_path)

    def new_chat(self, user_id: str):
        self.history.clear(user_id)
