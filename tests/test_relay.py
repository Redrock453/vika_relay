"""Offline tests: no network, no API keys."""
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_tmp = tempfile.mkdtemp()
os.environ["RELAY_DATA_DIR"] = _tmp
os.environ["HISTORY_FILE"] = os.path.join(_tmp, "history.json")
os.environ["ALLOWED_IDS"] = "owner1"
for k in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "DO_AI_API_KEY",
          "GROQ_API_KEY", "GEMINI_API_KEY"):
    os.environ[k] = ""

from src.core import llm as llm_mod  # noqa: E402
from src.core.agent import Agent  # noqa: E402
from src.core.history import HistoryManager  # noqa: E402
import turn_server  # noqa: E402


class FakeRAG:
    def search(self, q, limit=3):
        return ""


class FakeLLM:
    def __init__(self):
        self.turns, self.calls = [], []

    def set_turn(self, **kw):
        self.turns.append(kw)

    def clear_turn(self):
        pass

    def ask(self, messages, tool_executor=None, tools=None, max_rounds=10):
        self.calls.append({"tools": [t["function"]["name"] for t in tools or []], "max_rounds": max_rounds})
        return "ok:" + messages[-1]["content"]


def brain():
    agent = Agent(llm=FakeLLM(), history=HistoryManager(), rag=FakeRAG())
    return turn_server.Brain(agent=agent, state_path=Path(_tmp) / f"state{id(agent)}.json")


class TestPeak(unittest.TestCase):
    def test_peak_hours(self):
        self.assertTrue(llm_mod.peak_now_utc(datetime(2026, 10, 26, 2, 0, tzinfo=timezone.utc)))   # Monday
        self.assertFalse(llm_mod.peak_now_utc(datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)))
        self.assertFalse(llm_mod.peak_now_utc(datetime(2026, 10, 25, 2, 0, tzinfo=timezone.utc)))  # Sunday


class TestChain(unittest.TestCase):
    def test_no_keys_all_unavailable(self):
        p = llm_mod.LLMProvider()
        self.assertEqual([n for n, _ in p.chain], ["deepseek", "openrouter", "claude", "do", "groq", "gemini"])
        self.assertIn("unavailable", p.ask([{"role": "user", "content": "hi"}]))

    def test_provider_first(self):
        p = llm_mod.LLMProvider()
        p.set_turn(provider="gemini")
        self.assertEqual(p._ordered_chain()[0][0], "gemini")
        p.clear_turn()


class TestBrain(unittest.TestCase):
    def test_help(self):
        self.assertIn("/model", brain().handle({"chat_id": "x", "text": "/help"})["text"])

    def test_model_switch_owner_only(self):
        b = brain()
        self.assertIn("owner only", b.handle({"chat_id": "guest", "text": "/model pro"})["text"])
        self.assertIn("deepseek-v4-pro", b.handle({"chat_id": "owner1", "text": "/model pro"})["text"])
        b.handle({"chat_id": "owner1", "text": "hello"})
        self.assertEqual(b.agent.llm.turns[-1]["provider"], "deepseek")

    def test_guest_tools_restricted(self):
        b = brain()
        b.handle({"chat_id": "guest", "text": "hi"})
        b.handle({"chat_id": "owner1", "text": "hi"})
        self.assertEqual(b.agent.llm.calls[0]["tools"], ["web_search"])
        self.assertIn("rag_search", b.agent.llm.calls[1]["tools"])

    def test_guest_tool_executor_denies(self):
        a = Agent(llm=FakeLLM(), history=HistoryManager(), rag=FakeRAG())
        self.assertIn("not available", a._execute_tool("rag_search", {"query": "x"}, mode="guest"))


if __name__ == "__main__":
    unittest.main()
