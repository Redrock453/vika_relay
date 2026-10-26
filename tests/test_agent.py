"""Access-mode and context-assembly tests with a fake router and knowledge base."""
import pytest

from gate.agent import DENIED, Agent
from gate.history import HistoryManager
from gate.rag import RAGService


class FakeRouter:
    def __init__(self, reply="ok"):
        self.reply = reply
        self.calls = []

    def ask(self, messages, tool_executor=None, tools=None, max_rounds=10):
        self.calls.append({"messages": [dict(m) for m in messages], "executor": tool_executor,
                           "tools": tools, "max_rounds": max_rounds})
        return self.reply


class FakeRAG:
    def __init__(self, text=""):
        self.text = text
        self.queries = []

    def search(self, query, limit=3):
        self.queries.append(query)
        return self.text


@pytest.fixture
def make_agent(tmp_path):
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    (prompts / "owner.txt").write_text("OWNER PROMPT", encoding="utf-8")
    (prompts / "guest.txt").write_text("GUEST PROMPT", encoding="utf-8")

    def _make(rag_text=""):
        router, rag = FakeRouter(), FakeRAG(rag_text)
        agent = Agent(router=router, rag=rag, history=HistoryManager(str(tmp_path / "history.json")),
                      prompts_dir=str(prompts))
        return agent, router, rag

    return _make


def tool_names(tools):
    return {t["function"]["name"] for t in tools}


def test_guest_sees_only_web_search_and_no_knowledge_base(make_agent):
    agent, router, rag = make_agent(rag_text="private notes")
    agent.ask("tell me about the owner's servers please", "u1", mode="guest")
    call = router.calls[0]
    assert tool_names(call["tools"]) == {"web_search"}
    assert rag.queries == []
    assert call["messages"][0]["content"] == "GUEST PROMPT"
    assert "private notes" not in str(call["messages"])


def test_guest_executor_denies_tools_outside_allowlist(make_agent):
    agent, router, _ = make_agent()
    agent.ask("can you look something up for me", "u1", mode="guest")
    assert router.calls[0]["executor"]("rag_search", {"query": "x"}) == DENIED


def test_unknown_mode_gets_guest_rights(make_agent):
    agent, router, _ = make_agent()
    agent.ask("some question that is long enough", "u1", mode="admin")
    assert tool_names(router.calls[0]["tools"]) == {"web_search"}


def test_owner_gets_both_tools_and_kb_at_end_of_user_turn(make_agent):
    agent, router, _ = make_agent(rag_text="FACT: the chain has five providers")
    agent.ask("how many providers are in the chain?", "u1", mode="owner")
    call = router.calls[0]
    assert tool_names(call["tools"]) == {"web_search", "rag_search"}
    assert call["messages"][0]["content"] == "OWNER PROMPT"
    last = call["messages"][-1]
    assert last["role"] == "user"
    assert "FACT: the chain has five providers" in last["content"]
    assert last["content"].endswith("how many providers are in the chain?")


def test_history_is_kept_and_cleared(make_agent):
    agent, router, _ = make_agent()
    agent.ask("first question for history", "u1")
    agent.ask("second question for history", "u1")
    assert {"role": "assistant", "content": "ok"} in router.calls[1]["messages"]
    agent.new_chat("u1")
    assert agent.history.recent("u1") == []


def test_rag_skips_short_messages_without_network():
    assert RAGService().search("привіт") == ""
