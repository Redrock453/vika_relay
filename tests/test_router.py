"""Routing logic tests with scripted fake providers — no API keys or network needed."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import gate.router as router_mod
from gate.router import (MSG_ALL_DOWN, MSG_STOPPED, TRUNCATED_ARGS, LLMRouter, peak_now_utc,
                         peak_skip_deepseek)

USER = [{"role": "user", "content": "hi"}]


class APIError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class FakeProvider:
    """Replays scripted outcomes in order (the last one repeats). Exceptions are raised."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, messages, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, Exception):
            raise out
        return out


def answer(text):
    return {"content": text}


def tool_call(name, args, id_="call_1"):
    return SimpleNamespace(id=id_, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(router_mod.time, "sleep", lambda s: None)


@pytest.fixture
def router(monkeypatch):
    for key in ("DEEPSEEK_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    for attr in ("do_api_key", "groq_api_key", "gemini_api_key"):
        monkeypatch.setattr(router_mod.config, attr, "")
    monkeypatch.setattr(router_mod, "peak_now_utc", lambda now=None: False)
    return LLMRouter()


def test_falls_back_after_retries(router):
    a, b = FakeProvider(RuntimeError("timeout")), FakeProvider(answer("hi from b"))
    router.chain = [("a", a), ("b", b)]
    assert router.ask(list(USER)) == "hi from b"
    assert len(a.calls) == router_mod.MAX_RETRIES
    assert len(b.calls) == 1


@pytest.mark.parametrize("code", [401, 402, 403, 404])
def test_auth_and_billing_errors_skip_provider_without_retries(router, code):
    a, b = FakeProvider(APIError(code)), FakeProvider(answer("ok"))
    router.chain = [("a", a), ("b", b)]
    assert router.ask(list(USER)) == "ok"
    assert len(a.calls) == 1


def test_server_errors_are_retried(router):
    a = FakeProvider(APIError(503), answer("recovered"))
    router.chain = [("a", a)]
    assert router.ask(list(USER)) == "recovered"
    assert len(a.calls) == 2


def test_empty_answer_counts_as_failure(router):
    a, b = FakeProvider(answer("   ")), FakeProvider(answer("real answer"))
    router.chain = [("a", a), ("b", b)]
    assert router.ask(list(USER)) == "real answer"
    assert len(a.calls) == router_mod.MAX_RETRIES


def test_all_providers_down(router):
    router.chain = [("a", FakeProvider(RuntimeError("down")))]
    assert router.ask(list(USER)) == MSG_ALL_DOWN


def test_providers_without_keys_are_skipped(router):
    assert router.ask(list(USER)) == MSG_ALL_DOWN


def test_preferred_provider_goes_first(router):
    a, b = FakeProvider(answer("from a")), FakeProvider(answer("from b"))
    router.chain = [("a", a), ("b", b)]
    router.set_turn(provider="b")
    assert router.ask(list(USER)) == "from b"
    assert a.calls == []


def test_cancel_stops_the_turn(router):
    router.chain = [("a", FakeProvider(answer("never")))]
    router.set_turn(cancel_check=lambda: True)
    assert router.ask(list(USER)) == MSG_STOPPED


def test_tool_loop_executes_tool_and_returns_final_answer(router):
    p = FakeProvider({"content": "", "tool_calls": [tool_call("web_search", {"query": "weather"})]},
                     answer("It is sunny."))
    router.chain = [("p", p)]
    executed = []

    def executor(name, args):
        executed.append((name, args))
        return "sunny, +20"

    msgs = [{"role": "user", "content": "weather?"}]
    assert router.ask(msgs, tool_executor=executor) == "It is sunny."
    assert executed == [("web_search", {"query": "weather"})]
    assert msgs[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny, +20"}


def test_truncated_tool_arguments_are_reported_back(router):
    broken = SimpleNamespace(id="c1", function=SimpleNamespace(name="web_search", arguments='{"query": "unterm'))
    p = FakeProvider({"content": "", "tool_calls": [broken]}, answer("done"))
    router.chain = [("p", p)]
    executed = []
    msgs = list(USER)
    assert router.ask(msgs, tool_executor=lambda n, a: executed.append(n)) == "done"
    assert executed == []
    assert msgs[-1]["content"] == TRUNCATED_ARGS


def test_round_limit_ends_with_summary_not_silence(router):
    class Looping:
        def __call__(self, messages, tools=None):
            if tools is None:  # the summary request goes out without tools
                return answer("summary of what was done")
            return {"content": "", "tool_calls": [tool_call("web_search", {"query": "x"})]}

    router.chain = [("p", Looping())]
    calls = []
    result = router.ask(list(USER), tool_executor=lambda n, a: calls.append(n) or "r", max_rounds=3)
    assert result == "summary of what was done"
    assert len(calls) == 3


def test_reasoning_is_kept_for_deepseek_and_stripped_for_others(router):
    msgs = [{"role": "assistant", "content": "x", "reasoning_content": "thoughts"}]
    ds, other = FakeProvider(answer("a")), FakeProvider(answer("b"))
    router.chain = [("deepseek", ds)]
    router.ask(list(msgs))
    assert "reasoning_content" in ds.calls[0]["messages"][0]
    router.chain = [("groq", other)]
    router.ask(list(msgs))
    assert "reasoning_content" not in other.calls[0]["messages"][0]


MONDAY_PEAK = datetime(2026, 10, 26, 7, 0, tzinfo=timezone.utc)
MONDAY_NOON = datetime(2026, 10, 26, 12, 0, tzinfo=timezone.utc)
SATURDAY_PEAK_HOUR = datetime(2026, 10, 24, 7, 0, tzinfo=timezone.utc)


def test_peak_hours_are_weekdays_utc():
    assert peak_now_utc(MONDAY_PEAK)
    assert not peak_now_utc(MONDAY_NOON)
    assert not peak_now_utc(SATURDAY_PEAK_HOUR)


def test_peak_guard_skips_pro_model_and_heavy_requests():
    assert peak_skip_deepseek("deepseek-v4-pro", 0, now=MONDAY_PEAK)
    assert peak_skip_deepseek("deepseek-flash", 10_000, now=MONDAY_PEAK)
    assert not peak_skip_deepseek("deepseek-flash", 100, now=MONDAY_PEAK)
    assert not peak_skip_deepseek("deepseek-v4-pro", 10_000, now=MONDAY_NOON)


def test_at_peak_chain_starts_with_cheap_provider(router, monkeypatch):
    monkeypatch.setattr(router_mod, "peak_now_utc", lambda now=None: True)
    router.deepseek_model = "deepseek-v4-pro"
    names = [name for name, _ in router._ordered_chain()]
    assert names[0] == "groq"
    assert "deepseek" not in names and "openrouter" not in names
