"""Offline tests for the shared agent loop, using a fake LLM client."""
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import agent
import agent_core
import sql_agent


def tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def reply(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    """Returns scripted responses and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def with_options(self, **_):
        return self

    def _create(self, **kwargs):
        # Snapshot messages: the loop keeps appending to the same list
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)


@pytest.fixture
def fake_client(monkeypatch):
    def install(responses):
        client = FakeClient(responses)
        monkeypatch.setattr(agent_core, "get_ai_client", lambda: client)
        return client
    return install


def test_tool_result_goes_back_to_model(fake_client):
    client = fake_client([
        reply(tool_calls=[tool_call("add", {"a": 2, "b": 3})]),
        reply(content="The sum is 5."),
    ])

    answer, trace = agent_core.run_tool_agent(
        "system", [], {"add": lambda a, b: {"sum": a + b}}, "2+3?"
    )

    assert answer == "The sum is 5."
    assert trace == [
        {"tool": "add", "arguments": {"a": 2, "b": 3}, "result": {"sum": 5}}
    ]
    last = client.requests[1]["messages"][-1]
    assert last["role"] == "tool"
    assert json.loads(last["content"]) == {"sum": 5}


def test_unknown_tool_and_bad_arguments_become_errors(fake_client):
    fake_client([
        reply(tool_calls=[
            tool_call("missing", {}, "c1"),
            tool_call("add", {"a": 1}, "c2"),   # missing b -> TypeError
        ]),
        reply(content="done"),
    ])

    _, trace = agent_core.run_tool_agent(
        "system", [], {"add": lambda a, b: a + b}, "q"
    )

    assert trace[0]["result"] == {"error": "Unknown tool requested"}
    assert "error" in trace[1]["result"]


def test_out_of_rounds_forces_answer_without_tools(fake_client):
    client = fake_client(
        [reply(tool_calls=[tool_call("noop", {})])] * 2
        + [reply(content="best effort")]
    )

    answer, trace = agent_core.run_tool_agent(
        "system", [], {"noop": lambda: {}}, "q", max_rounds=2
    )

    assert answer == "best effort"
    assert len(trace) == 2
    assert client.requests[-1]["tool_choice"] == "none"


def test_memory_skips_errors_and_keeps_recent_turns(fake_client):
    client = fake_client([reply(content="ok")])
    history = []
    for i in range(5):
        history.append({"role": "user", "content": f"q{i}"})
        history.append({"role": "assistant", "content": f"a{i}"})
    history.append({"role": "assistant", "content": "boom", "is_error": True})

    agent_core.run_tool_agent("system", [], {}, "now", history, memory_turns=2)

    contents = [m["content"] for m in client.requests[0]["messages"]]
    assert contents == ["system", "q3", "a3", "q4", "a4", "now"]


def capture_call(monkeypatch, module):
    seen = {}

    def fake(system_prompt, tools, functions, question, history, **kw):
        seen.update(system_prompt=system_prompt, tools=tools,
                    functions=functions, question=question,
                    history=history, **kw)
        return "answer", []

    monkeypatch.setattr(module, "run_tool_agent", fake)
    return seen


def test_csv_agent_uses_shared_loop(monkeypatch):
    seen = capture_call(monkeypatch, agent)
    df = pd.DataFrame({"Quarter": ["2026Q1"], "GDP": [100.0]})

    assert agent.run_agent(df, "q", []) == ("answer", [])
    assert seen["system_prompt"] == agent.SYSTEM_PROMPT
    assert seen["tools"] is agent.CHAT_TOOLS
    assert "get_gdp" in seen["functions"]
    assert seen["max_rounds"] == agent.MAX_ROUNDS


def test_sql_agent_uses_shared_loop(monkeypatch):
    seen = capture_call(monkeypatch, sql_agent)

    assert sql_agent.run_sql_agent("q", []) == ("answer", [])
    assert seen["system_prompt"] == sql_agent.SYSTEM_PROMPT
    assert seen["tools"] is sql_agent.SQL_TOOLS
    assert seen["functions"] is sql_agent.FUNCTIONS
    assert seen["max_rounds"] == sql_agent.MAX_ROUNDS
