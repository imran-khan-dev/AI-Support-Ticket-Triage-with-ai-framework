"""Regression tests for LangGraph tool-call limits and ID validation."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, ToolMessage

from agent_orchestrator.tools.models import ToolResult

from app.agent_core.langgraph import (
    DEFAULT_MAX_TOOL_CALLS,
    MALFORMED_TOOL_CALL_IDS_ERROR,
    TOOL_CALL_LIMIT_MESSAGE,
    build_application_tool_runtime,
    compile_agent_graph,
    run_agent_graph,
)


def _tool_calls(count: int, *, prefix: str = "call") -> list[dict]:
    return [
        {
            "name": "search_knowledge",
            "args": {"query": f"query-{index}"},
            "id": f"{prefix}_{index}",
        }
        for index in range(count)
    ]


class ScriptedToolModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, messages, tools=None):
        self.calls += 1
        if not self.responses:
            raise AssertionError("ScriptedToolModel ran out of responses")
        return self.responses.pop(0)


class FreshBudgetModel:
    """Requests a full tool batch for each fresh user input, then answers."""

    def invoke(self, messages, tools=None):
        last = messages[-1]
        if getattr(last, "content", None) in {"first", "second"}:
            return AIMessage(content="", tool_calls=_tool_calls(25))
        return AIMessage(content="done")


class RecordingExecutor:
    def __init__(self, *, fail_ids: set[str] | None = None):
        self.calls = []
        self.fail_ids = fail_ids or set()

    def execute(self, tool_call):
        self.calls.append(tool_call)
        if tool_call.id in self.fail_ids:
            return ToolResult(
                tool_call_id=tool_call.id,
                tool_name=tool_call.name,
                output=None,
                success=False,
                error="RuntimeError: secret detail",
            )
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output={"tool": tool_call.name, "results": [], "count": 0},
            success=True,
            error=None,
        )


def _tool_messages(result):
    return [message for message in result.messages if isinstance(message, ToolMessage)]


def _payloads(result):
    return [json.loads(message.content) for message in _tool_messages(result)]


def test_single_response_is_limited_and_excess_calls_are_correlated():
    calls = _tool_calls(25)
    model = ScriptedToolModel(
        [
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="done"),
        ]
    )
    executor = RecordingExecutor()
    graph = compile_agent_graph(
        model,
        tool_executor=executor,
        max_tool_calls=20,
        max_iterations=2,
    )

    result = run_agent_graph("go", graph=graph)

    assert [call.id for call in executor.calls] == [f"call_{i}" for i in range(20)]
    messages = _tool_messages(result)
    assert [message.tool_call_id for message in messages] == [
        f"call_{i}" for i in range(25)
    ]
    payloads = _payloads(result)
    assert all("error" not in payload for payload in payloads[:20])
    assert all(payload["error"] == "tool_failure" for payload in payloads[20:])
    assert all(
        payload["message"] == TOOL_CALL_LIMIT_MESSAGE for payload in payloads[20:]
    )
    assert result.answer == "done"


def test_tool_budget_is_shared_across_model_tool_cycles():
    model = ScriptedToolModel(
        [
            AIMessage(content="", tool_calls=_tool_calls(2, prefix="a")),
            AIMessage(content="", tool_calls=_tool_calls(2, prefix="b")),
            AIMessage(content="done"),
        ]
    )
    executor = RecordingExecutor()
    graph = compile_agent_graph(
        model,
        tool_executor=executor,
        max_tool_calls=3,
        max_iterations=3,
    )

    result = run_agent_graph("go", graph=graph)

    assert [call.id for call in executor.calls] == ["a_0", "a_1", "b_0"]
    payloads = _payloads(result)
    assert len(payloads) == 4
    assert all("error" not in payload for payload in payloads[:3])
    assert payloads[3]["error"] == "tool_failure"
    assert payloads[3]["message"] == TOOL_CALL_LIMIT_MESSAGE
    assert result.answer == "done"


def test_fresh_graph_invocation_starts_with_a_fresh_budget():
    executor = RecordingExecutor()
    graph = compile_agent_graph(
        FreshBudgetModel(),
        tool_executor=executor,
        max_tool_calls=20,
        max_iterations=2,
    )

    first = run_agent_graph("first", graph=graph)
    second = run_agent_graph("second", graph=graph)

    assert len(executor.calls) == 40
    assert [call.id for call in executor.calls[:20]] == [
        f"call_{i}" for i in range(20)
    ]
    assert [call.id for call in executor.calls[20:]] == [
        f"call_{i}" for i in range(20)
    ]
    for result in (first, second):
        payloads = _payloads(result)
        assert len(payloads) == 25
        assert all("error" not in payload for payload in payloads[:20])
        assert all(payload["error"] == "tool_failure" for payload in payloads[20:])


def test_configured_tool_limit_is_respected():
    model = ScriptedToolModel(
        [
            AIMessage(content="", tool_calls=_tool_calls(5)),
            AIMessage(content="done"),
        ]
    )
    executor = RecordingExecutor()
    graph = compile_agent_graph(
        model,
        tool_executor=executor,
        max_tool_calls=2,
        max_iterations=2,
    )

    result = run_agent_graph("go", graph=graph)

    assert len(executor.calls) == 2
    payloads = _payloads(result)
    assert len(payloads) == 5
    assert all("error" not in payload for payload in payloads[:2])
    assert all(payload["error"] == "tool_failure" for payload in payloads[2:])
    assert DEFAULT_MAX_TOOL_CALLS == 20


def test_unknown_and_invalid_calls_consume_the_budget():
    runtime = build_application_tool_runtime(
        tenant_id="tenant-1",
        db=object(),
    )
    tool_calls = []
    for index in range(25):
        if index < 13:
            tool_calls.append(
                {
                    "name": "unknown_tool",
                    "args": {},
                    "id": f"call_{index}",
                }
            )
        else:
            tool_calls.append(
                {
                    "name": "search_knowledge",
                    "args": {"query": ""},
                    "id": f"call_{index}",
                }
            )

    model = ScriptedToolModel(
        [
            AIMessage(content="", tool_calls=tool_calls),
            AIMessage(content="done"),
        ]
    )
    graph = compile_agent_graph(
        model,
        tool_executor=runtime.executor,
        tool_definitions=runtime.definitions,
        max_tool_calls=20,
        max_iterations=2,
    )

    result = run_agent_graph("go", graph=graph)

    payloads = _payloads(result)
    assert len(payloads) == 25
    assert [payload["error"] for payload in payloads[:13]] == [
        "unknown_tool"
    ] * 13
    assert [payload["error"] for payload in payloads[13:20]] == [
        "invalid_arguments"
    ] * 7
    assert all(payload["error"] == "tool_failure" for payload in payloads[20:])
    assert all(
        payload["message"] == TOOL_CALL_LIMIT_MESSAGE for payload in payloads[20:]
    )


def test_missing_tool_call_id_rejects_without_execution():
    executor = RecordingExecutor()
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "x"},
                        "id": "",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = compile_agent_graph(model, tool_executor=executor, max_iterations=2)

    result = run_agent_graph("go", graph=graph)

    assert result.error == MALFORMED_TOOL_CALL_IDS_ERROR
    assert executor.calls == []
    assert _tool_messages(result) == []
    assert model.calls == 1


def test_duplicate_tool_call_ids_reject_without_execution():
    executor = RecordingExecutor()
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "one"},
                        "id": "duplicate",
                    },
                    {
                        "name": "search_knowledge",
                        "args": {"query": "two"},
                        "id": "duplicate",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = compile_agent_graph(model, tool_executor=executor, max_iterations=2)

    result = run_agent_graph("go", graph=graph)

    assert result.error == MALFORMED_TOOL_CALL_IDS_ERROR
    assert executor.calls == []
    assert _tool_messages(result) == []
    assert model.calls == 1


def test_one_missing_id_rejects_the_whole_ambiguous_batch():
    executor = RecordingExecutor()
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "valid"},
                        "id": "valid_1",
                    },
                    {
                        "name": "search_knowledge",
                        "args": {"query": "invalid"},
                        "id": "",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    graph = compile_agent_graph(model, tool_executor=executor, max_iterations=2)

    result = run_agent_graph("go", graph=graph)

    assert result.error == MALFORMED_TOOL_CALL_IDS_ERROR
    assert executor.calls == []
    assert _tool_messages(result) == []


def test_multiple_valid_ids_remain_correlated():
    model = ScriptedToolModel(
        [
            AIMessage(content="", tool_calls=_tool_calls(2)),
            AIMessage(content="done"),
        ]
    )
    executor = RecordingExecutor()
    graph = compile_agent_graph(model, tool_executor=executor, max_iterations=2)

    result = run_agent_graph("go", graph=graph)

    assert [call.id for call in executor.calls] == ["call_0", "call_1"]
    assert [message.tool_call_id for message in _tool_messages(result)] == [
        "call_0",
        "call_1",
    ]


def test_partial_tool_failure_is_correlated_and_does_not_leak():
    model = ScriptedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "good"},
                        "id": "good",
                    },
                    {
                        "name": "search_knowledge",
                        "args": {"query": "bad"},
                        "id": "bad",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    executor = RecordingExecutor(fail_ids={"bad"})
    graph = compile_agent_graph(model, tool_executor=executor, max_iterations=2)

    result = run_agent_graph("go", graph=graph)

    payloads = _payloads(result)
    assert [message.tool_call_id for message in _tool_messages(result)] == [
        "good",
        "bad",
    ]
    assert "error" not in payloads[0]
    assert payloads[1]["error"] == "tool_failure"
    assert "secret detail" not in json.dumps(payloads)