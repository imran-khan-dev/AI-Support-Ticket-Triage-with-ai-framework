"""Deterministic tests for LangGraph tool-calling and conditional routing."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from agent_orchestrator.tools.models import ToolResult

from app.agent_core.langgraph import (
    MAX_ITERATIONS_ERROR,
    build_agent_graph,
    compile_agent_graph,
    execute_tool_calls,
    route_after_model,
    run_agent_graph,
)
from app.agent_core.langgraph.nodes import build_call_model_node
from app.agent_core.langgraph.tools import build_application_tool_runtime


class FakeToolExecutor:
    """Deterministic executor that records calls and returns a safe result."""

    def __init__(self, *, fail_with: Exception | None = None):
        self.calls = []
        self.fail_with = fail_with

    def execute(self, tool_call):
        self.calls.append(tool_call)
        if self.fail_with is not None:
            raise self.fail_with
        return ToolResult(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            output={"tool": tool_call.name, "results": [], "count": 0},
            success=True,
            error=None,
        )


class FakeToolCallingModel:
    """Model that requests a tool first, then returns a final answer."""

    def __init__(self, tool_calls=None, final_answer="final answer"):
        self.tool_calls = tool_calls or [
            {
                "name": "search_knowledge",
                "args": {"query": "password reset"},
                "id": "call_1",
            }
        ]
        self.final_answer = final_answer
        self.calls = []
        self.tools_seen = []

    def invoke(self, messages, tools=None):
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if len(self.calls) == 1:
            return AIMessage(content="", tool_calls=self.tool_calls)
        return AIMessage(content=self.final_answer)


class AlwaysToolCallingModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, messages, tools=None):
        self.calls += 1
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_knowledge",
                    "args": {"query": "loop"},
                    "id": f"call_{self.calls}",
                }
            ],
        )


class RepeatingToolCallingModel:
    """Requests a tool for each fresh user input, then answers."""

    def invoke(self, messages, tools=None):
        last = messages[-1]
        if getattr(last, "content", None) in {"first", "second"}:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": str(last.content)},
                        "id": f"call_{last.content}",
                    }
                ],
            )
        return AIMessage(content="final answer")


def test_route_after_model_sends_tool_calls_to_execute_tools():
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_knowledge",
                        "args": {"query": "x"},
                        "id": "call_1",
                    }
                ],
            )
        ]
    }

    assert route_after_model(state) == "execute_tools"


def test_route_after_model_finishes_for_final_answer():
    state = {"messages": [AIMessage(content="done")]}

    assert route_after_model(state) == END


def test_graph_executes_tool_call_then_returns_final_answer():
    executor = FakeToolExecutor()
    model = FakeToolCallingModel()
    definitions = [
        {
            "name": "search_knowledge",
            "description": "Search knowledge",
            "parameters_schema": {"type": "object", "properties": {}},
        }
    ]
    graph = compile_agent_graph(
        model,
        tool_executor=executor,
        tool_definitions=definitions,
    )

    result = run_agent_graph("How do I reset my password?", graph=graph)

    assert result.error is None
    assert result.answer == "final answer"
    assert [type(message) for message in result.messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert result.messages[1].tool_calls[0]["name"] == "search_knowledge"
    assert result.messages[2].tool_call_id == "call_1"
    assert json.loads(result.messages[2].content)["tool"] == "search_knowledge"
    assert len(executor.calls) == 1
    assert executor.calls[0].name == "search_knowledge"
    assert executor.calls[0].arguments == {"query": "password reset"}
    assert model.tools_seen[0] == definitions


def test_multiple_tool_calls_execute_in_request_order():
    executor = FakeToolExecutor()
    model = FakeToolCallingModel(
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {"query": "billing"},
                "id": "call_b",
            },
            {
                "name": "lookup_ticket_history",
                "args": {"query": "login"},
                "id": "call_a",
            },
        ]
    )
    graph = compile_agent_graph(model, tool_executor=executor)

    result = run_agent_graph("Help", graph=graph)

    assert [message.tool_call_id for message in result.messages if isinstance(message, ToolMessage)] == [
        "call_b",
        "call_a",
    ]
    assert [call.id for call in executor.calls] == ["call_b", "call_a"]


def test_loop_limit_stops_before_executing_tools_without_a_final_round():
    executor = FakeToolExecutor()
    model = AlwaysToolCallingModel()
    graph = compile_agent_graph(
        model,
        tool_executor=executor,
        max_iterations=2,
    )

    result = run_agent_graph("Keep searching.", graph=graph)

    assert model.calls == 2
    assert len(executor.calls) == 1
    assert result.answer is None
    assert result.error == MAX_ITERATIONS_ERROR


def test_tool_failure_is_returned_as_safe_tool_message():
    executor = FakeToolExecutor(fail_with=RuntimeError("secret detail"))
    model = FakeToolCallingModel()
    graph = compile_agent_graph(model, tool_executor=executor)

    result = run_agent_graph("Help", graph=graph)

    tool_message = next(
        message for message in result.messages if isinstance(message, ToolMessage)
    )
    payload = json.loads(tool_message.content)
    assert payload["error"] == "tool_failure"
    assert "secret detail" not in tool_message.content


def test_unknown_tool_is_normalized_to_unknown_tool_error():
    runtime = build_application_tool_runtime(tenant_id="tenant-1", db=object())
    model = FakeToolCallingModel(
        tool_calls=[
            {
                "name": "unknown_tool",
                "args": {},
                "id": "call_1",
            }
        ]
    )
    graph = compile_agent_graph(
        model,
        tool_executor=runtime.executor,
        tool_definitions=runtime.definitions,
    )

    result = run_agent_graph("Help", graph=graph)

    tool_message = next(
        message for message in result.messages if isinstance(message, ToolMessage)
    )
    payload = json.loads(tool_message.content)
    assert payload["error"] == "unknown_tool"


def test_unapproved_gated_tool_is_not_executed():
    runtime = build_application_tool_runtime(
        tenant_id="tenant-1",
        db=object(),
        is_approved=False,
    )
    model = FakeToolCallingModel(
        tool_calls=[
            {
                "name": "create_support_case",
                "args": {
                    "subject": "s",
                    "description": "d",
                    "priority": "low",
                    "category": "billing",
                },
                "id": "call_1",
            }
        ]
    )
    graph = compile_agent_graph(
        model,
        tool_executor=runtime.executor,
        tool_definitions=runtime.definitions,
    )

    result = run_agent_graph("Create a case", graph=graph)

    tool_message = next(
        message for message in result.messages if isinstance(message, ToolMessage)
    )
    payload = json.loads(tool_message.content)
    assert payload["error"] == "approval_required"
    assert payload["results"] == []


def test_search_tool_uses_trusted_tenant_context():
    fake_chunk = SimpleNamespace(
        id="chunk-1",
        content="password reset",
        metadata={"filename": "doc.txt"},
        chunk_index=0,
    )
    fake_result = SimpleNamespace(chunk=fake_chunk, score=0.9, score_type="bm25")
    fake_service = Mock()
    fake_service.search.return_value = [fake_result]

    runtime = build_application_tool_runtime(
        tenant_id="trusted-tenant",
        db=object(),
    )
    model = FakeToolCallingModel(
        tool_calls=[
            {
                "name": "search_knowledge",
                "args": {"query": "password reset"},
                "id": "call_1",
            }
        ]
    )
    graph = compile_agent_graph(
        model,
        tool_executor=runtime.executor,
        tool_definitions=runtime.definitions,
    )

    with patch(
        "app.agent_core.tools.get_knowledge_service", return_value=fake_service
    ):
        result = run_agent_graph("How do I reset my password?", graph=graph)

    assert result.error is None
    assert fake_service.search.call_args.args[0] == "trusted-tenant"


def test_execute_tool_calls_returns_one_message_per_call():
    executor = FakeToolExecutor()
    messages = execute_tool_calls(
        [
            {"name": "search_knowledge", "args": {"query": "a"}, "id": "call_1"},
            {"name": "lookup_ticket_history", "args": {"query": "b"}, "id": "call_2"},
        ],
        executor,
    )

    assert [message.tool_call_id for message in messages] == ["call_1", "call_2"]
    assert [message.name for message in messages] == [
        "search_knowledge",
        "lookup_ticket_history",
    ]


def test_call_model_rejects_invalid_loop_limit():
    try:
        build_call_model_node(None, max_iterations=0)
    except ValueError as exc:
        assert "max_iterations" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_graph_without_executor_keeps_foundation_topology():
    graph = build_agent_graph()

    assert set(graph.nodes) == {"initialize_messages", "call_model"}
    assert "execute_tools" not in graph.nodes


def test_tools_argument_accepts_tool_runtime():
    runtime = build_application_tool_runtime(tenant_id="tenant-1", db=object())
    graph = build_agent_graph(FakeToolCallingModel(), tools=runtime)

    assert "execute_tools" in graph.nodes


def test_repeated_invocations_do_not_leak_tool_or_iteration_state():
    executor = FakeToolExecutor()
    graph = compile_agent_graph(
        RepeatingToolCallingModel(),
        tool_executor=executor,
    )

    first = run_agent_graph("first", graph=graph)
    second = run_agent_graph("second", graph=graph)

    assert [type(message) for message in first.messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert [type(message) for message in second.messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert first.messages[0].id != second.messages[0].id
    assert [call.id for call in executor.calls] == ["call_first", "call_second"]

    first_state = graph.invoke({"input": "first"})
    second_state = graph.invoke({"input": "second"})
    assert first_state["iteration_count"] == 2
    assert second_state["iteration_count"] == 2