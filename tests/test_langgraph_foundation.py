"""Deterministic tests for the LangGraph agent orchestration foundation."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START

from app.agent_core.langgraph import (
    AGENT_GRAPH,
    EMPTY_INPUT_ERROR,
    build_agent_graph,
    compile_agent_graph,
    initialize_messages,
    run_agent_graph,
)


def test_graph_builds_and_compiles():
    graph = build_agent_graph()

    assert "initialize_messages" in graph.nodes

    compiled = compile_agent_graph()

    assert hasattr(compiled, "invoke")
    assert compiled.checkpointer is None


def test_expected_graph_topology():
    graph = build_agent_graph()

    assert set(graph.nodes) == {"initialize_messages"}
    assert set(graph.edges) == {
        (START, "initialize_messages"),
        ("initialize_messages", END),
    }


def test_successful_graph_invocation():
    result = run_agent_graph("How do I reset my password?")

    assert result.error is None
    assert result.answer is None
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], HumanMessage)
    assert result.messages[0].content == "How do I reset my password?"


def test_initial_node_normalizes_input_message():
    update = initialize_messages({"input": "hello"})

    assert update == {
        "messages": [{"role": "user", "content": "hello"}],
        "answer": None,
        "error": None,
    }


def test_initial_node_clears_previous_error_on_valid_input():
    update = initialize_messages({"input": "hello", "error": "previous error"})

    assert update["error"] is None
    assert update["messages"] == [{"role": "user", "content": "hello"}]


def test_direct_message_state_preserves_order_and_types():
    messages = [
        HumanMessage(content="user turn"),
        AIMessage(content="assistant turn"),
        ToolMessage(content="tool turn", tool_call_id="call_1"),
    ]

    final_state = AGENT_GRAPH.invoke({"messages": messages})

    assert [type(message) for message in final_state["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
    ]
    assert [message.content for message in final_state["messages"]] == [
        "user turn",
        "assistant turn",
        "tool turn",
    ]
    assert final_state["error"] is None


def test_empty_and_whitespace_input_return_safe_error():
    for message in ("", "   ", "\n\t"):
        result = run_agent_graph(message)

        assert result.messages == []
        assert result.error == EMPTY_INPUT_ERROR


def test_invalid_input_type_returns_safe_error():
    result = run_agent_graph(None)  # type: ignore[arg-type]

    assert result.messages == []
    assert result.error == EMPTY_INPUT_ERROR


def test_direct_empty_message_list_returns_safe_error():
    final_state = AGENT_GRAPH.invoke({"messages": []})

    assert final_state["messages"] == []
    assert final_state["error"] == EMPTY_INPUT_ERROR


def test_repeated_invocations_do_not_leak_state():
    first = run_agent_graph("first message")
    second = run_agent_graph("second message")

    assert [message.content for message in first.messages] == ["first message"]
    assert [message.content for message in second.messages] == ["second message"]
    assert first.messages[0].id != second.messages[0].id