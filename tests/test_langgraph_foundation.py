"""Deterministic tests for the LangGraph agent orchestration foundation."""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START

from app.agent_core.langgraph import (
    AGENT_GRAPH,
    EMPTY_INPUT_ERROR,
    NO_MODEL_ERROR,
    build_agent_graph,
    compile_agent_graph,
    initialize_messages,
    run_agent_graph,
)


class FakeChatModel:
    """Deterministic model runner for graph tests."""

    def __init__(self, response: str = "model reply"):
        self.response = response
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.response)


def test_graph_builds_and_compiles():
    graph = build_agent_graph()

    assert "initialize_messages" in graph.nodes
    assert "call_model" in graph.nodes

    compiled = compile_agent_graph()

    assert hasattr(compiled, "invoke")
    assert compiled.checkpointer is None


def test_expected_graph_topology():
    graph = build_agent_graph()

    assert set(graph.nodes) == {"initialize_messages", "call_model"}
    assert set(graph.edges) == {
        (START, "initialize_messages"),
        ("initialize_messages", "call_model"),
        ("call_model", END),
    }


def test_successful_graph_invocation():
    model = FakeChatModel("How to reset your password.")
    graph = compile_agent_graph(model)

    result = run_agent_graph("How do I reset my password?", graph=graph)

    assert result.error is None
    assert result.answer == "How to reset your password."
    assert len(result.messages) == 2
    assert isinstance(result.messages[0], HumanMessage)
    assert result.messages[0].content == "How do I reset my password?"
    assert isinstance(result.messages[1], AIMessage)
    assert result.messages[1].content == "How to reset your password."
    assert len(model.calls) == 1


def test_default_graph_without_model_returns_safe_error():
    result = run_agent_graph("hello")

    assert result.answer is None
    assert result.error == NO_MODEL_ERROR
    assert len(result.messages) == 1
    assert isinstance(result.messages[0], HumanMessage)


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
    graph = compile_agent_graph(FakeChatModel("model turn"))

    final_state = graph.invoke({"messages": messages})

    assert [type(message) for message in final_state["messages"]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        AIMessage,
    ]
    assert [message.content for message in final_state["messages"]] == [
        "user turn",
        "assistant turn",
        "tool turn",
        "model turn",
    ]
    assert final_state["error"] is None


def test_empty_and_whitespace_input_return_safe_error():
    model = FakeChatModel()
    graph = compile_agent_graph(model)

    for message in ("", "   ", "\n\t"):
        result = run_agent_graph(message, graph=graph)

        assert result.messages == []
        assert result.error == EMPTY_INPUT_ERROR

    assert model.calls == []


def test_invalid_input_type_returns_safe_error():
    model = FakeChatModel()
    graph = compile_agent_graph(model)

    result = run_agent_graph(None, graph=graph)  # type: ignore[arg-type]

    assert result.messages == []
    assert result.error == EMPTY_INPUT_ERROR
    assert model.calls == []


def test_direct_empty_message_list_returns_safe_error():
    model = FakeChatModel()
    graph = compile_agent_graph(model)

    final_state = graph.invoke({"messages": []})

    assert final_state["messages"] == []
    assert final_state["error"] == EMPTY_INPUT_ERROR
    assert model.calls == []


def test_repeated_invocations_do_not_leak_state():
    graph = compile_agent_graph(FakeChatModel("model reply"))

    first = run_agent_graph("first message", graph=graph)
    second = run_agent_graph("second message", graph=graph)

    assert [message.content for message in first.messages] == [
        "first message",
        "model reply",
    ]
    assert [message.content for message in second.messages] == [
        "second message",
        "model reply",
    ]
    assert first.messages[0].id != second.messages[0].id
    assert len(first.messages) == 2
    assert len(second.messages) == 2