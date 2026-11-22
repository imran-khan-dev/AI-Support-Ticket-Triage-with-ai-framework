"""Deterministic tests for the LangGraph Groq model execution stage."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent_core.agent import _get_system_prompt
from app.agent_core.langgraph import (
    EMPTY_RESPONSE_ERROR,
    PROVIDER_ERROR,
    ChatModelProviderError,
    GroqChatModelRunner,
    build_groq_chat_model,
    compile_agent_graph,
    run_agent_graph,
)
from app.core.config import Settings


class FakeChatModel:
    """Deterministic model runner that records the messages it receives."""

    def __init__(self, response: str = "model reply"):
        self.response = response
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=self.response)


class FailingChatModel:
    def invoke(self, messages):
        raise ChatModelProviderError("secret provider detail")


class UnexpectedFailureChatModel:
    def invoke(self, messages):
        raise RuntimeError("unexpected model failure")


class _FakeGroqResponse:
    output = []
    output_text = "Groq response"
    finish_reason = None


class _FakeResponses:
    def __init__(self):
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeGroqResponse()


class _FakeGroqClient:
    def __init__(self):
        self.responses = _FakeResponses()


class _FakeToolCallItem:
    type = "function_call"

    def __init__(self):
        self.call_id = "call_1"
        self.name = "search_knowledge"
        self.arguments = '{"query": "password reset"}'


class _FakeToolResponse:
    output = [_FakeToolCallItem()]
    output_text = ""
    finish_reason = "tool_calls"


class _FakeToolResponses:
    def __init__(self):
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeToolResponse()


class _FakeToolGroqClient:
    def __init__(self):
        self.responses = _FakeToolResponses()


def _settings(**overrides):
    values = {
        "_env_file": None,
        "groq_api_key": "test-key",
        "groq_model": "openai/gpt-oss-20b",
        "groq_base_url": "https://api.groq.com/openai/v1",
        "database_url": "postgresql://localhost/test",
    }
    values.update(overrides)
    return Settings(**values)


def test_call_model_appends_assistant_response_and_sets_answer():
    model = FakeChatModel("Final answer")
    graph = compile_agent_graph(model)

    result = run_agent_graph("Question?", graph=graph)

    assert result.error is None
    assert result.answer == "Final answer"
    assert isinstance(result.messages[-1], AIMessage)
    assert result.messages[-1].content == "Final answer"
    assert len(model.calls) == 1


def test_model_receives_system_prompt_then_conversation_messages():
    history = [
        HumanMessage(content="old user"),
        AIMessage(content="old assistant"),
        ToolMessage(content="old tool", tool_call_id="call_1"),
    ]
    model = FakeChatModel("new answer")
    graph = compile_agent_graph(model)

    result = run_agent_graph("new user", graph=graph, history=history)

    sent_messages = model.calls[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == _get_system_prompt()
    assert [type(message) for message in sent_messages[1:]] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
    ]
    assert [message.content for message in sent_messages[1:]] == [
        "old user",
        "old assistant",
        "old tool",
        "new user",
    ]

    assert [type(message) for message in result.messages] == [
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
        AIMessage,
    ]
    assert result.answer == "new answer"


def test_dict_history_is_normalized_and_preserved():
    history = [
        {"role": "user", "content": "old user"},
        {"role": "assistant", "content": "old assistant"},
    ]
    graph = compile_agent_graph(FakeChatModel("new answer"))

    result = run_agent_graph("new user", graph=graph, history=history)

    assert [type(message) for message in result.messages] == [
        HumanMessage,
        AIMessage,
        HumanMessage,
        AIMessage,
    ]
    assert [message.content for message in result.messages] == [
        "old user",
        "old assistant",
        "new user",
        "new answer",
    ]


def test_provider_failure_returns_safe_error_without_assistant_message():
    graph = compile_agent_graph(FailingChatModel())

    result = run_agent_graph("hello", graph=graph)

    assert result.answer is None
    assert result.error == PROVIDER_ERROR
    assert "secret" not in (result.error or "")
    assert all(not isinstance(message, AIMessage) for message in result.messages)


def test_unexpected_model_exception_propagates():
    graph = compile_agent_graph(UnexpectedFailureChatModel())

    with pytest.raises(RuntimeError, match="unexpected model failure"):
        run_agent_graph("hello", graph=graph)


def test_empty_model_response_returns_safe_error():
    graph = compile_agent_graph(FakeChatModel(""))

    result = run_agent_graph("hello", graph=graph)

    assert result.answer is None
    assert result.error == EMPTY_RESPONSE_ERROR
    assert isinstance(result.messages[-1], AIMessage)


def test_model_injection_into_independently_constructed_graphs():
    first_model = FakeChatModel("first answer")
    second_model = FakeChatModel("second answer")

    first = run_agent_graph("first", graph=compile_agent_graph(first_model))
    second = run_agent_graph("second", graph=compile_agent_graph(second_model))

    assert first.answer == "first answer"
    assert second.answer == "second answer"
    assert len(first_model.calls) == 1
    assert len(second_model.calls) == 1


def test_groq_chat_model_runner_uses_configured_model_and_converts_messages():
    client = _FakeGroqClient()
    runner = GroqChatModelRunner(client, "openai/gpt-oss-20b")

    response = runner.invoke(
        [
            SystemMessage(content="system instructions"),
            HumanMessage(content="hello"),
        ]
    )

    assert isinstance(response, AIMessage)
    assert response.content == "Groq response"
    assert client.responses.kwargs is not None
    assert client.responses.kwargs["model"] == "openai/gpt-oss-20b"
    assert client.responses.kwargs["instructions"] == "system instructions"
    assert any(
        item.get("role") == "user"
        and item.get("content") == [{"type": "input_text", "text": "hello"}]
        for item in client.responses.kwargs["input"]
    )


def test_build_groq_chat_model_uses_existing_settings():
    runner = build_groq_chat_model(_settings())

    assert isinstance(runner, GroqChatModelRunner)


def test_build_groq_chat_model_requires_api_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_groq_chat_model(_settings(groq_api_key=None))


def test_build_groq_chat_model_requires_model(monkeypatch):
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(ValueError, match="GROQ_MODEL"):
        build_groq_chat_model(_settings(groq_model=None))


def test_groq_chat_model_runner_preserves_provider_tool_calls():
    client = _FakeToolGroqClient()
    runner = GroqChatModelRunner(client, "openai/gpt-oss-20b")
    tools = [
        {
            "name": "search_knowledge",
            "description": "Search knowledge",
            "parameters_schema": {"type": "object", "properties": {}},
        }
    ]

    response = runner.invoke([HumanMessage(content="Find password reset docs.")], tools=tools)

    assert isinstance(response, AIMessage)
    assert response.content == ""
    assert response.tool_calls[0]["name"] == "search_knowledge"
    assert response.tool_calls[0]["args"] == {"query": "password reset"}
    assert response.tool_calls[0]["id"] == "call_1"
    assert client.responses.kwargs is not None
    assert client.responses.kwargs["tools"] == [
        {
            "type": "function",
            "name": "search_knowledge",
            "description": "Search knowledge",
            "parameters": {"type": "object", "properties": {}},
        }
    ]