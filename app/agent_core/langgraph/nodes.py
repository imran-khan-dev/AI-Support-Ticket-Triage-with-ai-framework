"""Node implementations for the LangGraph agent foundation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.messages import SystemMessage
from langgraph.graph import END

from app.agent_core.langgraph.model import ChatModelProviderError, ChatModelRunner
from app.agent_core.langgraph.state import AgentGraphState
from app.agent_core.langgraph.tools import (
    DEFAULT_MAX_TOOL_CALLS,
    MALFORMED_TOOL_CALL_IDS_ERROR,
    MalformedToolCallIdsError,
    execute_tool_calls,
)

EMPTY_INPUT_ERROR = "A non-empty message is required."
EMPTY_RESPONSE_ERROR = "The model did not produce a final answer."
NO_MODEL_ERROR = "No chat model is configured for the agent graph."
PROVIDER_ERROR = "Provider service is temporarily unavailable."
NO_TOOL_EXECUTOR_ERROR = (
    "The model requested a tool but no tool executor is configured."
)
NO_TOOL_CALLS_ERROR = "The model did not request any tool calls."
MAX_ITERATIONS_ERROR = (
    "Agent reached the maximum number of iterations without producing a final answer."
)

MAX_ITERATIONS = 10


def _validate_max_iterations(max_iterations: int) -> None:
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer.")


def initialize_messages(state: AgentGraphState) -> dict[str, Any]:
    """Initialize and normalize the graph's message state.

    The application supplies the initial input string. This node converts it
    into LangGraph's message representation through the ``add_messages``
    reducer. A graph invoked directly with existing messages leaves their order
    and types untouched.
    """
    if "input" in state and state["input"] is not None:
        raw_input = state["input"]
        if not isinstance(raw_input, str) or not raw_input.strip():
            return {"answer": None, "error": EMPTY_INPUT_ERROR}
        return {
            "messages": [{"role": "user", "content": raw_input}],
            "answer": None,
            "error": None,
        }

    if not state.get("messages"):
        return {"answer": None, "error": EMPTY_INPUT_ERROR}

    return {"answer": None, "error": None}


def build_call_model_node(
    model: ChatModelRunner | None,
    *,
    tool_definitions: Sequence[Any] | None = None,
    tool_executor: Any | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> Callable[[AgentGraphState], dict[str, Any]]:
    """Build the model-invocation node for a specific model runner.

    Tool definitions are advertised only when an executor is available. The
    node records each model round in ``iteration_count`` and stops with a safe
    error when the configured loop limit is reached.
    """
    _validate_max_iterations(max_iterations)

    definitions = list(tool_definitions or [])
    if tool_executor is None:
        definitions = []
    has_tool_executor = tool_executor is not None

    def call_model(state: AgentGraphState) -> dict[str, Any]:
        if state.get("error"):
            return {"answer": None, "error": state["error"]}

        messages = list(state.get("messages") or [])
        if not messages:
            return {"answer": None, "error": EMPTY_INPUT_ERROR}

        if model is None:
            return {"answer": None, "error": NO_MODEL_ERROR}

        iteration = int(state.get("iteration_count") or 0) + 1
        if iteration > max_iterations:
            return {
                "answer": None,
                "error": MAX_ITERATIONS_ERROR,
                "iteration_count": iteration,
            }

        from app.agent_core.agent import _get_system_prompt

        model_messages = [SystemMessage(content=_get_system_prompt()), *messages]

        try:
            if definitions:
                response = model.invoke(model_messages, tools=definitions)
            else:
                response = model.invoke(model_messages)
        except ChatModelProviderError:
            return {
                "answer": None,
                "error": PROVIDER_ERROR,
                "iteration_count": iteration,
            }

        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        if tool_calls:
            if not has_tool_executor:
                return {
                    "messages": [response],
                    "answer": None,
                    "error": NO_TOOL_EXECUTOR_ERROR,
                    "iteration_count": iteration,
                }
            if iteration >= max_iterations:
                return {
                    "messages": [response],
                    "answer": None,
                    "error": MAX_ITERATIONS_ERROR,
                    "iteration_count": iteration,
                }
            return {
                "messages": [response],
                "answer": None,
                "error": None,
                "iteration_count": iteration,
            }

        if not content.strip():
            return {
                "messages": [response],
                "answer": None,
                "error": EMPTY_RESPONSE_ERROR,
                "iteration_count": iteration,
            }

        return {
            "messages": [response],
            "answer": content,
            "error": None,
            "iteration_count": iteration,
        }

    return call_model


def build_execute_tools_node(
    tool_executor: Any,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> Callable[[AgentGraphState], dict[str, Any]]:
    """Build the node that executes the previous assistant message's tools.

    The budget is stored in graph state so it is shared across model/tool cycles
    but isolated between independent graph invocations.
    """
    from app.agent_core.langgraph.tools import _validate_max_tool_calls

    _validate_max_tool_calls(max_tool_calls)

    def execute_tools(state: AgentGraphState) -> dict[str, Any]:
        if state.get("error"):
            return {"answer": None, "error": state["error"]}

        messages = list(state.get("messages") or [])
        if not messages:
            return {"answer": None, "error": EMPTY_INPUT_ERROR}

        tool_calls = list(getattr(messages[-1], "tool_calls", None) or [])
        if not tool_calls:
            return {"answer": None, "error": NO_TOOL_CALLS_ERROR}

        used_tool_calls = int(state.get("tool_call_count") or 0)
        try:
            tool_messages = execute_tool_calls(
                tool_calls,
                tool_executor,
                max_tool_calls=max_tool_calls,
                used_tool_calls=used_tool_calls,
            )
        except MalformedToolCallIdsError:
            return {
                "messages": [],
                "answer": None,
                "error": MALFORMED_TOOL_CALL_IDS_ERROR,
            }

        return {
            "messages": tool_messages,
            "answer": None,
            "error": None,
            "tool_call_count": used_tool_calls + len(tool_calls),
        }

    return execute_tools


def route_after_model(state: AgentGraphState) -> str:
    """Route to tools when the model requested them, otherwise finish."""
    if state.get("error"):
        return END

    messages = state.get("messages") or []
    if messages and getattr(messages[-1], "tool_calls", None):
        return "execute_tools"
    return END


# Descriptive alias for callers that prefer the common LangGraph name.
should_continue = route_after_model


__all__ = [
    "EMPTY_INPUT_ERROR",
    "EMPTY_RESPONSE_ERROR",
    "MAX_ITERATIONS",
    "MAX_ITERATIONS_ERROR",
    "NO_MODEL_ERROR",
    "NO_TOOL_CALLS_ERROR",
    "NO_TOOL_EXECUTOR_ERROR",
    "PROVIDER_ERROR",
    "build_call_model_node",
    "build_execute_tools_node",
    "initialize_messages",
    "route_after_model",
    "should_continue",
]