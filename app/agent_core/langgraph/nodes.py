"""Node implementations for the LangGraph agent foundation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain_core.messages import SystemMessage

from app.agent_core.langgraph.model import ChatModelProviderError, ChatModelRunner
from app.agent_core.langgraph.state import AgentGraphState

EMPTY_INPUT_ERROR = "A non-empty message is required."
EMPTY_RESPONSE_ERROR = "The model did not produce a final answer."
NO_MODEL_ERROR = "No chat model is configured for the agent graph."
PROVIDER_ERROR = "Provider service is temporarily unavailable."


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
) -> Callable[[AgentGraphState], dict[str, Any]]:
    """Build the model-invocation node for a specific model runner."""

    def call_model(state: AgentGraphState) -> dict[str, Any]:
        if state.get("error"):
            return {"answer": None, "error": state["error"]}

        messages = list(state.get("messages") or [])
        if not messages:
            return {"answer": None, "error": EMPTY_INPUT_ERROR}

        if model is None:
            return {"answer": None, "error": NO_MODEL_ERROR}

        from app.agent_core.agent import _get_system_prompt

        model_messages = [SystemMessage(content=_get_system_prompt()), *messages]

        try:
            response = model.invoke(model_messages)
        except ChatModelProviderError:
            return {"answer": None, "error": PROVIDER_ERROR}

        content = getattr(response, "content", "")
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        if not content.strip():
            return {
                "messages": [response],
                "answer": None,
                "error": EMPTY_RESPONSE_ERROR,
            }

        return {
            "messages": [response],
            "answer": content,
            "error": None,
        }

    return call_model