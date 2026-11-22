"""Initial node implementations for the LangGraph agent foundation."""

from __future__ import annotations

from typing import Any

from app.agent_core.langgraph.state import AgentGraphState

EMPTY_INPUT_ERROR = "A non-empty message is required."


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