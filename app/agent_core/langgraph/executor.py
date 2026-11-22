"""Execution boundary for the LangGraph agent foundation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage

from app.agent_core.langgraph.graph import AGENT_GRAPH
from app.agent_core.langgraph.state import AgentGraphResult


def run_agent_graph(
    message: str,
    *,
    graph: Any = AGENT_GRAPH,
    history: Sequence[BaseMessage | dict[str, Any]] | None = None,
) -> AgentGraphResult:
    """Invoke the compiled agent graph with one initial input message.

    ``history`` is optional application-neutral conversation context. It is
    passed into the graph's message state before the current input message is
    appended. The caller must not include the current message in ``history``.

    This is a synchronous execution boundary. It intentionally does not add an
    async abstraction until a milestone needs one. Each call starts from a new
    graph state, so independent invocations do not share messages.
    """
    payload: dict[str, Any] = {"input": message}
    if history:
        payload["messages"] = list(history)

    final_state = graph.invoke(payload)
    return AgentGraphResult(
        messages=list(final_state.get("messages") or []),
        answer=final_state.get("answer"),
        error=final_state.get("error"),
    )