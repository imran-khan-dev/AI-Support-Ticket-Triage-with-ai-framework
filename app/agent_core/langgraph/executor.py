"""Execution boundary for the LangGraph agent foundation."""

from __future__ import annotations

from typing import Any

from app.agent_core.langgraph.graph import AGENT_GRAPH
from app.agent_core.langgraph.state import AgentGraphResult


def run_agent_graph(
    message: str,
    *,
    graph: Any = AGENT_GRAPH,
) -> AgentGraphResult:
    """Invoke the compiled agent graph with one initial input message.

    This is a synchronous execution boundary. It intentionally does not add an
    async abstraction until a milestone needs one. Each call starts from a new
    graph state, so independent invocations do not share messages.
    """
    final_state = graph.invoke({"input": message})
    return AgentGraphResult(
        messages=list(final_state.get("messages") or []),
        answer=final_state.get("answer"),
        error=final_state.get("error"),
    )