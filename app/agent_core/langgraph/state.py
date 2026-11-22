"""Typed state for the LangGraph agent foundation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentGraphState(TypedDict, total=False):
    """Minimal execution state for the agent graph.

    ``messages`` uses LangGraph's ``add_messages`` reducer so message order and
    message types are preserved as nodes append updates. ``iteration_count``
    tracks model rounds for the graph's loop limit. Application-owned session,
    history, memory, tenant, and approval state deliberately stay outside this
    graph state.
    """

    input: str | None
    messages: Annotated[list[AnyMessage], add_messages]
    answer: str | None
    error: str | None
    iteration_count: int


@dataclass(frozen=True)
class AgentGraphResult:
    """Execution result returned by the graph boundary."""

    messages: list[AnyMessage]
    answer: str | None = None
    error: str | None = None