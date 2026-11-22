"""Graph construction for the LangGraph agent foundation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent_core.langgraph.nodes import initialize_messages
from app.agent_core.langgraph.state import AgentGraphState


def build_agent_graph() -> StateGraph:
    """Build the initial agent graph.

    Topology: ``START -> initialize_messages -> END``.

    Future milestones can add model, tool-routing, and tool-execution nodes
    without changing the application-owned execution boundary.
    """
    graph = StateGraph(AgentGraphState)
    graph.add_node("initialize_messages", initialize_messages)
    graph.add_edge(START, "initialize_messages")
    graph.add_edge("initialize_messages", END)
    return graph


def compile_agent_graph() -> CompiledStateGraph:
    """Compile the agent graph without checkpointing."""
    return build_agent_graph().compile()


AGENT_GRAPH = compile_agent_graph()