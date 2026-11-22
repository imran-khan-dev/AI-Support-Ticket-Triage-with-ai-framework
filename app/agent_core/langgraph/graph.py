"""Graph construction for the LangGraph agent foundation."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent_core.langgraph.model import ChatModelRunner, build_groq_chat_model
from app.agent_core.langgraph.nodes import build_call_model_node, initialize_messages
from app.agent_core.langgraph.state import AgentGraphState
from app.core.config import Settings


def build_agent_graph(model: ChatModelRunner | None = None) -> StateGraph:
    """Build the agent graph.

    Topology: ``START -> initialize_messages -> call_model -> END``.

    The model runner is injected at construction time so tests can supply a
    deterministic model and production can supply a Groq-backed runner without
    global mutable model state.
    """
    graph = StateGraph(AgentGraphState)
    graph.add_node("initialize_messages", initialize_messages)
    graph.add_node("call_model", build_call_model_node(model))
    graph.add_edge(START, "initialize_messages")
    graph.add_edge("initialize_messages", "call_model")
    graph.add_edge("call_model", END)
    return graph


def compile_agent_graph(
    model: ChatModelRunner | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph without checkpointing."""
    return build_agent_graph(model).compile()


def build_groq_agent_graph(settings: Settings) -> CompiledStateGraph:
    """Build a graph backed by the configured Groq model."""
    return compile_agent_graph(build_groq_chat_model(settings))


AGENT_GRAPH = compile_agent_graph()