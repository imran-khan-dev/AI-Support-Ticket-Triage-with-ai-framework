"""Graph construction for the LangGraph agent foundation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent_core.langgraph.model import ChatModelRunner, build_groq_chat_model
from app.agent_core.langgraph.nodes import (
    MAX_ITERATIONS,
    build_call_model_node,
    build_execute_tools_node,
    initialize_messages,
    route_after_model,
)
from app.agent_core.langgraph.state import AgentGraphState
from app.agent_core.langgraph.tools import (
    ToolRuntime,
    build_application_tool_runtime,
)
from app.core.config import Settings


def _resolve_tools(
    tools: Any | None,
    tool_executor: Any | None,
    tool_definitions: Sequence[Any] | None,
) -> tuple[Any | None, Sequence[Any] | None]:
    """Resolve the accepted tool argument forms into executor and definitions."""
    if tools is None:
        return tool_executor, tool_definitions

    if isinstance(tools, ToolRuntime) or (
        hasattr(tools, "executor") and hasattr(tools, "definitions")
    ):
        if tool_executor is None:
            tool_executor = tools.executor
        if tool_definitions is None:
            tool_definitions = tools.definitions
        return tool_executor, tool_definitions

    if hasattr(tools, "execute"):
        if tool_executor is None:
            tool_executor = tools
        return tool_executor, tool_definitions

    if hasattr(tools, "list_tools") and hasattr(tools, "get_definition"):
        from app.agent_core.orchestrator_tools import (
            build_advertised_tool_definitions,
            build_tool_executor,
        )

        if tool_executor is None:
            tool_executor = build_tool_executor(tools)
        if tool_definitions is None:
            tool_definitions = build_advertised_tool_definitions(tools)
        return tool_executor, tool_definitions

    if tool_definitions is None:
        if isinstance(tools, Sequence) and not isinstance(
            tools, (str, bytes, dict)
        ):
            tool_definitions = tools
        else:
            tool_definitions = [tools]
    return tool_executor, tool_definitions


def build_agent_graph(
    model: ChatModelRunner | None = None,
    tools: Any | None = None,
    *,
    tool_executor: Any | None = None,
    tool_definitions: Sequence[Any] | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> StateGraph:
    """Build the agent graph.

    Without a tool executor the graph keeps the foundation topology:
    ``START -> initialize_messages -> call_model -> END``.

    With a tool executor the graph adds a tool round:
    ``START -> initialize_messages -> call_model -> execute_tools -> call_model``
    with a conditional edge from ``call_model`` to ``execute_tools`` or ``END``.

    The model runner and tool executor are injected at construction time so
    tests can supply deterministic fakes and production can supply Groq-backed
    runners and application-owned tools without global mutable state.
    """
    graph = StateGraph(AgentGraphState)
    tool_executor, tool_definitions = _resolve_tools(
        tools, tool_executor, tool_definitions
    )
    graph.add_node("initialize_messages", initialize_messages)
    graph.add_node(
        "call_model",
        build_call_model_node(
            model,
            tool_definitions=tool_definitions,
            tool_executor=tool_executor,
            max_iterations=max_iterations,
        ),
    )
    graph.add_edge(START, "initialize_messages")
    graph.add_edge("initialize_messages", "call_model")

    if tool_executor is None:
        graph.add_edge("call_model", END)
    else:
        graph.add_node("execute_tools", build_execute_tools_node(tool_executor))
        graph.add_conditional_edges(
            "call_model",
            route_after_model,
            {
                "execute_tools": "execute_tools",
                END: END,
            },
        )
        graph.add_edge("execute_tools", "call_model")

    return graph


def compile_agent_graph(
    model: ChatModelRunner | None = None,
    tools: Any | None = None,
    *,
    tool_executor: Any | None = None,
    tool_definitions: Sequence[Any] | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> CompiledStateGraph:
    """Compile the agent graph without checkpointing."""
    return build_agent_graph(
        model,
        tools,
        tool_executor=tool_executor,
        tool_definitions=tool_definitions,
        max_iterations=max_iterations,
    ).compile()


def build_groq_agent_graph(
    settings: Settings,
    tools: Any | None = None,
    *,
    tenant_id: str | None = None,
    db: Any = None,
    is_approved: bool = False,
    max_iterations: int = MAX_ITERATIONS,
    tool_executor: Any | None = None,
    tool_definitions: Sequence[Any] | None = None,
) -> CompiledStateGraph:
    """Build a graph backed by the configured Groq model.

    When trusted tenant context is supplied and no explicit tool runtime is
    given, the application-owned tool registry is built and bound to that
    context. Approval state is also trusted application context.
    """
    if tools is not None:
        tool_executor, tool_definitions = _resolve_tools(
            tools, tool_executor, tool_definitions
        )
    elif tool_executor is None and (
        tenant_id is not None or db is not None or is_approved
    ):
        runtime = build_application_tool_runtime(
            tenant_id=tenant_id,
            db=db,
            is_approved=is_approved,
        )
        tool_executor = runtime.executor
        if tool_definitions is None:
            tool_definitions = runtime.definitions

    return compile_agent_graph(
        build_groq_chat_model(settings),
        tool_executor=tool_executor,
        tool_definitions=tool_definitions,
        max_iterations=max_iterations,
    )


AGENT_GRAPH = compile_agent_graph()