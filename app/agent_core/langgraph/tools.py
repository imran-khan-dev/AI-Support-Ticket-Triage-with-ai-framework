"""Tool execution boundary for the LangGraph agent foundation.

The application owns tenant, database, and approval context. This module adapts
the existing Agent Orchestrator tool registry/executor to LangGraph messages so
the graph can execute provider-requested tool calls without owning any of those
trusted dependencies itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from langchain_core.messages import ToolMessage

from app.agent_core.orchestrator_tools import (
    build_advertised_tool_definitions,
    build_tool_executor,
    build_tool_registry,
)

DEFAULT_MAX_TOOL_CALLS = 20
TOOL_CALL_LIMIT_ERROR = "tool_failure"
TOOL_CALL_LIMIT_MESSAGE = "Tool call limit reached."
MALFORMED_TOOL_CALL_IDS_ERROR = (
    "The model returned malformed tool-call identifiers."
)


class MalformedToolCallIdsError(ValueError):
    """Raised when tool-call IDs are missing, invalid, or duplicated."""


class ToolExecutor(Protocol):
    """Minimal executor contract used by the graph's tool node."""

    def execute(self, tool_call: Any) -> Any:
        """Execute one orchestrator-style tool call and return a result."""


@dataclass(frozen=True)
class ToolRuntime:
    """An executor paired with the tool definitions advertised to the model."""

    executor: ToolExecutor
    definitions: tuple[Any, ...] = ()
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS


def _validate_max_tool_calls(max_tool_calls: int) -> None:
    if (
        not isinstance(max_tool_calls, int)
        or isinstance(max_tool_calls, bool)
        or max_tool_calls < 1
    ):
        raise ValueError("max_tool_calls must be a positive integer.")


def build_application_tool_runtime(
    *,
    tenant_id: str | None = None,
    db: Any = None,
    is_approved: bool = False,
    config: Any = None,
) -> ToolRuntime:
    """Build the application-owned tool runtime for a graph invocation.

    ``tenant_id``, ``db``, and ``is_approved`` are trusted server-side context.
    They are bound into the tool handlers here and can never be overridden by
    model-supplied arguments.
    """
    registry = build_tool_registry(
        tenant_id=tenant_id,
        db=db,
        is_approved=is_approved,
        config=config,
    )
    executor = build_tool_executor(registry, config=config)
    definitions = tuple(build_advertised_tool_definitions(registry))
    max_tool_calls = getattr(config, "max_tool_calls", DEFAULT_MAX_TOOL_CALLS)
    _validate_max_tool_calls(max_tool_calls)
    return ToolRuntime(
        executor=executor,
        definitions=definitions,
        max_tool_calls=max_tool_calls,
    )


def _safe_tool_error(tool_name: str, error_message: str) -> dict[str, Any]:
    """Build a safe ``execute_tool``-style error result.

    This mirrors :func:`app.agent_core.agent._safe_tool_error` while keeping
    the LangGraph layer independent of the legacy agent loop.
    """
    lowered = (error_message or "").lower()
    if "toolnotfounderror" in lowered or "not registered" in lowered:
        return {
            "tool": tool_name,
            "error": "unknown_tool",
            "message": f"Tool '{tool_name}' is not registered.",
            "results": [],
        }
    if "toolvalidationerror" in lowered or "invalid" in lowered:
        return {
            "tool": tool_name,
            "error": "invalid_arguments",
            "message": f"Invalid arguments for tool '{tool_name}'.",
            "results": [],
        }
    return {
        "tool": tool_name,
        "error": "tool_failure",
        "message": f"Tool '{tool_name}' failed to execute.",
        "results": [],
    }


def _coerce_tool_call(tool_call: dict[str, Any]) -> Any:
    """Convert a LangChain tool-call dict to the orchestrator ToolCall model."""
    from agent_orchestrator.tools.models import ToolCall

    call_id = tool_call.get("id") or tool_call.get("call_id") or "unknown"
    name = tool_call.get("name") or "unknown"
    arguments = tool_call.get("args")
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolCall(id=str(call_id), name=str(name), arguments=arguments)


def _tool_call_id(raw_call: Any) -> Any:
    if not isinstance(raw_call, dict):
        return None
    return raw_call.get("id") or raw_call.get("call_id")


def validate_tool_call_ids(tool_calls: Sequence[dict[str, Any]]) -> None:
    """Reject missing or duplicate tool-call IDs before any call executes.

    A missing ID cannot be correlated with an assistant tool call, and a
    duplicate ID makes multiple tool results ambiguous. Rejecting the whole
    response is safer than guessing an ID or executing a subset.
    """
    seen: set[str] = set()
    for raw_call in tool_calls:
        call_id = _tool_call_id(raw_call)
        if not isinstance(call_id, str) or not call_id.strip():
            raise MalformedToolCallIdsError(MALFORMED_TOOL_CALL_IDS_ERROR)
        if call_id in seen:
            raise MalformedToolCallIdsError(MALFORMED_TOOL_CALL_IDS_ERROR)
        seen.add(call_id)


def _result_to_dict(tool_call: Any, result: Any) -> dict[str, Any]:
    """Normalize a ToolResult into the safe dict sent back to the model."""
    if getattr(result, "success", False):
        output = getattr(result, "output", None)
        if isinstance(output, dict):
            return output
        return {
            "tool": tool_call.name,
            "error": "tool_failure",
            "message": "Tool returned a non-object result.",
            "results": [],
        }
    return _safe_tool_error(tool_call.name, getattr(result, "error", "") or "")


def _tool_call_limit_result(tool_call: Any) -> dict[str, Any]:
    return {
        "tool": tool_call.name,
        "error": TOOL_CALL_LIMIT_ERROR,
        "message": TOOL_CALL_LIMIT_MESSAGE,
        "results": [],
    }


def execute_tool_calls(
    tool_calls: Sequence[dict[str, Any]],
    executor: ToolExecutor,
    *,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    used_tool_calls: int = 0,
) -> list[ToolMessage]:
    """Execute model-requested tool calls and return LangChain tool messages.

    Every call produces exactly one ``ToolMessage`` in the order requested.
    Calls beyond the per-invocation budget are reported safely and never sent
    to the executor. Invalid and unknown tool calls still consume the budget
    because they pass through the executor boundary. Tool failures are
    converted to safe, secret-free result dictionaries instead of aborting the
    graph.
    """
    _validate_max_tool_calls(max_tool_calls)
    validate_tool_call_ids(tool_calls)

    used_tool_calls = max(0, int(used_tool_calls or 0))
    remaining = max(0, max_tool_calls - used_tool_calls)

    messages: list[ToolMessage] = []
    for index, raw_call in enumerate(tool_calls):
        tool_call = _coerce_tool_call(raw_call)
        call_id = _tool_call_id(raw_call)

        if index >= remaining:
            result_dict = _tool_call_limit_result(tool_call)
        else:
            try:
                result = executor.execute(tool_call)
            except Exception as exc:  # noqa: BLE001 - safe boundary
                result_dict = _safe_tool_error(
                    tool_call.name, f"{type(exc).__name__}: {exc}"
                )
            else:
                result_dict = _result_to_dict(tool_call, result)

        messages.append(
            ToolMessage(
                content=json.dumps(result_dict),
                tool_call_id=str(call_id),
                name=tool_call.name,
            )
        )
    return messages


# Descriptive aliases used by callers that prefer registry-oriented naming.
build_tool_runtime = build_application_tool_runtime

__all__ = [
    "DEFAULT_MAX_TOOL_CALLS",
    "MALFORMED_TOOL_CALL_IDS_ERROR",
    "MalformedToolCallIdsError",
    "TOOL_CALL_LIMIT_ERROR",
    "TOOL_CALL_LIMIT_MESSAGE",
    "ToolExecutor",
    "ToolRuntime",
    "build_application_tool_runtime",
    "build_tool_runtime",
    "execute_tool_calls",
    "validate_tool_call_ids",
]