import json
import logging
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.agent_core.agent import AgentResult
from app.agent_core.langgraph import (
    MALFORMED_TOOL_CALL_IDS_ERROR,
    MAX_ITERATIONS,
    MAX_ITERATIONS_ERROR,
    PROVIDER_ERROR,
    GroqChatModelRunner,
    build_application_tool_runtime,
    compile_agent_graph,
    run_agent_graph,
)
from app.ai.provider import get_groq_client
from app.core.config import Settings
from app.database import get_database
from app.modules.agent.schema import TicketResolveRequest
from app.domain.agent.services import (
    prepare_session_store,
    resolve_session_id,
    load_session_history,
    load_session_memory,
    persist_agent_run,
    final_answer_message_id,
    store_run_memory,
)

settings = Settings()
groq_client = get_groq_client(settings)

logger = logging.getLogger("agent.service")

PROVIDER_ERROR_MESSAGE = "Provider service is temporarily unavailable."
EMPTY_RESPONSE_ERROR = "The model did not produce a final answer or a tool call."


@dataclass(frozen=True)
class AgentRunOutcome:
    """Result of a completed agent workflow."""
    result: AgentResult
    session_id: str
    transcript_stored: bool


def _tool_calls_from_messages(messages) -> list[dict]:
    """Build the application's tool-call records from graph messages.

    Tool messages are matched to the assistant tool call that produced them by
    ``tool_call_id``. A queue is used per id so repeated ids across cycles
    cannot overwrite an earlier result.
    """
    pending = defaultdict(deque)
    for message in messages:
        if isinstance(message, ToolMessage) and message.tool_call_id:
            pending[message.tool_call_id].append(message)

    calls: list[dict] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        for tool_call in getattr(message, "tool_calls", None) or []:
            call_id = tool_call.get("id")
            queue = pending.get(call_id)
            if not queue:
                continue
            tool_message = queue.popleft()
            try:
                result = json.loads(tool_message.content)
            except (TypeError, ValueError):
                result = {}
            if not isinstance(result, dict):
                result = {}
            calls.append(
                {
                    "call_id": call_id,
                    "tool": tool_call.get("name"),
                    "arguments": tool_call.get("args") or {},
                    "result": result,
                }
            )
    return calls


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _history_to_messages(history) -> list[BaseMessage]:
    """Convert application/provider history items into LangGraph messages.

    The session store produces provider Responses API input items. LangGraph's
    message reducer needs message objects, so function calls and outputs are
    converted to ``AIMessage``/``ToolMessage`` pairs here. Application-owned
    history is never passed through model-generated tool arguments.
    """
    messages: list[BaseMessage] = []
    for item in history or []:
        if isinstance(item, BaseMessage):
            messages.append(item)
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        if item_type == "function_call":
            try:
                arguments = json.loads(item.get("arguments") or "{}")
            except (TypeError, ValueError):
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name") or ""
            messages.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": arguments,
                            "id": str(call_id),
                        }
                    ],
                )
            )
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id") or item.get("id") or ""
            name = item.get("name") or ""
            messages.append(
                ToolMessage(
                    content=str(item.get("output") or ""),
                    tool_call_id=str(call_id),
                    name=name,
                )
            )
            continue

        role = item.get("role")
        content = _content_text(item.get("content"))
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))

    return messages


def _graph_result_to_agent_result(result, max_iterations: int) -> AgentResult:
    """Map the LangGraph result onto the existing application contract."""
    tool_calls = _tool_calls_from_messages(result.messages)

    if result.answer:
        return AgentResult(
            status="success",
            answer=result.answer,
            tool_calls=tool_calls,
        )

    if result.error in {PROVIDER_ERROR, MALFORMED_TOOL_CALL_IDS_ERROR}:
        return AgentResult(
            status="provider_failure",
            tool_calls=tool_calls,
            error=PROVIDER_ERROR_MESSAGE,
        )

    if result.error == MAX_ITERATIONS_ERROR:
        return AgentResult(
            status="max_iterations_reached",
            tool_calls=tool_calls,
            error=(
                f"Agent reached the maximum of {max_iterations} iterations "
                "without producing a final answer."
            ),
        )

    if result.error:
        return AgentResult(
            status="max_iterations_reached",
            tool_calls=tool_calls,
            error=result.error,
        )

    return AgentResult(
        status="max_iterations_reached",
        tool_calls=tool_calls,
        error=EMPTY_RESPONSE_ERROR,
    )


def run_agent(
    message: str,
    settings: Settings,
    client=None,
    max_iterations: int = MAX_ITERATIONS,
    history: list[dict] | None = None,
    tenant_id: str | None = None,
    db=None,
    is_approved: bool = False,
) -> AgentResult:
    """Run the support agent through the LangGraph application graph.

    This is the active application execution path. Session, history, memory,
    transcript, tenant, and approval responsibilities remain in the caller.
    The trusted tenant id, database handle, and approval state are bound into
    the application-owned tool runtime; model-supplied tool arguments cannot
    override them.
    """
    if client is None:
        return AgentResult(
            status="mock",
            answer=(
                "Agent mode requires a configured Groq API key. "
                "No tool calls were made."
            ),
            tool_calls=[],
        )

    try:
        runtime = build_application_tool_runtime(
            tenant_id=tenant_id,
            db=db,
            is_approved=is_approved,
        )
        graph = compile_agent_graph(
            GroqChatModelRunner(client, settings.groq_model),
            tool_executor=runtime.executor,
            tool_definitions=runtime.definitions,
            max_iterations=max_iterations,
            max_tool_calls=runtime.max_tool_calls,
        )
        graph_result = run_agent_graph(
            message,
            graph=graph,
            history=_history_to_messages(history),
        )
    except Exception as exc:  # noqa: BLE001 - safe application boundary
        logger.exception(
            "LangGraph agent execution failed: %s", type(exc).__name__
        )
        return AgentResult(
            status="provider_failure",
            tool_calls=[],
            error=PROVIDER_ERROR_MESSAGE,
        )

    return _graph_result_to_agent_result(graph_result, max_iterations)


def run_ticket_service(
    request: TicketResolveRequest,
    http_request=None,
) -> AgentRunOutcome:
    """Coordinate session setup, agent execution, and persistence."""
    run_id = str(uuid.uuid4())

    # 1. Prepare the database and resolve the session.
    database = get_database(settings.database_url)
    prepare_session_store(database)
    tenant_id = None
    if http_request is not None:
        tenant = getattr(http_request.state, "tenant", None)
        if tenant is not None:
            tenant_id = getattr(tenant, "id", None)
    session_id = resolve_session_id(database, request.session_id, tenant_id=tenant_id)

    # 2. Load prior history and relevant memories.
    history = load_session_history(
        database,
        session_id,
        max_messages=settings.agent_max_history_messages,
        exclude_correlation_ids={run_id},
    )

    memory_items = load_session_memory(
        database,
        session_id,
        request.message,
        max_items=settings.agent_max_memory_items,
        max_chars=settings.agent_max_memory_chars,
    )

    agent_history = [*memory_items, *history]

    # 3. Run the agent through LangGraph.
    result = run_agent(
        request.message,
        settings,
        groq_client,
        history=agent_history,
        tenant_id=tenant_id,
        db=database,
    )

    # 4. Persist the transcript before attempting to store its memory.
    stored_messages = persist_agent_run(
        database,
        session_id,
        request.message,
        result,
        correlation_id=run_id,
    )

    transcript_stored = bool(stored_messages)

    # 5. Store memory only when a persisted final answer exists.
    answer_message_id = final_answer_message_id(stored_messages)

    if answer_message_id is not None:
        store_run_memory(
            database,
            session_id,
            result,
            answer_message_id,
        )

    # 6. Return the outcome; the controller handles HTTP responses.
    return AgentRunOutcome(
        result=result,
        session_id=session_id,
        transcript_stored=transcript_stored,
    )