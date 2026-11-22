import os
import uuid
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app
from app.modules.agent.schema import TicketResolveResponse
from app.database import get_database
from app.modules.integrations.service import issue_integration_credential
from app.database.repository import _canonical_uuid

client = TestClient(app)

# The agent endpoint stores every run in the session store, so these tests
# need the PostgreSQL test database.
pytestmark = pytest.mark.usefixtures("clean_database")


def _make_agent_result(status, answer=None, tool_calls=None, error=None):
    from app.agent_core.agent import AgentResult
    return AgentResult(
        status=status,
        answer=answer,
        tool_calls=tool_calls if tool_calls is not None else [],
        error=error,
    )


def _issue_integration_auth_header(tenant_id=None):
    db = get_database(os.environ["DATABASE_URL"])
    integration, credential, token = issue_integration_credential(
        db, integration_name=f"test-resolve-{uuid.uuid4().hex[:8]}"
    )
    tid = tenant_id or str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) ON CONFLICT (id) DO NOTHING",
            (_canonical_uuid(tid), f"resolve-tenant-{tid[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tid), _canonical_uuid(integration.id)),
        )
    return {"Authorization": f"Bearer {token}"}, tid


@contextmanager
def _authenticated():
    """Yield headers for an authenticated request."""
    headers, _ = _issue_integration_auth_header()
    yield headers


# ---------------------------------------------------------------------------
# Successful agent result
# ---------------------------------------------------------------------------


def test_resolve_endpoint_successful_agent_result():
    """A successful agent run returns 200 with the answer and tool calls."""
    with patch('app.modules.agent.service.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="success",
            answer="To reset your password, visit account settings.",
            tool_calls=[
                {
                    "call_id": "call_1",
                    "tool": "search_support_docs",
                    "arguments": {"query": "password reset"},
                    "result": {"tool": "search_support_docs", "results": [], "count": 0},
                }
            ],
        )
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "How do I reset my password?"},
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["answer"] == "To reset your password, visit account settings."
        assert data["mock_mode"] is False
        assert len(data["tool_calls"]) == 1
        TicketResolveResponse(**data)


def test_resolve_endpoint_validates_response_schema():
    """The response conforms to TicketResolveResponse."""
    with patch('app.modules.agent.service.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="success",
            answer="Final answer.",
        )
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Hello."},
                headers=headers,
            )
        assert response.status_code == 200
        TicketResolveResponse(**response.json())


# ---------------------------------------------------------------------------
# Mock mode when no API key is configured
# ---------------------------------------------------------------------------


def test_resolve_endpoint_mock_mode():
    """When no API key is configured, the endpoint reports mock_mode=True."""
    with patch('app.modules.agent.service.groq_client', None):
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "How do I reset my password?"},
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "mock"
        assert data["mock_mode"] is True
        assert data["answer"] is not None
        assert "API key" in data["answer"]
        TicketResolveResponse(**data)


# ---------------------------------------------------------------------------
# Request validation failures
# ---------------------------------------------------------------------------


def test_resolve_endpoint_empty_message():
    """An empty message returns 422."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": ""},
            headers=headers,
        )
    assert response.status_code == 422


def test_resolve_endpoint_whitespace_message():
    """A whitespace-only message returns 422."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={"message": "   "},
            headers=headers,
        )
    assert response.status_code == 422


def test_resolve_endpoint_missing_message():
    """A missing message field returns 422."""
    with _authenticated() as headers:
        response = client.post(
            "/api/v1/tickets/agent/run",
            json={},
            headers=headers,
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Safe handling of provider errors
# ---------------------------------------------------------------------------


def test_resolve_endpoint_provider_failure_returns_502():
    """A provider failure returns 502 with a safe message."""
    with patch('app.modules.agent.service.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="provider_failure",
            error="Provider service is temporarily unavailable.",
        )
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Hello."},
                headers=headers,
            )
        assert response.status_code == 502
        detail = response.json().get("detail", "")
        assert "temporarily unavailable" in detail
        assert "api_key" not in detail.lower()


def test_resolve_endpoint_unexpected_agent_error_returns_generic_500():
    """An unexpected graph execution error uses the generic failure response."""
    with patch('app.modules.agent.service.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="internal_error",
            error="Agent service error.",
        )
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Hello."},
                headers=headers,
            )
    assert response.status_code == 500
    assert response.json()["detail"] == "Agent service error."


# ---------------------------------------------------------------------------
# Iteration-limit outcome
# ---------------------------------------------------------------------------


def test_resolve_endpoint_max_iterations_reached():
    """Hitting the iteration limit returns a structured non-success result."""
    with patch('app.modules.agent.service.run_agent') as mock_run_agent:
        mock_run_agent.return_value = _make_agent_result(
            status="max_iterations_reached",
            answer=None,
            tool_calls=[],
            error="Agent reached the maximum of 10 iterations without producing a final answer.",
        )
        with _authenticated() as headers:
            response = client.post(
                "/api/v1/tickets/agent/run",
                json={"message": "Keep searching forever."},
                headers=headers,
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "max_iterations_reached"
        assert data["error"] is not None
        TicketResolveResponse(**data)


def test_agent_endpoint_executes_langgraph_application_path():
    """The active endpoint reaches the LangGraph-backed application path."""
    from langchain_core.messages import AIMessage, HumanMessage
    from app.agent_core.langgraph import AgentGraphResult

    sentinel_graph = object()
    graph_result = AgentGraphResult(
        messages=[
            HumanMessage(content="Graph question"),
            AIMessage(content="Graph answer"),
        ],
        answer="Graph answer",
        error=None,
    )

    with patch('app.modules.agent.service.groq_client', object()):
        with patch(
            'app.modules.agent.service.compile_agent_graph',
            return_value=sentinel_graph,
        ) as mock_compile:
            with patch(
                'app.modules.agent.service.run_agent_graph',
                return_value=graph_result,
            ) as mock_run:
                with _authenticated() as headers:
                    response = client.post(
                        "/api/v1/tickets/agent/run",
                        json={"message": "Graph question"},
                        headers=headers,
                    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["answer"] == "Graph answer"
    assert mock_compile.called
    compile_kwargs = mock_compile.call_args.kwargs
    assert compile_kwargs["tool_executor"] is not None
    assert compile_kwargs["tool_definitions"]
    assert mock_run.call_args.args[0] == "Graph question"
    assert mock_run.call_args.kwargs["graph"] is sentinel_graph


# ---------------------------------------------------------------------------
# Existing /analyze endpoint continues to work
# ---------------------------------------------------------------------------


def test_analyze_endpoint_still_works():
    """The existing /analyze endpoint is unaffected by the new resolve route."""
    with patch('app.modules.analyze.route.groq_client', None):
        response = client.post(
            "/api/v1/tickets/analyze",
            json={"message": "Hello, I need help."},
        )
        assert response.status_code == 200
        json_data = response.json()
        assert "category" in json_data


def test_health_endpoint_still_works():
    """The health endpoint is unaffected."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
