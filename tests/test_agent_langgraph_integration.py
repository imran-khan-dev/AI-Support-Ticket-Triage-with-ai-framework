"""Focused integration tests for the LangGraph-backed agent service."""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agent_core.agent import AgentResult
from app.agent_core.langgraph import (
    MALFORMED_TOOL_CALL_IDS_ERROR,
    MAX_ITERATIONS_ERROR,
    PROVIDER_ERROR,
    AgentGraphResult,
)
from app.core.config import Settings
from app.modules.agent.service import run_agent


def _settings():
    return Settings(
        _env_file=None,
        groq_api_key="test_key",
        groq_model="openai/gpt-oss-20b",
        groq_base_url="https://api.groq.com/openai/v1",
    )


class _SdkFunctionCall:
    def __init__(self, call_id, name, arguments):
        self.type = "function_call"
        self.call_id = call_id
        self.name = name
        self.arguments = arguments


class _SdkMessage:
    def __init__(self, text):
        self.type = "message"
        self.content = [SimpleNamespace(type="output_text", text=text)]


class _FakeResponse:
    def __init__(self, output=None, output_text=""):
        self.output = output or []
        self.output_text = output_text


class _FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("fake Groq responses exhausted")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.responses = _FakeResponses(responses)


def _agent_result(answer="Done.", tool_calls=None):
    return AgentGraphResult(
        messages=[HumanMessage(content="question"), AIMessage(content=answer)],
        answer=answer,
        error=None,
    )


def test_langgraph_search_knowledge_uses_trusted_tenant_and_maps_tool_calls():
    fake_chunk = SimpleNamespace(
        id="chunk-1",
        content="password reset",
        metadata={"filename": "doc.txt"},
        chunk_index=0,
    )
    fake_result = SimpleNamespace(chunk=fake_chunk, score=0.9, score_type="bm25")
    fake_service = Mock()
    fake_service.search.return_value = [fake_result]

    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "search_knowledge",
                        '{"query": "password reset"}',
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    with patch(
        "app.agent_core.tools.get_knowledge_service", return_value=fake_service
    ):
        result = run_agent(
            "How do I reset my password?",
            _settings(),
            fake_client,
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "success"
    assert result.answer == "Done."
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["call_id"] == "call_1"
    assert call["tool"] == "search_knowledge"
    assert call["arguments"] == {"query": "password reset"}
    assert call["result"]["count"] == 1
    assert fake_service.search.call_args.args[0] == "trusted-tenant"


def test_langgraph_rejects_model_supplied_tenant_id():
    fake_service = Mock()
    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "search_knowledge",
                        '{"query": "password reset", "tenant_id": "llm-tenant"}',
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Done.")],
                output_text="Done.",
            ),
        ]
    )

    with patch(
        "app.agent_core.tools.get_knowledge_service", return_value=fake_service
    ):
        result = run_agent(
            "How do I reset my password?",
            _settings(),
            fake_client,
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "success"
    assert result.tool_calls[0]["result"]["error"] == "invalid_arguments"
    fake_service.search.assert_not_called()


def test_langgraph_unapproved_support_case_is_refused():
    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "create_support_case",
                        '{"subject": "s", "description": "d", '
                        '"priority": "low", "category": "billing"}',
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("I cannot create that case.")],
                output_text="I cannot create that case.",
            ),
        ]
    )

    result = run_agent(
        "Create a case.",
        _settings(),
        fake_client,
        tenant_id="trusted-tenant",
        db=object(),
        is_approved=False,
    )

    assert result.status == "success"
    assert result.tool_calls[0]["result"]["error"] == "approval_required"


def test_langgraph_approved_support_case_executes():
    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "create_support_case",
                        '{"subject": "s", "description": "d", '
                        '"priority": "low", "category": "billing"}',
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("Created.")],
                output_text="Created.",
            ),
        ]
    )

    result = run_agent(
        "Create a case.",
        _settings(),
        fake_client,
        tenant_id="trusted-tenant",
        db=object(),
        is_approved=True,
    )

    assert result.status == "success"
    payload = result.tool_calls[0]["result"]
    assert payload["tool"] == "create_support_case"
    assert payload["count"] == 1
    assert payload["results"][0]["status"] == "created"


def test_langgraph_provider_failure_is_mapped_safely():
    graph_result = AgentGraphResult(messages=[], answer=None, error=PROVIDER_ERROR)

    with patch(
        "app.modules.agent.service.run_agent_graph", return_value=graph_result
    ):
        result = run_agent(
            "Hello.",
            _settings(),
            object(),
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "provider_failure"
    assert result.error == "Provider service is temporarily unavailable."


def test_langgraph_iteration_limit_is_classified_as_iteration_limit():
    graph_result = AgentGraphResult(
        messages=[],
        answer=None,
        error=MAX_ITERATIONS_ERROR,
    )

    with patch(
        "app.modules.agent.service.run_agent_graph", return_value=graph_result
    ):
        result = run_agent(
            "Keep searching.",
            _settings(),
            object(),
            max_iterations=3,
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "max_iterations_reached"
    assert result.error == (
        "Agent reached the maximum of 3 iterations without producing a final answer."
    )


def test_langgraph_malformed_tool_call_ids_remain_provider_failure():
    graph_result = AgentGraphResult(
        messages=[],
        answer=None,
        error=MALFORMED_TOOL_CALL_IDS_ERROR,
    )

    with patch(
        "app.modules.agent.service.run_agent_graph", return_value=graph_result
    ):
        result = run_agent(
            "Hello.",
            _settings(),
            object(),
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "provider_failure"
    assert result.error == "Provider service is temporarily unavailable."


def test_langgraph_unexpected_graph_error_is_not_iteration_limit():
    graph_result = AgentGraphResult(
        messages=[],
        answer=None,
        error="unexpected graph failure",
    )

    with patch(
        "app.modules.agent.service.run_agent_graph", return_value=graph_result
    ):
        result = run_agent(
            "Hello.",
            _settings(),
            object(),
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "internal_error"
    assert result.status != "max_iterations_reached"
    assert result.error == "Agent service error."
    assert "unexpected graph failure" not in (result.error or "")


def test_langgraph_unexpected_exception_is_mapped_safely():
    with patch(
        "app.modules.agent.service.run_agent_graph",
        side_effect=RuntimeError("secret detail"),
    ):
        result = run_agent(
            "Hello.",
            _settings(),
            object(),
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "internal_error"
    assert "secret detail" not in (result.error or "")


def test_langgraph_receives_application_history():
    history = [{"role": "user", "content": "An older question."}]
    graph_result = _agent_result()

    with patch(
        "app.modules.agent.service.run_agent_graph", return_value=graph_result
    ) as mock_run:
        result = run_agent(
            "The current question.",
            _settings(),
            object(),
            history=history,
            tenant_id="trusted-tenant",
            db=object(),
        )

    assert result.status == "success"
    passed_history = mock_run.call_args.kwargs["history"]
    assert [type(message).__name__ for message in passed_history] == ["HumanMessage"]
    assert passed_history[0].content == "An older question."


def test_langgraph_builds_a_fresh_graph_for_each_request():
    graph_result = _agent_result()

    with patch(
        "app.modules.agent.service.compile_agent_graph", return_value=object()
    ) as mock_compile:
        with patch(
            "app.modules.agent.service.run_agent_graph", return_value=graph_result
        ):
            first = run_agent(
                "first",
                _settings(),
                object(),
                tenant_id="tenant-1",
                db=object(),
            )
            second = run_agent(
                "second",
                _settings(),
                object(),
                tenant_id="tenant-2",
                db=object(),
            )

    assert first.status == "success"
    assert second.status == "success"
    assert mock_compile.call_count == 2


def test_langgraph_model_cannot_self_approve_support_case():
    fake_client = _FakeClient(
        [
            _FakeResponse(
                output=[
                    _SdkFunctionCall(
                        "call_1",
                        "create_support_case",
                        '{"subject": "s", "description": "d", '
                        '"priority": "low", "category": "billing", '
                        '"is_approved": true}',
                    )
                ]
            ),
            _FakeResponse(
                output=[_SdkMessage("I cannot create that case.")],
                output_text="I cannot create that case.",
            ),
        ]
    )

    result = run_agent(
        "Create a case.",
        _settings(),
        fake_client,
        tenant_id="trusted-tenant",
        db=object(),
        is_approved=False,
    )

    payload = result.tool_calls[0]["result"]
    assert payload["error"] in {"approval_required", "invalid_arguments"}
    assert payload["results"] == []


@pytest.mark.usefixtures("clean_database")
def test_approved_support_case_executes_through_application_approval_flow(database):
    from app.core.security import generate_approver_token
    from app.database import create_action
    from app.modules.approvals.decision_service import decide_approval
    from app.modules.approvals.execution_service import execute_approved_action
    from app.modules.approvals.service import register_approver

    secret = "test-only-secret-for-approver-credentials-0123456789"
    token = generate_approver_token()
    register_approver(
        database,
        display_name="Approver",
        role="supervisor",
        token=token,
        secret=secret,
    )

    action = create_action(
        database,
        request_key=f"req-{uuid.uuid4()}",
        tool_name="create_support_case",
        effect_class="gated_tool",
        args=json.dumps(
            {
                "subject": "s",
                "description": "d",
                "priority": "low",
                "category": "billing",
            },
            sort_keys=True,
        ),
        requested_by="agent",
        correlation_id=str(uuid.uuid4()),
    )

    decide_approval(
        database,
        action_id=action.id,
        decision="approved",
        approver_token=token,
        secret=secret,
    )
    result = execute_approved_action(database, action_id=action.id)

    assert result.executed is True
    assert result.result[0]["status"] == "created"