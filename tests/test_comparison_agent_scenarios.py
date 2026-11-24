"""Tests for the agent comparison scenario dataset (app/comparison/agent_scenarios.py).

All scenarios are deterministic, implementation-agnostic, and use scripted
providers plus controlled tool handlers.  No external LLM or paid provider is
required.
"""

from __future__ import annotations

from app.comparison.agent_scenarios import (
    AgentEvaluationDataset,
    AgentScenarioSpec,
    CONTROLLED_TOOL_HANDLERS,
    SCENARIO_CATEGORIES,
    ScriptExhaustion,
    ScriptedStep,
    ScriptedToolCall,
    TOOL_SPECS,
    build_scenarios,
    load_agent_dataset,
    TRUSTED_TENANT,
    SPOOFED_TENANT,
)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def test_build_scenarios_returns_expected_count():
    scenarios = build_scenarios()
    assert len(scenarios) == 14
    ids = [s.scenario_id for s in scenarios]
    assert len(set(ids)) == len(ids), "scenario ids must be unique"


def test_build_scenarios_known_ids():
    scenarios = build_scenarios()
    ids = [s.scenario_id for s in scenarios]
    expected = [
        "agent-simple-direct-response",
        "agent-single-tool-success",
        "agent-multiple-sequential-tools",
        "agent-conditional-tool-routing",
        "agent-tool-execution-failure",
        "agent-provider-failure",
        "agent-tool-retry-behaviour",
        "agent-maximum-tool-call-limit",
        "agent-loop-prevention",
        "agent-missing-invalid-arguments",
        "agent-tenant-isolation",
        "agent-approval-required",
        "agent-cancellation",
        "agent-final-answer-after-tool",
    ]
    assert ids == expected


def test_load_agent_dataset_versioned():
    ds = load_agent_dataset()
    assert ds.dataset_id == "agent-eval-v1"
    assert ds.version == "1.0.0"
    assert ds.name == "agent-evaluation"
    assert len(ds.scenarios) == 14
    assert ds.metadata["total_scenarios"] == 14
    assert isinstance(ds.metadata["categories"], list)


# ---------------------------------------------------------------------------
# Scenario spec validation
# ---------------------------------------------------------------------------

def test_scenario_spec_required_fields():
    spec = AgentScenarioSpec(
        scenario_id="t1",
        category="cat",
        user_request="do something",
    )
    assert spec.scenario_id == "t1"
    assert spec.expected_outcome == "success"
    assert spec.max_iterations == 10
    assert spec.max_tool_calls == 20
    assert spec.supported_implementations == ["custom", "framework"]


def test_scenario_spec_invalid_max_iterations():
    try:
        AgentScenarioSpec(
            scenario_id="t1",
            category="cat",
            user_request="x",
            max_iterations=0,
        )
        raise AssertionError("expected validation error")
    except Exception:
        pass


def test_scenario_spec_invalid_max_tool_calls():
    try:
        AgentScenarioSpec(
            scenario_id="t1",
            category="cat",
            user_request="x",
            max_tool_calls=0,
        )
        raise AssertionError("expected validation error")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Scenario coverage
# ---------------------------------------------------------------------------

def test_scenario_categories_non_empty():
    assert SCENARIO_CATEGORIES
    assert "simple_direct_response" in SCENARIO_CATEGORIES
    assert "tenant_isolation" in SCENARIO_CATEGORIES
    assert "approval_required" in SCENARIO_CATEGORIES


def test_tenant_constants():
    assert TRUSTED_TENANT == "tenant-alpha"
    assert SPOOFED_TENANT == "tenant-beta"
    assert TRUSTED_TENANT != SPOOFED_TENANT


def test_script_exhaustion_enum_values():
    assert ScriptExhaustion.FINAL_ANSWER == "final_answer"
    assert ScriptExhaustion.REPEAT_LAST == "repeat_last"
    assert ScriptExhaustion.PROVIDER_ERROR == "provider_error"


# ---------------------------------------------------------------------------
# Tool specs
# ---------------------------------------------------------------------------

def test_tool_specs_known_names():
    expected = {
        "search_knowledge",
        "lookup_ticket_history",
        "create_support_case",
        "fail_tool",
        "retry_tool",
        "echo_tool",
    }
    assert set(TOOL_SPECS.keys()) == expected


def test_tool_specs_additional_properties_false():
    for name, spec in TOOL_SPECS.items():
        assert spec.parameters_schema.get("additionalProperties") is False, name


def test_controlled_tool_handlers_match_specs():
    assert set(CONTROLLED_TOOL_HANDLERS.keys()) == set(TOOL_SPECS.keys())


# ---------------------------------------------------------------------------
# Individual scenario expectations
# ---------------------------------------------------------------------------

def _find(scenario_id: str) -> AgentScenarioSpec:
    for s in build_scenarios():
        if s.scenario_id == scenario_id:
            return s
    raise AssertionError(f"scenario not found: {scenario_id}")


def test_simple_direct_response():
    s = _find("agent-simple-direct-response")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 0
    assert s.expected_tool_names == []
    assert s.expected_final_answer_contains == "help"
    assert "no_unauthorized_tool_execution" in s.safety_constraints


def test_single_tool_success():
    s = _find("agent-single-tool-success")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["search_knowledge"]
    assert s.expected_final_answer_contains == "Forgot password"
    assert len(s.script) == 2


def test_multiple_sequential_tools():
    s = _find("agent-multiple-sequential-tools")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 2
    assert s.expected_tool_names == ["search_knowledge", "lookup_ticket_history"]


def test_conditional_tool_routing():
    s = _find("agent-conditional-tool-routing")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["search_knowledge"]


def test_tool_execution_failure():
    s = _find("agent-tool-execution-failure")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["fail_tool"]
    assert s.expected_tool_failures == 1
    assert s.expected_final_answer_contains == "failed"


def test_provider_failure():
    s = _find("agent-provider-failure")
    assert s.expected_outcome == "provider_failure"
    assert s.expected_tool_call_count == 0
    assert s.provider_retry_max_attempts == 1
    assert "provider_failure_is_safe" in s.safety_constraints


def test_tool_retry_behaviour():
    s = _find("agent-tool-retry-behaviour")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["retry_tool"]
    assert s.tool_retry_max_attempts == 2


def test_maximum_tool_call_limit():
    s = _find("agent-maximum-tool-call-limit")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 3
    assert s.max_tool_calls == 3
    assert s.expected_tool_names == ["search_knowledge"]


def test_loop_prevention():
    s = _find("agent-loop-prevention")
    assert s.expected_outcome == "max_iterations_reached"
    assert s.expected_tool_call_count == 3
    assert s.max_iterations == 3
    assert s.script_exhaustion == ScriptExhaustion.REPEAT_LAST
    assert "loop_prevention_enforced" in s.safety_constraints


def test_missing_invalid_arguments():
    s = _find("agent-missing-invalid-arguments")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["create_support_case"]
    assert s.expected_tool_failures == 1
    assert s.is_approved is True
    assert "invalid_argument_rejection" in s.safety_constraints


def test_tenant_isolation():
    s = _find("agent-tenant-isolation")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["search_knowledge"]
    assert s.trusted_tenant_id == TRUSTED_TENANT
    assert s.spoofed_tenant_id == SPOOFED_TENANT
    assert "cross_tenant_access_prevention" in s.safety_constraints
    assert "tenant_id_from_trusted_context" in s.safety_constraints


def test_approval_required():
    s = _find("agent-approval-required")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["create_support_case"]
    assert s.expected_tool_failures == 1
    assert s.is_approved is False
    assert s.requires_approval is True
    assert "approval_gated_action_enforcement" in s.safety_constraints


def test_cancellation_custom_only():
    s = _find("agent-cancellation")
    assert s.cancellation_supported is True
    assert s.supported_implementations == ["custom"]
    assert s.expected_outcome == "success"


def test_final_answer_after_tool():
    s = _find("agent-final-answer-after-tool")
    assert s.expected_outcome == "success"
    assert s.expected_tool_call_count == 1
    assert s.expected_tool_names == ["search_knowledge"]
    assert s.expected_final_answer_contains == "5 business days"


# ---------------------------------------------------------------------------
# Controlled tool handlers
# ---------------------------------------------------------------------------

def test_handle_search_knowledge_trusted_tenant():
    result = CONTROLLED_TOOL_HANDLERS["search_knowledge"](
        {"query": "password"}, tenant_id=TRUSTED_TENANT
    )
    assert result["tenant_id"] == TRUSTED_TENANT
    assert result["count"] >= 1
    assert any("password" in d["content"].lower() for d in result["results"])


def test_handle_search_knowledge_tenant_isolation():
    """Results are bound to the trusted tenant, never the spoofed one."""
    result = CONTROLLED_TOOL_HANDLERS["search_knowledge"](
        {"query": "password"}, tenant_id=TRUSTED_TENANT
    )
    assert result["tenant_id"] == TRUSTED_TENANT
    assert not any("beta_private" in d["document"] for d in result["results"])


def test_handle_create_support_case_requires_approval():
    result = CONTROLLED_TOOL_HANDLERS["create_support_case"](
        {"subject": "s", "description": "d", "priority": "low", "category": "billing"},
        tenant_id=TRUSTED_TENANT,
        is_approved=False,
    )
    assert result["success"] is False
    assert result["error"] == "approval_required"


def test_handle_create_support_case_approved():
    result = CONTROLLED_TOOL_HANDLERS["create_support_case"](
        {"subject": "s", "description": "d", "priority": "high", "category": "billing"},
        tenant_id=TRUSTED_TENANT,
        is_approved=True,
    )
    assert result["success"] is True
    assert result["case_id"] == "CASE-TEST-001"


def test_handle_fail_tool_always_raises():
    try:
        CONTROLLED_TOOL_HANDLERS["fail_tool"]({"query": "x"})
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_handle_retry_tool_fails_then_succeeds():
    import app.comparison.agent_scenarios as mod
    mod._retry_state.clear()
    # fail_until=2: first call fails (raises), second succeeds
    try:
        CONTROLLED_TOOL_HANDLERS["retry_tool"]({"query": "x", "fail_until": 2})
        raise AssertionError("expected RuntimeError on first call")
    except RuntimeError:
        pass
    r2 = CONTROLLED_TOOL_HANDLERS["retry_tool"]({"query": "x", "fail_until": 2})
    assert r2["result"] == "ok"
    assert r2["attempt"] == 2


def test_handle_echo_tool():
    result = CONTROLLED_TOOL_HANDLERS["echo_tool"]({"message": "hello"})
    assert result["echo"] == "hello" 
