"""Tests for agent comparison metrics (app/comparison/agent_metrics.py).

All metrics are computed from actual execution evidence captured by the
adapters.  Tests use synthetic AgentScenarioResult instances with known
values to verify each metric's numerator/denominator semantics.
"""

from __future__ import annotations

from app.comparison.agent_adapters import AgentScenarioResult
from app.comparison.agent_metrics import (
    ALL_METRICS,
    METRIC_GROUPS,
    average_execution_steps_per_scenario,
    average_provider_calls_per_scenario,
    average_retry_attempts_per_scenario,
    average_tool_calls_per_scenario,
    cancellation_propagation_rate,
    compute_aggregate,
    cross_tenant_access_prevention_rate,
    expected_tool_call_accuracy,
    final_outcome_correctness,
    invalid_argument_rejection_rate,
    loop_prevention_rate,
    malformed_response_handling_rate,
    max_step_enforcement_rate,
    retry_success_rate,
    scenario_success_rate,
    tool_failure_recovery_rate,
    unexpected_tool_call_rate,
    unauthorized_tool_prevention_rate,
)


def _make_result(
    scenario_id: str,
    *,
    status: str = "success",
    outcome_category: str | None = None,
    expected_outcome: str = "success",
    tool_calls_count: int = 0,
    expected_tool_call_count: int | None = None,
    tool_failures: int = 0,
    provider_calls: int = 1,
    provider_retry_attempts: int = 0,
    tool_retry_attempts: int = 0,
    execution_steps: int = 1,
    max_iterations: int = 10,
    expected_tool_names: list[str] | None = None,
    safety_constraints: list[str] | None = None,
    tenant_isolation_verified: bool = False,
    approval_gate_enforced: bool = False,
    invalid_arguments_rejected: bool = False,
    cancellation_exercised: bool = False,
    cancellation_honoured: bool = False,
    final_answer: str | None = "answer",
) -> AgentScenarioResult:
    if outcome_category is None:
        outcome_category = status
    return AgentScenarioResult(
        scenario_id=scenario_id,
        implementation="custom",
        category="cat",
        status=status,
        outcome_category=outcome_category,
        final_answer=final_answer,
        tool_calls_count=tool_calls_count,
        tool_failures=tool_failures,
        provider_calls=provider_calls,
        provider_retry_attempts=provider_retry_attempts,
        tool_retry_attempts=tool_retry_attempts,
        execution_steps=execution_steps,
        max_iterations=max_iterations,
        expected_tool_names=expected_tool_names or [],
        expected_tool_call_count=expected_tool_call_count,
        expected_outcome=expected_outcome,
        expected_final_answer_contains=None,
        safety_constraints=safety_constraints or [],
        tenant_isolation_verified=tenant_isolation_verified,
        approval_gate_enforced=approval_gate_enforced,
        invalid_arguments_rejected=invalid_arguments_rejected,
        cancellation_exercised=cancellation_exercised,
        cancellation_honoured=cancellation_honoured,
    )


def _results_dict(**kwargs) -> dict[str, AgentScenarioResult]:
    out: dict[str, AgentScenarioResult] = {}
    for sid, kw in kwargs.items():
        out[sid] = _make_result(sid, **kw)
    return out


# ---------------------------------------------------------------------------
# outcome_success
# ---------------------------------------------------------------------------

def test_outcome_success_true():
    r = _make_result("s1", status="success")
    from app.comparison.agent_metrics import outcome_success
    assert outcome_success(r) is True


def test_outcome_success_false():
    r = _make_result("s1", status="failed")
    from app.comparison.agent_metrics import outcome_success
    assert outcome_success(r) is False


# ---------------------------------------------------------------------------
# scenario_success_rate
# ---------------------------------------------------------------------------

def test_scenario_success_rate_all_pass():
    results = _results_dict(
        s1=dict(status="success"),
        s2=dict(status="success"),
    )
    assert scenario_success_rate(results) == 1.0


def test_scenario_success_rate_half_pass():
    results = _results_dict(
        s1=dict(status="success"),
        s2=dict(status="failed"),
    )
    assert scenario_success_rate(results) == 0.5


def test_scenario_success_rate_empty():
    assert scenario_success_rate({}) == 0.0


# ---------------------------------------------------------------------------
# expected_tool_call_accuracy
# ---------------------------------------------------------------------------

def test_expected_tool_call_accuracy_all_match():
    results = _results_dict(
        s1=dict(tool_calls_count=2, expected_tool_call_count=2),
        s2=dict(tool_calls_count=3, expected_tool_call_count=3),
    )
    assert expected_tool_call_accuracy(results) == 1.0


def test_expected_tool_call_accuracy_half_match():
    results = _results_dict(
        s1=dict(tool_calls_count=2, expected_tool_call_count=2),
        s2=dict(tool_calls_count=5, expected_tool_call_count=3),
    )
    assert expected_tool_call_accuracy(results) == 0.5


def test_expected_tool_call_accuracy_no_expected():
    results = _results_dict(
        s1=dict(tool_calls_count=2, expected_tool_call_count=None),
    )
    assert expected_tool_call_accuracy(results) == 0.0


# ---------------------------------------------------------------------------
# unexpected_tool_call_rate
# ---------------------------------------------------------------------------

def test_unexpected_tool_call_rate_none():
    results = _results_dict(
        s1=dict(tool_calls_count=1, expected_tool_names=["search_knowledge"]),
    )
    # We need to set tool_calls list too; default is empty so no unexpected
    assert unexpected_tool_call_rate(results) == 0.0


def test_unexpected_tool_call_rate_some():
    results = _results_dict(
        s1=dict(tool_calls_count=1, expected_tool_names=["search_knowledge"]),
    )
    # Manually add an unexpected tool call
    results["s1"].tool_calls.append(
        __import__("app.comparison.agent_adapters", fromlist=["ToolCallRecord"]).ToolCallRecord(
            call_id="c2", name="unexpected_tool", arguments={}, success=True
        )
    )
    assert unexpected_tool_call_rate(results) == 1.0


def test_unexpected_tool_call_rate_empty():
    assert unexpected_tool_call_rate({}) == 0.0


# ---------------------------------------------------------------------------
# final_outcome_correctness
# ---------------------------------------------------------------------------

def test_final_outcome_correctness_all_match():
    results = _results_dict(
        s1=dict(status="success", expected_outcome="success"),
        s2=dict(status="success", expected_outcome="success"),
    )
    assert final_outcome_correctness(results) == 1.0


def test_final_outcome_correctness_mismatch():
    results = _results_dict(
        s1=dict(status="success", expected_outcome="success"),
        s2=dict(status="failed", expected_outcome="success"),
    )
    assert final_outcome_correctness(results) == 0.5


def test_final_outcome_correctness_empty():
    assert final_outcome_correctness({}) == 0.0


# ---------------------------------------------------------------------------
# tool_failure_recovery_rate
# ---------------------------------------------------------------------------

def test_tool_failure_recovery_rate_all_recover():
    results = _results_dict(
        s1=dict(tool_failures=1, status="success"),
        s2=dict(tool_failures=2, status="success"),
    )
    assert tool_failure_recovery_rate(results) == 1.0


def test_tool_failure_recovery_rate_none_recover():
    results = _results_dict(
        s1=dict(tool_failures=1, status="failed"),
    )
    assert tool_failure_recovery_rate(results) == 0.0


def test_tool_failure_recovery_rate_no_failures():
    results = _results_dict(
        s1=dict(tool_failures=0, status="success"),
    )
    assert tool_failure_recovery_rate(results) == 0.0


# ---------------------------------------------------------------------------
# retry_success_rate
# ---------------------------------------------------------------------------

def test_retry_success_rate_all_recover():
    results = _results_dict(
        s1=dict(tool_retry_attempts=1, status="success"),
        s2=dict(provider_retry_attempts=1, status="success"),
    )
    assert retry_success_rate(results) == 1.0


def test_retry_success_rate_none_recover():
    results = _results_dict(
        s1=dict(tool_retry_attempts=1, status="failed"),
    )
    assert retry_success_rate(results) == 0.0


def test_retry_success_rate_no_retries_returns_none():
    results = _results_dict(
        s1=dict(tool_retry_attempts=0, provider_retry_attempts=0),
    )
    assert retry_success_rate(results) is None


def test_retry_success_rate_empty():
    assert retry_success_rate({}) is None


# ---------------------------------------------------------------------------
# malformed_response_handling_rate
# ---------------------------------------------------------------------------

def test_malformed_response_handling_rate_all_safe():
    results = _results_dict(
        s1=dict(
            safety_constraints=["invalid_argument_rejection"],
            status="success",
            invalid_arguments_rejected=True,
        ),
        s2=dict(
            safety_constraints=["invalid_argument_rejection"],
            status="tool_failure",
            invalid_arguments_rejected=True,
        ),
    )
    assert malformed_response_handling_rate(results) == 1.0


def test_malformed_response_handling_rate_not_safe():
    results = _results_dict(
        s1=dict(
            safety_constraints=["invalid_argument_rejection"],
            status="failed",
            invalid_arguments_rejected=True,
        ),
    )
    assert malformed_response_handling_rate(results) == 0.0


def test_malformed_response_handling_rate_empty():
    assert malformed_response_handling_rate({}) == 0.0


# ---------------------------------------------------------------------------
# max_step_enforcement_rate
# ---------------------------------------------------------------------------

def test_max_step_enforcement_rate_all_enforced():
    results = _results_dict(
        s1=dict(
            safety_constraints=["loop_prevention_enforced"],
            status="max_iterations_reached",
        ),
        s2=dict(
            safety_constraints=["loop_prevention_enforced"],
            status="max_iterations_reached",
        ),
    )
    assert max_step_enforcement_rate(results) == 1.0


def test_max_step_enforcement_rate_not_enforced():
    results = _results_dict(
        s1=dict(
            safety_constraints=["loop_prevention_enforced"],
            status="success",
        ),
    )
    assert max_step_enforcement_rate(results) == 0.0


def test_max_step_enforcement_rate_empty():
    assert max_step_enforcement_rate({}) == 0.0


# ---------------------------------------------------------------------------
# loop_prevention_rate
# ---------------------------------------------------------------------------

def test_loop_prevention_rate_all_within_budget():
    results = _results_dict(
        s1=dict(execution_steps=3, max_iterations=10),
        s2=dict(execution_steps=5, max_iterations=10),
    )
    assert loop_prevention_rate(results) == 1.0


def test_loop_prevention_rate_exceeds_budget():
    results = _results_dict(
        s1=dict(execution_steps=15, max_iterations=10),
    )
    assert loop_prevention_rate(results) == 0.0


def test_loop_prevention_rate_empty():
    assert loop_prevention_rate({}) == 0.0


# ---------------------------------------------------------------------------
# cancellation_propagation_rate
# ---------------------------------------------------------------------------

def test_cancellation_propagation_rate_all_honoured():
    results = _results_dict(
        s1=dict(cancellation_exercised=True, cancellation_honoured=True),
        s2=dict(cancellation_exercised=True, cancellation_honoured=True),
    )
    assert cancellation_propagation_rate(results) == 1.0


def test_cancellation_propagation_rate_none_honoured():
    results = _results_dict(
        s1=dict(cancellation_exercised=True, cancellation_honoured=False),
    )
    assert cancellation_propagation_rate(results) == 0.0


def test_cancellation_propagation_rate_no_cancellation_returns_none():
    results = _results_dict(
        s1=dict(cancellation_exercised=False),
    )
    assert cancellation_propagation_rate(results) is None


def test_cancellation_propagation_rate_empty():
    assert cancellation_propagation_rate({}) is None


# ---------------------------------------------------------------------------
# cross_tenant_access_prevention_rate
# ---------------------------------------------------------------------------

def test_cross_tenant_access_prevention_rate_all_verified():
    results = _results_dict(
        s1=dict(
            safety_constraints=["cross_tenant_access_prevention"],
            tenant_isolation_verified=True,
        ),
        s2=dict(
            safety_constraints=["cross_tenant_access_prevention"],
            tenant_isolation_verified=True,
        ),
    )
    assert cross_tenant_access_prevention_rate(results) == 1.0


def test_cross_tenant_access_prevention_rate_not_verified():
    results = _results_dict(
        s1=dict(
            safety_constraints=["cross_tenant_access_prevention"],
            tenant_isolation_verified=False,
        ),
    )
    assert cross_tenant_access_prevention_rate(results) == 0.0


def test_cross_tenant_access_prevention_rate_empty():
    assert cross_tenant_access_prevention_rate({}) == 0.0


# ---------------------------------------------------------------------------
# unauthorized_tool_prevention_rate
# ---------------------------------------------------------------------------

def test_unauthorized_tool_prevention_rate_all_enforced():
    results = _results_dict(
        s1=dict(
            safety_constraints=["approval_gated_action_enforcement"],
            approval_gate_enforced=True,
        ),
        s2=dict(
            safety_constraints=["approval_gated_action_enforcement"],
            approval_gate_enforced=True,
        ),
    )
    assert unauthorized_tool_prevention_rate(results) == 1.0


def test_unauthorized_tool_prevention_rate_not_enforced():
    results = _results_dict(
        s1=dict(
            safety_constraints=["approval_gated_action_enforcement"],
            approval_gate_enforced=False,
        ),
    )
    assert unauthorized_tool_prevention_rate(results) == 0.0


def test_unauthorized_tool_prevention_rate_empty():
    assert unauthorized_tool_prevention_rate({}) == 0.0


# ---------------------------------------------------------------------------
# invalid_argument_rejection_rate
# ---------------------------------------------------------------------------

def test_invalid_argument_rejection_rate_all_rejected():
    results = _results_dict(
        s1=dict(
            safety_constraints=["invalid_argument_rejection"],
            invalid_arguments_rejected=True,
        ),
        s2=dict(
            safety_constraints=["invalid_argument_rejection"],
            invalid_arguments_rejected=True,
        ),
    )
    assert invalid_argument_rejection_rate(results) == 1.0


def test_invalid_argument_rejection_rate_not_rejected():
    results = _results_dict(
        s1=dict(
            safety_constraints=["invalid_argument_rejection"],
            invalid_arguments_rejected=False,
        ),
    )
    assert invalid_argument_rejection_rate(results) == 0.0


def test_invalid_argument_rejection_rate_empty():
    assert invalid_argument_rejection_rate({}) == 0.0


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def test_average_tool_calls_per_scenario():
    results = _results_dict(
        s1=dict(tool_calls_count=2),
        s2=dict(tool_calls_count=4),
        s3=dict(tool_calls_count=6),
    )
    assert average_tool_calls_per_scenario(results) == 4.0


def test_average_provider_calls_per_scenario():
    results = _results_dict(
        s1=dict(provider_calls=2),
        s2=dict(provider_calls=4),
    )
    assert average_provider_calls_per_scenario(results) == 3.0


def test_average_retry_attempts_per_scenario():
    results = _results_dict(
        s1=dict(provider_retry_attempts=1, tool_retry_attempts=2),
        s2=dict(provider_retry_attempts=0, tool_retry_attempts=0),
    )
    assert average_retry_attempts_per_scenario(results) == 1.5


def test_average_execution_steps_per_scenario():
    results = _results_dict(
        s1=dict(execution_steps=3),
        s2=dict(execution_steps=7),
    )
    assert average_execution_steps_per_scenario(results) == 5.0

    # Empty
    assert average_execution_steps_per_scenario({}) == 0.0


# ---------------------------------------------------------------------------
# compute_aggregate
# ---------------------------------------------------------------------------

def test_compute_aggregate_returns_all_metrics():
    results = _results_dict(
        s1=dict(status="success", tool_calls_count=1),
    )
    agg = compute_aggregate(results)
    assert set(agg.keys()) == set(ALL_METRICS.keys())
    assert agg["scenario_success_rate"] == 1.0


def test_metric_groups_cover_all_metrics():
    all_grouped = set()
    for names in METRIC_GROUPS.values():
        all_grouped.update(names)
    assert all_grouped == set(ALL_METRICS.keys())


# ---------------------------------------------------------------------------
# Zero denominators
# ---------------------------------------------------------------------------

def test_all_metrics_zero_denominator_return_zero_or_none():
    results: dict[str, AgentScenarioResult] = {}
    agg = compute_aggregate(results)
    for name, value in agg.items():
        if name in ("retry_success_rate", "cancellation_propagation_rate"):
            assert value is None, f"{name} should be None for empty results"
        else:
            assert value == 0.0, f"{name} should be 0.0 for empty results, got {value}"
