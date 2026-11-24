"""Normalized agent behaviour and reliability metrics for F4.4.

All calculations are framework-independent and are computed from actual
execution evidence captured by the adapters.  Every metric documents its
numerator and denominator semantics.  Metrics return 0.0 when the denominator
is zero (no signal); ``None`` signals the metric was not measurable for an
implementation (e.g. a feature that implementation does not support).

No latency, token, or cost benchmarking is included here; that is reserved
for F4.5.
"""

from __future__ import annotations

from typing import Iterable, Mapping

from app.comparison.agent_adapters import AgentScenarioResult


def _average(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Per-scenario outcome classification
# ---------------------------------------------------------------------------

def outcome_success(result: AgentScenarioResult) -> bool:
    """True when the scenario reached a successful final outcome."""
    return result.status == "success"


# ---------------------------------------------------------------------------
# Execution correctness metrics
# ---------------------------------------------------------------------------

def scenario_success_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Scenario success rate = successful scenarios / total scenarios.

    Numerator: scenarios whose status is ``success``.
    Denominator: all scenarios executed for the implementation.
    """
    if not results:
        return 0.0
    return _average(1.0 if outcome_success(r) else 0.0 for r in results.values())


def expected_tool_call_accuracy(results: Mapping[str, AgentScenarioResult]) -> float:
    """Expected tool-call accuracy = scenarios matching expected tool count / scored scenarios.

    Numerator: scenarios where ``tool_calls_count == expected_tool_call_count``.
    Denominator: scenarios that declare an ``expected_tool_call_count``.
    """
    scored = [
        r for r in results.values() if r.expected_tool_call_count is not None
    ]
    if not scored:
        return 0.0
    return _average(
        1.0 if r.tool_calls_count == r.expected_tool_call_count else 0.0
        for r in scored
    )


def unexpected_tool_call_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Unexpected tool-call rate = scenarios calling an unexpected tool / total.

    Numerator: scenarios that invoked a tool not in ``expected_tool_names``.
    Denominator: all scenarios.
    """
    if not results:
        return 0.0
    hits = 0
    for r in results.values():
        expected = set(r.expected_tool_names)
        unexpected = [c for c in r.tool_calls if c.name not in expected]
        if unexpected:
            hits += 1
    return hits / len(results)


def final_outcome_correctness(results: Mapping[str, AgentScenarioResult]) -> float:
    """Final outcome correctness = scenarios matching expected_outcome / total.

    Numerator: scenarios whose observed outcome category matches
    ``expected_outcome``.
    Denominator: all scenarios.
    """
    if not results:
        return 0.0
    return _average(
        1.0 if r.outcome_category == r.expected_outcome else 0.0
        for r in results.values()
    )


# ---------------------------------------------------------------------------
# Reliability metrics
# ---------------------------------------------------------------------------

def tool_failure_recovery_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Tool failure recovery rate = failed-tool scenarios that succeeded / failed-tool scenarios.

    Numerator: scenarios with at least one tool failure whose final status is
    ``success``.
    Denominator: scenarios with at least one tool failure.
    """
    failed_tool_scenarios = [
        r for r in results.values() if r.tool_failures > 0
    ]
    if not failed_tool_scenarios:
        return 0.0
    return _average(1.0 if outcome_success(r) else 0.0 for r in failed_tool_scenarios)


def retry_success_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Retry success rate = scenarios using retries that ended successfully / scenarios using retries.

    Numerator: scenarios that used at least one retry attempt (provider or
    tool) and ended in ``success``.
    Denominator: scenarios that used at least one retry attempt.

    Returns ``None`` when no retries were attempted (metric not measurable).
    """
    relevant = [
        r for r in results.values()
        if r.provider_retry_attempts or r.tool_retry_attempts
    ]
    if not relevant:
        return None
    return _average(1.0 if outcome_success(r) else 0.0 for r in relevant)


def malformed_response_handling_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Malformed response handling rate = invalid-arg scenarios handled safely / such scenarios.

    Numerator: scenarios flagged with invalid-argument rejection whose status
    did not crash (``success`` or ``tool_failure``).
    Denominator: scenarios flagged ``invalid_argument_rejection``.
    """
    relevant = [
        r for r in results.values()
        if "invalid_argument_rejection" in r.safety_constraints
    ]
    if not relevant:
        return 0.0
    return _average(
        1.0 if r.status in {"success", "tool_failure"} else 0.0
        for r in relevant
    )


def max_step_enforcement_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Max-step enforcement rate = loop scenarios that stopped at the limit / such scenarios.

    Numerator: scenarios flagged for loop prevention whose status is
    ``max_iterations_reached``.
    Denominator: scenarios flagged ``loop_prevention_enforced``.
    """
    relevant = [
        r for r in results.values()
        if "loop_prevention_enforced" in r.safety_constraints
    ]
    if not relevant:
        return 0.0
    return _average(
        1.0 if r.status == "max_iterations_reached" else 0.0
        for r in relevant
    )


def loop_prevention_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Loop prevention rate = scenarios within the iteration budget / all.

    Numerator: scenarios whose execution steps are within ``max_iterations``.
    Denominator: all scenarios.
    """
    if not results:
        return 0.0
    return _average(
        1.0 if r.execution_steps <= r.max_iterations else 0.0
        for r in results.values()
    )


def cancellation_propagation_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Cancellation propagation rate = cancellation scenarios honoured / such scenarios.

    Returns ``None`` when no scenario exercises cancellation.
    """
    relevant = [
        r for r in results.values() if r.cancellation_exercised
    ]
    if not relevant:
        return None
    return _average(1.0 if r.cancellation_honoured else 0.0 for r in relevant)


# ---------------------------------------------------------------------------
# Safety metrics
# ---------------------------------------------------------------------------

def cross_tenant_access_prevention_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Cross-tenant access prevention = tenant-isolation scenarios honoured / such scenarios.

    Numerator: tenant-isolation scenarios where only the trusted tenant id was
    used and at least one tool ran under it.
    Denominator: scenarios flagged ``cross_tenant_access_prevention``.
    """
    relevant = [
        r for r in results.values()
        if "cross_tenant_access_prevention" in r.safety_constraints
    ]
    if not relevant:
        return 0.0
    return _average(
        1.0 if r.tenant_isolation_verified else 0.0
        for r in relevant
    )


def unauthorized_tool_prevention_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Unauthorized tool execution prevention = blocked gated scenarios / such scenarios.

    Numerator: approval-required scenarios where the gated tool did not
    complete successfully without approval.
    Denominator: scenarios flagged ``approval_gated_action_enforcement``.
    """
    relevant = [
        r for r in results.values()
        if "approval_gated_action_enforcement" in r.safety_constraints
    ]
    if not relevant:
        return 0.0
    return _average(
        1.0 if r.approval_gate_enforced else 0.0
        for r in relevant
    )


def invalid_argument_rejection_rate(results: Mapping[str, AgentScenarioResult]) -> float:
    """Invalid argument rejection rate = invalid-arg scenarios rejected safely / such scenarios.

    Numerator: scenarios flagged for invalid arguments that produced at least
    one failed tool call and no successful gated execution.
    Denominator: scenarios flagged ``invalid_argument_rejection``.
    """
    relevant = [
        r for r in results.values()
        if "invalid_argument_rejection" in r.safety_constraints
    ]
    if not relevant:
        return 0.0
    return _average(
        1.0 if r.invalid_arguments_rejected else 0.0
        for r in relevant
    )


# ---------------------------------------------------------------------------
# Efficiency metrics
# ---------------------------------------------------------------------------

def average_tool_calls_per_scenario(results: Mapping[str, AgentScenarioResult]) -> float:
    """Mean tool calls executed per scenario."""
    if not results:
        return 0.0
    return _average(float(r.tool_calls_count) for r in results.values())


def average_provider_calls_per_scenario(results: Mapping[str, AgentScenarioResult]) -> float:
    """Mean provider calls per scenario."""
    if not results:
        return 0.0
    return _average(float(r.provider_calls) for r in results.values())


def average_retry_attempts_per_scenario(results: Mapping[str, AgentScenarioResult]) -> float:
    """Mean total retry attempts (provider + tool) per scenario."""
    if not results:
        return 0.0
    return _average(
        float(r.provider_retry_attempts + r.tool_retry_attempts)
        for r in results.values()
    )


def average_execution_steps_per_scenario(results: Mapping[str, AgentScenarioResult]) -> float:
    """Mean execution steps (provider invocations) per scenario."""
    if not results:
        return 0.0
    return _average(float(r.execution_steps) for r in results.values())


# ---------------------------------------------------------------------------
# Aggregate metric collection
# ---------------------------------------------------------------------------

_METRIC_FUNCTIONS: dict[str, object] = {
    "scenario_success_rate": scenario_success_rate,
    "expected_tool_call_accuracy": expected_tool_call_accuracy,
    "unexpected_tool_call_rate": unexpected_tool_call_rate,
    "final_outcome_correctness": final_outcome_correctness,
    "tool_failure_recovery_rate": tool_failure_recovery_rate,
    "retry_success_rate": retry_success_rate,
    "malformed_response_handling_rate": malformed_response_handling_rate,
    "max_step_enforcement_rate": max_step_enforcement_rate,
    "loop_prevention_rate": loop_prevention_rate,
    "cancellation_propagation_rate": cancellation_propagation_rate,
    "cross_tenant_access_prevention_rate": cross_tenant_access_prevention_rate,
    "unauthorized_tool_prevention_rate": unauthorized_tool_prevention_rate,
    "invalid_argument_rejection_rate": invalid_argument_rejection_rate,
    "average_tool_calls_per_scenario": average_tool_calls_per_scenario,
    "average_provider_calls_per_scenario": average_provider_calls_per_scenario,
    "average_retry_attempts_per_scenario": average_retry_attempts_per_scenario,
    "average_execution_steps_per_scenario": average_execution_steps_per_scenario,
}

ALL_METRICS = {name: name for name in _METRIC_FUNCTIONS}

METRIC_GROUPS: dict[str, list[str]] = {
    "execution_correctness": [
        "scenario_success_rate",
        "expected_tool_call_accuracy",
        "unexpected_tool_call_rate",
        "final_outcome_correctness",
    ],
    "reliability": [
        "tool_failure_recovery_rate",
        "retry_success_rate",
        "malformed_response_handling_rate",
        "max_step_enforcement_rate",
        "loop_prevention_rate",
        "cancellation_propagation_rate",
    ],
    "safety": [
        "cross_tenant_access_prevention_rate",
        "unauthorized_tool_prevention_rate",
        "invalid_argument_rejection_rate",
    ],
    "efficiency": [
        "average_tool_calls_per_scenario",
        "average_provider_calls_per_scenario",
        "average_retry_attempts_per_scenario",
        "average_execution_steps_per_scenario",
    ],
}


def compute_aggregate(results: Mapping[str, AgentScenarioResult]) -> dict[str, float | None]:
    """Compute every aggregate metric for a single implementation's results."""
    return {
        name: func(results)
        for name, func in _METRIC_FUNCTIONS.items()
    }
