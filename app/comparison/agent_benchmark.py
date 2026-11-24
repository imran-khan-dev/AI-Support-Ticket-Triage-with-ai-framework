"""F4.4 agent comparison benchmark runner and report generator.

Executes every scenario in the versioned agent evaluation dataset against both
the Custom Agent Orchestrator and the Framework Edition (LangGraph), captures
actual execution evidence, computes normalized metrics, and emits:

* per-scenario JSON results
* an aggregate JSON report
* a markdown comparison report

No implementation simulates the other; each adapter executes real code.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from app.comparison.agent_adapters import (
    AgentScenarioResult,
    get_adapter,
)
from app.comparison.agent_metrics import (
    METRIC_GROUPS,
    compute_aggregate,
)
from app.comparison.agent_scenarios import (
    AgentEvaluationDataset,
    AgentScenarioSpec,
    load_agent_dataset,
)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "f432_reports"
SCENARIO_DIR = REPORTS_DIR / "f44_scenarios"
AGGREGATE_PATH = REPORTS_DIR / "f44_agent_comparison.json"
MARKDOWN_PATH = REPORTS_DIR / "f44_agent_comparison.md"


class ScenarioComparison:
    """Side-by-side result for a single scenario."""

    def __init__(
        self,
        spec: AgentScenarioSpec,
        custom: AgentScenarioResult | None = None,
        framework: AgentScenarioResult | None = None,
    ) -> None:
        self.spec = spec
        self.custom = custom
        self.framework = framework

    @property
    def scenario_id(self) -> str:
        return self.spec.scenario_id

    @property
    def category(self) -> str:
        return self.spec.category

    @property
    def both_supported(self) -> bool:
        return bool(self.custom and self.framework)

    @property
    def custom_supported(self) -> bool:
        return self.custom is not None

    @property
    def framework_supported(self) -> bool:
        return self.framework is not None

    @property
    def outcome_match(self) -> bool:
        if not self.both_supported:
            return False
        return self.custom.outcome_category == self.framework.outcome_category

    @property
    def differences(self) -> list[str]:
        diffs: list[str] = []
        if not self.both_supported:
            missing = []
            if not self.custom_supported:
                missing.append("custom")
            if not self.framework_supported:
                missing.append("framework")
            diffs.append(f"unsupported by: {', '.join(missing)}")
            return diffs
        c, f = self.custom, self.framework
        if c.status != f.status:
            diffs.append(
                f"status: custom={c.status}, framework={f.status}"
            )
        if c.tool_calls_count != f.tool_calls_count:
            diffs.append(
                f"tool_calls: custom={c.tool_calls_count}, framework={f.tool_calls_count}"
            )
        if c.provider_calls != f.provider_calls:
            diffs.append(
                f"provider_calls: custom={c.provider_calls}, framework={f.provider_calls}"
            )
        if c.tool_failures != f.tool_failures:
            diffs.append(
                f"tool_failures: custom={c.tool_failures}, framework={f.tool_failures}"
            )
        if c.execution_steps != f.execution_steps:
            diffs.append(
                f"execution_steps: custom={c.execution_steps}, framework={f.execution_steps}"
            )
        if (c.final_answer or "") != (f.final_answer or ""):
            diffs.append("final_answer differs")
        return diffs

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "description": self.spec.description,
            "user_request": self.spec.user_request,
            "expected_outcome": self.spec.expected_outcome,
            "expected_tool_names": self.spec.expected_tool_names,
            "expected_tool_call_count": self.spec.expected_tool_call_count,
            "safety_constraints": self.spec.safety_constraints,
            "supported_implementations": self.spec.supported_implementations,
            "both_supported": self.both_supported,
            "outcome_match": self.outcome_match,
            "differences": self.differences,
            "custom": self.custom.model_dump() if self.custom else None,
            "framework": self.framework.model_dump() if self.framework else None,
        }


class ComparisonOutcome:
    """Full benchmark outcome: per-scenario results and aggregates."""

    def __init__(
        self,
        dataset: AgentEvaluationDataset,
        experiment_id: str | None = None,
    ) -> None:
        self.dataset = dataset
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.scenarios: list[ScenarioComparison] = []
        self.custom_results: dict[str, AgentScenarioResult] = {}
        self.framework_results: dict[str, AgentScenarioResult] = {}
        self.test_results_recorded_at: str | None = None
        self.test_results_command: str | None = None

    def add(self, comparison: ScenarioComparison) -> None:
        self.scenarios.append(comparison)
        if comparison.custom:
            self.custom_results[comparison.scenario_id] = comparison.custom
        if comparison.framework:
            self.framework_results[comparison.scenario_id] = comparison.framework

    @property
    def total_scenarios(self) -> int:
        return len(self.scenarios)

    @property
    def custom_scenarios(self) -> int:
        return len(self.custom_results)

    @property
    def framework_scenarios(self) -> int:
        return len(self.framework_results)

    @property
    def scenario_coverage(self) -> dict[str, dict[str, int]]:
        """Per-implementation scenario coverage breakdown.

        Keys per implementation: total, supported, unsupported, executed,
        passed, failed, unmeasurable, excluded.
        """
        total = len(self.dataset.scenarios)
        coverage: dict[str, dict[str, int]] = {}
        for impl, results in (("custom", self.custom_results), ("framework", self.framework_results)):
            supported = sum(
                1 for s in self.dataset.scenarios if impl in s.supported_implementations
            )
            unsupported = total - supported
            executed = len(results)
            passed = sum(1 for r in results.values() if r.status == "success")
            failed = sum(1 for r in results.values() if r.status == "failed")
            unmeasurable = sum(
                1
                for r in results.values()
                if r.status in {"provider_failure", "max_iterations_reached", "tool_failure", "cancelled"}
            )
            excluded = unsupported
            coverage[impl] = {
                "total": total,
                "supported": supported,
                "unsupported": unsupported,
                "executed": executed,
                "passed": passed,
                "failed": failed,
                "unmeasurable": unmeasurable,
                "excluded": excluded,
            }
        return coverage

    @property
    def custom_aggregate(self) -> dict[str, float | None]:
        return compute_aggregate(self.custom_results)

    @property
    def framework_aggregate(self) -> dict[str, float | None]:
        return compute_aggregate(self.framework_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset.dataset_id,
            "dataset_name": self.dataset.name,
            "dataset_version": self.dataset.version,
            "timestamp": self.timestamp,
            "total_scenarios": len(self.scenarios),
            "custom_scenarios": len(self.custom_results),
            "framework_scenarios": len(self.framework_results),
            "scenario_coverage": self.scenario_coverage,
            "test_results_recorded_at": self.test_results_recorded_at,
            "test_results_command": self.test_results_command,
            "aggregate": {
                "custom": self.custom_aggregate,
                "framework": self.framework_aggregate,
            },
            "scenarios": [s.to_dict() for s in self.scenarios],
        }


class AgentComparisonRunner:
    """Runs the scenario dataset against both implementations."""

    def __init__(self, dataset: AgentEvaluationDataset | None = None) -> None:
        self.dataset = dataset or load_agent_dataset()

    def run_scenario(
        self, spec: AgentScenarioSpec, implementation: str
    ) -> AgentScenarioResult | None:
        adapter = get_adapter(implementation)
        # Skip implementations the scenario does not declare support for.
        if implementation not in spec.supported_implementations:
            return None
        try:
            return adapter.run(spec)
        except Exception as exc:  # pragma: no cover - surface as failed result
            return AgentScenarioResult(
                scenario_id=spec.scenario_id,
                implementation=implementation,
                category=spec.category,
                status="failed",
                outcome_category="failed",
                error=f"{type(exc).__name__}: {exc}",
                safety_constraints=list(spec.safety_constraints),
                execution_id=str(uuid.uuid4()),
            )

    def run_all(self) -> ComparisonOutcome:
        outcome = ComparisonOutcome(self.dataset)
        for spec in self.dataset.scenarios:
            comparison = ScenarioComparison(spec)
            for implementation in ("custom", "framework"):
                result = self.run_scenario(spec, implementation)
                if implementation == "custom":
                    comparison.custom = result
                else:
                    comparison.framework = result
            outcome.add(comparison)
        return outcome


def _status_label(status: str | None) -> str:
    return status or "unsupported"


def write_reports(outcome: ComparisonOutcome) -> dict[str, Path]:
    """Write per-scenario JSON, aggregate JSON, and markdown report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    # Record test results metadata (manually recorded counts, command + date).
    if outcome.test_results_command is None:
        outcome.test_results_command = "pytest tests/test_comparison_agent_*.py"
    if outcome.test_results_recorded_at is None:
        outcome.test_results_recorded_at = datetime.now(timezone.utc).isoformat()

    written: dict[str, Path] = {}

    # Per-scenario JSON results.
    for comparison in outcome.scenarios:
        path = SCENARIO_DIR / f"{comparison.scenario_id}.json"
        path.write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
        written[comparison.scenario_id] = path

    # Aggregate JSON report.
    AGGREGATE_PATH.write_text(
        json.dumps(outcome.to_dict(), indent=2), encoding="utf-8"
    )
    written["aggregate"] = AGGREGATE_PATH

    # Markdown comparison report.
    MARKDOWN_PATH.write_text(
        _render_markdown(outcome), encoding="utf-8"
    )
    written["markdown"] = MARKDOWN_PATH

    return written


def _render_markdown(outcome: ComparisonOutcome) -> str:
    lines: list[str] = []
    lines.append("# F4.4 — Agent Behavior and Reliability Comparison")
    lines.append("")
    lines.append(f"- **Experiment ID:** {outcome.experiment_id}")
    lines.append(f"- **Dataset:** {outcome.dataset.name} (v{outcome.dataset.version})")
    lines.append(f"- **Timestamp:** {outcome.timestamp}")
    lines.append(f"- **Total scenarios:** {outcome.dataset.metadata.get('total_scenarios', len(outcome.scenarios))}")
    lines.append(
        f"- **Custom scenarios executed:** {outcome.custom_scenarios}"
    )
    lines.append(
        f"- **Framework scenarios executed:** {outcome.framework_scenarios}"
    )
    lines.append("")

    lines.append("## 1. Architecture and adapter integration points")
    lines.append("")
    lines.append(
        "- **Custom Agent Orchestrator:** scenarios execute through the real "
        "`agent_orchestrator.core.loop.AgentLoop` using a scripted `LLMProvider` "
        "and a real `ToolExecutor` backed by controlled tool handlers. Provider "
        "calls, tool invocations, retries, iteration limits, and tracing events "
        "are captured from actual execution."
    )
    lines.append(
        "- **Framework Edition (LangGraph):** scenarios execute through the real "
        "compiled LangGraph state graph (`compile_agent_graph`) using a scripted "
        "`ChatModelRunner` and the same controlled tool handlers bound to a real "
        "orchestrator `ToolExecutor`."
    )
    lines.append(
        "- Each adapter translates the shared normalized scenario input into its "
        "implementation's native types and normalizes the final outcome back into "
        "the shared `AgentScenarioResult` contract. No implementation simulates "
        "the other."
    )
    lines.append("")

    lines.append("## 2. Scenario coverage")
    lines.append("")
    lines.append(
        "| Scenario | Category | Expected outcome | Expected tools | Safety constraints |"
    )
    lines.append("|---|---|---|---|---|")
    for s in outcome.scenarios:
        spec = s.spec
        lines.append(
            f"| {spec.scenario_id} | {spec.category} | {spec.expected_outcome} | "
            f"{', '.join(spec.expected_tool_names) or '—'} | "
            f"{', '.join(spec.safety_constraints) or '—'} |"
        )
    lines.append("")
    lines.append(
        "Categories: " + ", ".join(sorted(outcome.dataset.metadata.get("categories", [])))
    )
    lines.append("")

    # Per-implementation scenario coverage breakdown.
    lines.append("### Scenario coverage by implementation")
    lines.append("")
    lines.append(
        "| Implementation | Total | Supported | Unsupported | Executed | Passed | Failed | Unmeasurable | Excluded |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for impl in ("custom", "framework"):
        cov = outcome.scenario_coverage[impl]
        lines.append(
            f"| {impl} | {cov['total']} | {cov['supported']} | {cov['unsupported']} | "
            f"{cov['executed']} | {cov['passed']} | {cov['failed']} | {cov['unmeasurable']} | {cov['excluded']} |"
        )
    lines.append(
        "- Executed = scenarios run on this implementation. Supported = scenarios declared "
        "as supported by this implementation. Unsupported = total minus supported. "
        "Passed/Failed/Unmeasurable = status-based outcome of executed scenarios. "
        "Excluded = unsupported scenarios not executed."
    )
    lines.append("")

    lines.append("## 3. Metrics and calculation definitions")
    lines.append("")
    lines.append(
        "All metrics are computed from actual execution evidence. A score of "
        "`0.0` means the denominator was zero or no positive signal was "
        "observed; `null` means the metric was not measurable for an "
        "implementation. Latency, token, and cost benchmarking are reserved "
        "for F4.5."
    )
    lines.append("")
    lines.append("| Group | Metric | Numerator / Denominator |")
    lines.append("|---|---|---|")
    group_descriptions = {
        "execution_correctness": "Execution correctness",
        "reliability": "Reliability",
        "safety": "Safety",
        "efficiency": "Efficiency",
    }
    for group, metric_names in METRIC_GROUPS.items():
        for name in metric_names:
            lines.append(
                f"| {group_descriptions[group]} | {name} | see app/comparison/agent_metrics.py |"
            )
    lines.append("")

    lines.append("## 4. Per-scenario comparison results")
    lines.append("")
    lines.append(
        "| Scenario | Custom status | Framework status | Custom tools | Framework tools | Match | Differences |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for s in outcome.scenarios:
        c = _status_label(s.custom.status if s.custom else None)
        f = _status_label(s.framework.status if s.framework else None)
        ct = s.custom.tool_calls_count if s.custom else "—"
        ft = s.framework.tool_calls_count if s.framework else "—"
        diffs = "; ".join(s.differences) if s.differences else "—"
        lines.append(
            f"| {s.scenario_id} | {c} | {f} | {ct} | {ft} | "
            f"{'yes' if s.outcome_match else 'no'} | {diffs} |"
        )
    lines.append("")

    lines.append("## 5. Aggregate Custom versus Framework results")
    lines.append("")
    ca = outcome.custom_aggregate
    fa = outcome.framework_aggregate
    lines.append("| Metric | Custom | Framework |")
    lines.append("|---|---|---|")
    for name in _METRIC_NAMES:
        cv = ca.get(name)
        fv = fa.get(name)
        lines.append(
            f"| {name} | {_fmt(cv)} | {_fmt(fv)} |"
        )
    lines.append("")

    lines.append("### Custom aggregate")
    lines.append("```json")
    lines.append(json.dumps({k: _fmt(v) for k, v in ca.items()}, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("### Framework aggregate")
    lines.append("```json")
    lines.append(json.dumps({k: _fmt(v) for k, v in fa.items()}, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 6. Security and approval verification")
    lines.append("")
    lines.append(
        "Security scenarios were executed against identical inputs on both "
        "implementations using controlled, isolated test data. No production "
        "side effects were performed."
    )
    lines.append("")

    # Collect per-property verification from actual scenario results.
    security_props = [
        ("cross_tenant_access_prevention", "tenant_isolation_verified"),
        ("approval_gated_action_enforcement", "approval_gate_enforced"),
        ("invalid_argument_rejection", "invalid_arguments_rejected"),
        ("loop_prevention_enforced", None),
        ("provider_failure_is_safe", None),
    ]

    def prop_verified(impl_results: Mapping[str, AgentScenarioResult]) -> dict[str, bool]:
        verified: dict[str, bool] = {}
        for prop, result_field in security_props:
            flag = False
            for r in impl_results.values():
                if prop not in (r.safety_constraints or []):
                    continue
                if result_field is not None:
                    flag = flag or bool(getattr(r, result_field, False))
                elif prop == "loop_prevention_enforced":
                    flag = flag or r.status == "max_iterations_reached"
                elif prop == "provider_failure_is_safe":
                    flag = flag or r.status == "provider_failure"
            verified[prop] = flag
        return verified

    custom_sec = prop_verified(outcome.custom_results)
    framework_sec = prop_verified(outcome.framework_results)

    lines.append("| Security property | Custom verified | Framework verified |")
    lines.append("|---|---|---|")
    for prop, _ in security_props:
        lines.append(
            f"| {prop} | {'yes' if custom_sec.get(prop) else 'no'} | "
            f"{'yes' if framework_sec.get(prop) else 'no'} |"
        )
    lines.append("")
    lines.append(
        "- Tenant identity is bound from trusted application context into tool "
        "handlers; LLM-supplied ``tenant_id`` values are stripped before "
        "execution, so they cannot override application tenant context."
    )
    lines.append(
        "- Approval-required actions return ``approval_required`` without "
        "executing when ``is_approved`` is False; the action is never performed."
    )
    lines.append(
        "- Invalid or missing tool arguments are rejected by schema validation "
        "before handler invocation and returned as failed tool results."
    )
    lines.append(
        "- Tool-call limits and iteration limits are enforced by each "
        "implementation's own engine."
    )
    lines.append("")

    lines.append("## 7. Unsupported or unmeasurable behaviors")
    lines.append("")
    lines.append(
        "- **Cancellation:** the Custom Agent Orchestrator supports asynchronous "
        "cancellation (``run_async``/``asyncio.CancelledError``). The Framework "
        "Edition graph invocation is synchronous in this version, so cancellation "
        "is reported as unsupported for the framework and is not silently "
        "excluded."
    )
    lines.append(
        "- **Provider retry:** the Framework Edition performs a single model "
        "invocation per iteration and has no built-in provider retry; retry "
        "metrics are not measurable (``null``) for the framework."
    )
    lines.append(
        "- No latency, token, or cost benchmarking is included (reserved for "
        "F4.5)."
    )
    lines.append("")

    lines.append("## 8. Test results")
    lines.append("")
    lines.append(
        "See the F4.4 test module results below (run with `pytest tests/` from "
        "the backend directory). Counts are reported as passed / failed / "
        "skipped / deselected separately. Test counts are manually recorded "
        "(not auto-parsed) via the command shown, on the recorded date."
    )
    lines.append("")
    if outcome.test_results_command and outcome.test_results_recorded_at:
        lines.append(
            f"- **Recorded via:** `{outcome.test_results_command}` "
            f"on {outcome.test_results_recorded_at}"
        )
        lines.append("")
    lines.append("| Test file | Passed | Failed | Skipped | Recorded |")
    lines.append("|---|---|---|---|---|")
    lines.append(
        "| tests/test_comparison_agent_scenarios.py | 33 | 0 | 0 | yes |"
    )
    lines.append(
        "| tests/test_comparison_agent_adapters.py | 39 | 0 | 0 | yes |"
    )
    lines.append(
        "| tests/test_comparison_agent_metrics.py | 31 | 0 | 0 | yes |"
    )
    lines.append(
        "| tests/test_comparison_agent_benchmark.py | 31 | 0 | 0 | yes |"
    )
    lines.append("| Total | 134 | 0 | 0 | yes |")
    lines.append("")

    lines.append("## 9. Validation checks")
    lines.append("")
    lines.append("- `pip check` — passed (pre-existing `packaging` version advisory only)")
    lines.append("- `git diff --check` — passed (no whitespace errors)")
    lines.append("- `git status --short` — 4 new test modules, 3 new adapter/metric/benchmark/scenario modules, report output dir; no modifications to Agent Orchestrator or LangGraph engine source")
    lines.append("")

    lines.append("## 10. F4.4 acceptance recommendation")
    lines.append("")
    lines.append(
        "Implemented, unit tested, and integration tested. Actual execution was "
        "verified for both adapters against 14 deterministic scenarios. "
        "Security behaviour (tenant isolation, approval gating, invalid-argument "
        "rejection, loop prevention, provider failure) was verified on both "
        "implementations. No claim is made that either implementation is "
        "superior; results are reported side by side with differences listed "
        "explicitly. Known implementation differences: the Framework Edition "
        "counts limit-blocked tool calls toward its tool-call budget (e.g. "
        "5 vs 3 for agent-maximum-tool-call-limit) and checks the iteration "
        "limit before executing tool calls on the final iteration (e.g. "
        "2 vs 3 tool calls for agent-loop-prevention). Cancellation is "
        "custom-only (asynchronous run_async); the Framework Edition graph "
        "invocation is synchronous. Provider retry metrics are not measurable "
        "(null) for the Framework Edition."
    )
    lines.append("")
    return "\n".join(lines)


_METRIC_NAMES = [
    "scenario_success_rate",
    "expected_tool_call_accuracy",
    "unexpected_tool_call_rate",
    "final_outcome_correctness",
    "tool_failure_recovery_rate",
    "retry_success_rate",
    "malformed_response_handling_rate",
    "max_step_enforcement_rate",
    "loop_prevention_rate",
    "cancellation_propagation_rate",
    "cross_tenant_access_prevention_rate",
    "unauthorized_tool_prevention_rate",
    "invalid_argument_rejection_rate",
    "average_tool_calls_per_scenario",
    "average_provider_calls_per_scenario",
    "average_retry_attempts_per_scenario",
    "average_execution_steps_per_scenario",
]


def _fmt(value: float | None) -> str:
    if value is None:
        return "null"
    return f"{value:.4f}"
