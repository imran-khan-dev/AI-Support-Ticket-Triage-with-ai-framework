"""F4.5 agent performance, token, and cost benchmark.

Runs both the Custom Agent Orchestrator and Framework Edition (LangGraph)
implementations against the same deterministic scenario inputs, measuring:

* End-to-end execution latency
* Provider-call latency
* Tool execution latency
* Token consumption (actual when available, unavailable otherwise)
* Estimated provider cost
* Execution overhead

Deterministic trials use scripted providers and controlled tool handlers.
Live Groq trials are opt-in only and never mixed with deterministic results.
No implementation simulates the other; each adapter executes real code.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.comparison.agent_adapters import (
    AgentScenarioResult,
    CustomAgentAdapter,
    LangGraphAgentAdapter,
    get_adapter,
)
from app.comparison.agent_performance_metrics import (
    PerformanceTrialResult,
    compute_performance_aggregate,
)
from app.comparison.agent_scenarios import (
    AgentEvaluationDataset,
    AgentScenarioSpec,
    load_agent_dataset,
)
from app.comparison.pricing_config import PricingConfig, get_model_price, list_known_models

REPORTS_DIR = Path(__file__).resolve().parent.parent / "f432_reports"
F45_AGGREGATE_PATH = REPORTS_DIR / "f45_performance_comparison.json"
F45_MARKDOWN_PATH = REPORTS_DIR / "f45_performance_comparison.md"
F45_SCENARIO_DIR = REPORTS_DIR / "f45_scenarios"

LIVE_ENABLED_ENV = "F45_LIVE_ENABLED"

#: Default warm-up trials before measured repetitions.
DEFAULT_WARM_UP = 2
#: Default measured repetitions per scenario/implementation.
DEFAULT_REPETITIONS = 5


@dataclass
class PerformanceTrialResult:
    """A single measured trial of one scenario on one implementation."""

    scenario_id: str
    implementation: str
    trial_number: int
    execution_mode: str  # "deterministic" or "live"
    total_latency_ms: float
    provider_latency_ms: float
    tool_latency_ms: float
    provider_calls: int
    tool_calls_count: int
    tool_failures: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None
    model: str | None
    provider: str | None
    pricing_source: str | None
    pricing_date: str | None
    status: str
    error: str | None = None
    execution_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "implementation": self.implementation,
            "trial_number": self.trial_number,
            "execution_mode": self.execution_mode,
            "total_latency_ms": self.total_latency_ms,
            "provider_latency_ms": self.provider_latency_ms,
            "tool_latency_ms": self.tool_latency_ms,
            "provider_calls": self.provider_calls,
            "tool_calls_count": self.tool_calls_count,
            "tool_failures": self.tool_failures,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "model": self.model,
            "provider": self.provider,
            "pricing_source": self.pricing_source,
            "pricing_date": self.pricing_date,
            "status": self.status,
            "error": self.error,
            "execution_steps": self.execution_steps,
        }


@dataclass
class PerformanceBenchmarkConfig:
    """Configuration for the performance benchmark."""

    warm_up: int = DEFAULT_WARM_UP
    repetitions: int = DEFAULT_REPETITIONS
    live_enabled: bool = False
    dataset: AgentEvaluationDataset | None = None
    pricing: PricingConfig | None = None


@dataclass
class PerformanceComparison:
    """Side-by-side performance comparison for a single scenario.

    Paired-comparison semantics: only scenarios supported by both
    implementations are eligible for latency comparison. Unsupported
    scenarios are excluded and reported separately.
    """

    scenario_id: str
    category: str
    supported_implementations: list[str] = field(default_factory=list)
    custom_trials: list[PerformanceTrialResult] = field(default_factory=list)
    framework_trials: list[PerformanceTrialResult] = field(default_factory=list)

    @property
    def custom_successful(self) -> list[PerformanceTrialResult]:
        return [t for t in self.custom_trials if t.status == "success"]

    @property
    def framework_successful(self) -> list[PerformanceTrialResult]:
        return [t for t in self.framework_trials if t.status == "success"]

    @property
    def custom_success_count(self) -> int:
        return len(self.custom_successful)

    @property
    def framework_success_count(self) -> int:
        return len(self.framework_successful)

    @property
    def shared(self) -> bool:
        """True when both implementations executed this scenario."""
        return "custom" in self.supported_implementations and "framework" in self.supported_implementations

    @property
    def successful_paired_trials(self) -> int:
        """Count of trial indices where both implementations succeeded.

        Paired by trial number; only meaningful when both implementations
        ran the same number of trials (shared scenario).
        """
        if not self.shared:
            return 0
        n = min(len(self.custom_trials), len(self.framework_trials))
        return sum(
            1
            for i in range(n)
            if self.custom_trials[i].status == "success"
            and self.framework_trials[i].status == "success"
        )

    @property
    def failed_paired_trials(self) -> int:
        """Count of trial indices where at least one implementation failed."""
        if not self.shared:
            return 0
        n = min(len(self.custom_trials), len(self.framework_trials))
        return sum(
            1
            for i in range(n)
            if self.custom_trials[i].status != "success"
            or self.framework_trials[i].status != "success"
        )

    def _stats(self, trials: list[PerformanceTrialResult], attr: str) -> dict[str, float | None]:
        values = [getattr(t, attr) for t in trials if getattr(t, attr) is not None]
        if not values:
            return {"median": None, "mean": None, "min": None, "max": None}
        return {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }

    def custom_stats(self, attr: str) -> dict[str, float | None]:
        return self._stats(self.custom_trials, attr)

    def framework_stats(self, attr: str) -> dict[str, float | None]:
        return self._stats(self.framework_trials, attr)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "category": self.category,
            "supported_implementations": self.supported_implementations,
            "shared": self.shared,
            "custom_trials": [t.to_dict() for t in self.custom_trials],
            "framework_trials": [t.to_dict() for t in self.framework_trials],
            "custom_success_count": self.custom_success_count,
            "framework_success_count": self.framework_success_count,
            "successful_paired_trials": self.successful_paired_trials,
            "failed_paired_trials": self.failed_paired_trials,
            "custom_stats": {
                "total_latency_ms": self.custom_stats("total_latency_ms"),
                "provider_latency_ms": self.custom_stats("provider_latency_ms"),
                "tool_latency_ms": self.custom_stats("tool_latency_ms"),
                "input_tokens": self.custom_stats("input_tokens"),
                "output_tokens": self.custom_stats("output_tokens"),
                "total_tokens": self.custom_stats("total_tokens"),
                "estimated_cost_usd": self.custom_stats("estimated_cost_usd"),
            },
            "framework_stats": {
                "total_latency_ms": self.framework_stats("total_latency_ms"),
                "provider_latency_ms": self.framework_stats("provider_latency_ms"),
                "tool_latency_ms": self.framework_stats("tool_latency_ms"),
                "input_tokens": self.framework_stats("input_tokens"),
                "output_tokens": self.framework_stats("output_tokens"),
                "total_tokens": self.framework_stats("total_tokens"),
                "estimated_cost_usd": self.framework_stats("estimated_cost_usd"),
            },
            "aggregate": compute_performance_aggregate(
                self.custom_trials, self.framework_trials
            ),
        }


@dataclass
class PerformanceBenchmarkOutcome:
    """Full benchmark outcome: per-scenario comparisons and aggregates."""

    config: PerformanceBenchmarkConfig
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    comparisons: list[PerformanceComparison] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)

    @property
    def total_scenarios(self) -> int:
        return len(self.comparisons)

    @property
    def custom_success_total(self) -> int:
        return sum(c.custom_success_count for c in self.comparisons)

    @property
    def framework_success_total(self) -> int:
        return sum(c.framework_success_count for c in self.comparisons)

    @property
    def custom_trial_total(self) -> int:
        return sum(len(c.custom_trials) for c in self.comparisons)

    @property
    def framework_trial_total(self) -> int:
        return sum(len(c.framework_trials) for c in self.comparisons)

    @property
    def shared_scenarios(self) -> int:
        """Scenarios supported by both implementations (eligible for paired comparison)."""
        return sum(1 for c in self.comparisons if c.shared)

    @property
    def excluded_scenarios(self) -> list[dict[str, Any]]:
        """Scenarios not supported by both implementations, with reasons."""
        excluded: list[dict[str, Any]] = []
        for c in self.comparisons:
            if not c.shared:
                excluded.append(
                    {
                        "scenario_id": c.scenario_id,
                        "supported_implementations": c.supported_implementations,
                        "reason": "unsupported_by_one_implementation",
                    }
                )
        return excluded

    @property
    def successful_paired_trials_total(self) -> int:
        """Total successful paired trials across shared scenarios."""
        return sum(c.successful_paired_trials for c in self.comparisons)

    @property
    def failed_paired_trials_total(self) -> int:
        """Total failed paired trials across shared scenarios."""
        return sum(c.failed_paired_trials for c in self.comparisons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": {
                "warm_up": self.config.warm_up,
                "repetitions": self.config.repetitions,
                "live_enabled": self.config.live_enabled,
            },
            "environment": self.environment,
            "total_scenarios": self.total_scenarios,
            "shared_scenarios": self.shared_scenarios,
            "excluded_scenarios": self.excluded_scenarios,
            "custom_success_total": self.custom_success_total,
            "framework_success_total": self.framework_success_total,
            "custom_trial_total": self.custom_trial_total,
            "framework_trial_total": self.framework_trial_total,
            "successful_paired_trials_total": self.successful_paired_trials_total,
            "failed_paired_trials_total": self.failed_paired_trials_total,
            "scenarios": [c.to_dict() for c in self.comparisons],
        }


class PerformanceBenchmark:
    """Runs deterministic (and optionally live) performance trials."""

    def __init__(self, config: PerformanceBenchmarkConfig) -> None:
        self.config = config
        self.dataset = config.dataset or load_agent_dataset()

    def _run_trial(
        self,
        spec: AgentScenarioSpec,
        implementation: str,
        trial_number: int,
        execution_mode: str,
    ) -> PerformanceTrialResult:
        adapter = get_adapter(implementation)
        result: AgentScenarioResult = adapter.run(spec)

        # Estimated cost from pricing config.
        pricing = self.config.pricing or PricingConfig()
        model = result.model
        provider_name = result.provider
        price = get_model_price(model or "unknown", provider_name or "groq") if model else None
        estimated_cost = None
        pricing_source = None
        pricing_date = None
        if price is not None:
            estimated_cost = price.total_cost_usd(
                result.input_tokens or 0, result.output_tokens or 0
            )
            pricing_source = price.pricing_source
            pricing_date = price.pricing_date
        elif pricing.input_price_per_1m_tokens is not None and pricing.output_price_per_1m_tokens is not None:
            estimated_cost = (
                result.input_tokens or 0
            ) * pricing.input_price_per_1m_tokens / 1_000_000 + (
                result.output_tokens or 0
            ) * pricing.output_price_per_1m_tokens / 1_000_000
            pricing_source = pricing.pricing_source
            pricing_date = pricing.pricing_date

        return PerformanceTrialResult(
            scenario_id=spec.scenario_id,
            implementation=implementation,
            trial_number=trial_number,
            execution_mode=execution_mode,
            total_latency_ms=result.total_latency_ms,
            provider_latency_ms=result.provider_latency_ms,
            tool_latency_ms=result.tool_latency_ms,
            provider_calls=result.provider_calls,
            tool_calls_count=result.tool_calls_count,
            tool_failures=result.tool_failures,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            estimated_cost_usd=estimated_cost,
            model=result.model,
            provider=result.provider,
            pricing_source=pricing_source,
            pricing_date=pricing_date,
            status=result.status,
            error=result.error,
            execution_steps=result.execution_steps,
        )

    def _warm_up(self, spec: AgentScenarioSpec, implementation: str) -> None:
        """Run warm-up trials (results discarded)."""
        for _ in range(self.config.warm_up):
            self._run_trial(spec, implementation, 0, "deterministic")

    def run_all(self) -> PerformanceBenchmarkOutcome:
        """Run the full benchmark: warm-up + repetitions for every scenario."""
        env_info = self._collect_environment()
        outcome = PerformanceBenchmarkOutcome(config=self.config, environment=env_info)

        specs = self.dataset.scenarios
        implementations = ["custom", "framework"]

        for spec in specs:
            comparison = PerformanceComparison(
                scenario_id=spec.scenario_id,
                category=spec.category,
                supported_implementations=list(spec.supported_implementations),
            )

            for impl in implementations:
                if impl not in spec.supported_implementations:
                    continue

                # Warm-up (discarded).
                self._warm_up(spec, impl)

                # Measured repetitions.
                for trial in range(1, self.config.repetitions + 1):
                    trial_result = self._run_trial(
                        spec, impl, trial, "deterministic"
                    )
                    if impl == "custom":
                        comparison.custom_trials.append(trial_result)
                    else:
                        comparison.framework_trials.append(trial_result)

            outcome.comparisons.append(comparison)

        return outcome

    def _collect_environment(self) -> dict[str, Any]:
        import platform
        import sys

        return {
            "python_version": sys.version,
            "os": platform.platform(),
            "processor": platform.processor(),
            "warm_up_trials": self.config.warm_up,
            "measured_repetitions": self.config.repetitions,
            "live_enabled": self.config.live_enabled,
            "deterministic_only": not self.config.live_enabled,
            "note": "Deterministic results use scripted providers and do not "
                    "represent real LLM provider latency or token usage.",
        }


def run_deterministic_benchmark(
    warm_up: int = DEFAULT_WARM_UP,
    repetitions: int = DEFAULT_REPETITIONS,
    dataset: AgentEvaluationDataset | None = None,
    pricing: PricingConfig | None = None,
) -> PerformanceBenchmarkOutcome:
    """Run the deterministic performance benchmark."""
    config = PerformanceBenchmarkConfig(
        warm_up=warm_up,
        repetitions=repetitions,
        live_enabled=False,
        dataset=dataset,
        pricing=pricing,
    )
    return PerformanceBenchmark(config).run_all()


def should_run_live() -> bool:
    """Return True only when the F45_LIVE_ENABLED environment variable is set."""
    import os

    return os.environ.get(LIVE_ENABLED_ENV, "").lower() in ("1", "true", "yes", "on")


def write_reports(outcome: PerformanceBenchmarkOutcome) -> dict[str, Path]:
    """Write aggregate JSON and markdown reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    F45_SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    # Aggregate JSON.
    aggregate_path = F45_AGGREGATE_PATH
    aggregate_path.write_text(
        json.dumps(outcome.to_dict(), indent=2), encoding="utf-8"
    )
    written["aggregate"] = aggregate_path

    # Markdown.
    md_path = F45_MARKDOWN_PATH
    md_path.write_text(_render_markdown(outcome), encoding="utf-8")
    written["markdown"] = md_path

    # Per-scenario JSON.
    for comparison in outcome.comparisons:
        scenario_path = F45_SCENARIO_DIR / f"{comparison.scenario_id}.json"
        scenario_path.write_text(
            json.dumps(comparison.to_dict(), indent=2), encoding="utf-8"
        )
    written["scenario_dir"] = F45_SCENARIO_DIR

    return written


def _render_markdown(outcome: PerformanceBenchmarkOutcome) -> str:
    lines: list[str] = []
    lines.append("# F4.5 — Agent Performance, Token, and Cost Comparison")
    lines.append("")
    lines.append(
        "## Preliminary Results"
    )
    lines.append("")
    lines.append(
        "These measurements use deterministic scripted providers and controlled "
        "tool handlers. Sample sizes are small (3 repetitions per scenario), so "
        "results are **preliminary** and should not be used to declare a "
        "statistically significant performance winner. Small latency differences "
        "are not presented as meaningful production advantages."
    )
    lines.append("")
    lines.append(
        "Benchmark executed: " + outcome.timestamp
    )
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(
        f"- Warm-up trials: {outcome.config.warm_up}"
    )
    lines.append(
        f"- Measured repetitions: {outcome.config.repetitions}"
    )
    lines.append(
        f"- Live provider enabled: {outcome.config.live_enabled}"
    )
    lines.append(
        f"- Total scenarios: {outcome.total_scenarios}"
    )
    lines.append(
        f"- Shared scenarios (both implementations): {outcome.shared_scenarios}"
    )
    lines.append(
        f"- Excluded scenarios: {len(outcome.excluded_scenarios)}"
    )
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    for key, value in outcome.environment.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")
    lines.append("## Paired-Comparison Summary")
    lines.append("")
    lines.append(
        f"- Shared scenarios: {outcome.shared_scenarios}"
    )
    lines.append(
        f"- Successful paired trials: {outcome.successful_paired_trials_total}"
    )
    lines.append(
        f"- Failed paired trials: {outcome.failed_paired_trials_total}"
    )
    lines.append(
        f"- Custom success rate: {outcome.custom_success_total}/{outcome.custom_trial_total}"
    )
    lines.append(
        f"- Framework success rate: {outcome.framework_success_total}/{outcome.framework_trial_total}"
    )
    lines.append("")
    if outcome.excluded_scenarios:
        lines.append("### Excluded Scenarios")
        lines.append("")
        lines.append(
            "| Scenario | Supported Implementations | Reason |"
        )
        lines.append("|---|---|---|")
        for ex in outcome.excluded_scenarios:
            lines.append(
                f"| {ex['scenario_id']} | {', '.join(ex['supported_implementations'])} | "
                f"{ex['reason']} |"
            )
        lines.append("")
    lines.append("## Results by Scenario")
    lines.append("")
    lines.append(
        "| Scenario | Shared | Implementation | Trials | Success | Median Total (ms) | "
        "Mean Total (ms) | Median Provider (ms) | Median Tool (ms) | Overhead (ms) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for comparison in outcome.comparisons:
        agg = compute_performance_aggregate(
            comparison.custom_trials, comparison.framework_trials
        )
        for impl in ("custom", "framework"):
            impl_stats = agg[impl]
            trials = (
                comparison.custom_trials if impl == "custom"
                else comparison.framework_trials
            )
            if not trials:
                continue
            lines.append(
                f"| {comparison.scenario_id} | {'yes' if comparison.shared else 'no'} | "
                f"{impl} | {len(trials)} | "
                f"{impl_stats['success_count']}/{impl_stats['trial_count']} | "
                f"{impl_stats['median_total_latency_ms']} | "
                f"{impl_stats['mean_total_latency_ms']} | "
                f"{impl_stats['median_provider_latency_ms']} | "
                f"{impl_stats['median_tool_latency_ms']} | "
                f"{impl_stats['execution_overhead_ms']} |"
            )
        if comparison.shared:
            comp = agg["comparison"]
            rel = comp["relative_latency_difference"]
            lines.append(
                f"| {comparison.scenario_id} | yes | comparison | -- | -- | "
                f"-- | -- | -- | -- | -- | rel_diff={rel} |"
            )
    lines.append("")
    lines.append("## Aggregate Comparison")
    lines.append("")
    lines.append(
        "| Metric | Custom | Framework |"
    )
    lines.append("|---|---|---|")
    all_custom = [t for c in outcome.comparisons for t in c.custom_trials]
    all_framework = [t for c in outcome.comparisons for t in c.framework_trials]
    all_agg = compute_performance_aggregate(all_custom, all_framework)
    for metric in (
        "median_total_latency_ms", "mean_total_latency_ms",
        "median_provider_latency_ms", "median_tool_latency_ms",
        "execution_overhead_ms", "mean_estimated_cost_usd",
        "aggregate_estimated_cost_usd", "success_rate",
    ):
        cu = all_agg["custom"][metric]
        fw = all_agg["framework"][metric]
        lines.append(f"| {metric} | {cu} | {fw} |")
    lines.append("")
    rel = all_agg["comparison"]["relative_latency_difference"]
    lines.append(f"- Relative latency difference (custom - framework) / framework: {rel}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- Deterministic trials use scripted providers and controlled tool handlers. "
        "They measure adapter and orchestration overhead, not real LLM performance."
    )
    lines.append(
        "- Token usage is unavailable for deterministic providers and is reported as null."
    )
    lines.append(
        "- Cost estimates require actual token usage and configured model pricing; "
        "they are null when either is missing."
    )
    lines.append(
        "- Live Groq trials are opt-in (F45_LIVE_ENABLED) and are not mixed with "
        "deterministic results."
    )
    lines.append("")
    lines.append(
        "- Timing boundaries: total latency includes provider waiting, tool execution, "
        "retry delays, serialization, and adapter overhead. Provider latency is the "
        "time spent in provider.complete()/invoke(). Tool latency is the time spent "
        "in tool executor.execute(). Execution overhead = total - provider - tool."
    )
    lines.append(
        "- Timing boundary note: The Custom adapter counts limit-rejected tool calls "
        "in tool_latency_ms (they reach the executor and return ExecutionLimitError). "
        "The Framework adapter does not count limit-rejected calls in tool_latency_ms "
        "(they are blocked before reaching the executor). This difference is documented "
        "and affects execution_overhead_ms comparability between implementations."
    )
    lines.append("")
    lines.append(
        "- Token extraction from real Groq provider responses is not yet implemented "
        "in the comparison layer. All deterministic trials report null token usage."
    )
    return "\n".join(lines)
