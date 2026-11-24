"""Normalized performance, token, and cost metrics for F4.5.

All calculations are framework-independent and computed from actual
execution evidence captured by the performance benchmark. Every metric
documents its numerator and denominator semantics. Metrics return 0.0
when the denominator is zero (no signal); ``None`` signals the metric
was not measurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


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


def _average(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return sum(vals) / len(vals) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _min(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return min(vals) if vals else None


def _max(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values]
    return max(vals) if vals else None


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


# ---------------------------------------------------------------------------
# Per-trial statistics
# ---------------------------------------------------------------------------

def trial_stats(trials: Iterable[PerformanceTrialResult], attr: str) -> dict[str, float | None]:
    """Compute median/mean/min/max for a numeric attribute across trials.

    Missing values (None) are excluded. Returns None for all fields when
    no values are available.
    """
    values = [getattr(t, attr) for t in trials if getattr(t, attr) is not None]
    if not values:
        return {"median": None, "mean": None, "min": None, "max": None}
    return {
        "median": _median(values),
        "mean": _average(values),
        "min": _min(values),
        "max": _max(values),
    }


# ---------------------------------------------------------------------------
# Execution performance metrics
# ---------------------------------------------------------------------------

def median_total_latency_ms(trials: list[PerformanceTrialResult]) -> float | None:
    """Median total end-to-end latency in milliseconds.

    Includes provider waiting, tool execution, retry delays, serialization,
    and adapter overhead.
    """
    stats = trial_stats(trials, "total_latency_ms")
    return stats["median"]


def mean_total_latency_ms(trials: list[PerformanceTrialResult]) -> float | None:
    """Mean total end-to-end latency in milliseconds."""
    stats = trial_stats(trials, "total_latency_ms")
    return stats["mean"]


def median_provider_latency_ms(trials: list[PerformanceTrialResult]) -> float | None:
    """Median provider-call latency in milliseconds."""
    stats = trial_stats(trials, "provider_latency_ms")
    return stats["median"]


def median_tool_latency_ms(trials: list[PerformanceTrialResult]) -> float | None:
    """Median tool execution latency in milliseconds."""
    stats = trial_stats(trials, "tool_latency_ms")
    return stats["median"]


def execution_overhead_ms(trials: list[PerformanceTrialResult]) -> float | None:
    """Execution overhead = total - provider - tool, per trial, then averaged.

    Represents time spent in serialization, adapter overhead, and orchestration
    scheduling outside of provider and tool calls.
    """
    overheads = []
    for t in trials:
        if t.total_latency_ms is None:
            continue
        provider = t.provider_latency_ms or 0.0
        tool = t.tool_latency_ms or 0.0
        overheads.append(t.total_latency_ms - provider - tool)
    return _average(overheads) if overheads else None


def relative_latency_difference(
    custom_trials: list[PerformanceTrialResult],
    framework_trials: list[PerformanceTrialResult],
) -> float | None:
    """Relative latency difference = (custom - framework) / framework.

    Returns None when framework median is missing or zero. Positive values
    mean custom is slower; negative values mean custom is faster.
    """
    fw_median = median_total_latency_ms(framework_trials)
    cu_median = median_total_latency_ms(custom_trials)
    if fw_median is None or fw_median == 0 or cu_median is None:
        return None
    return (cu_median - fw_median) / fw_median


# ---------------------------------------------------------------------------
# Token usage metrics
# ---------------------------------------------------------------------------

def mean_input_tokens(trials: list[PerformanceTrialResult]) -> float | None:
    """Mean input tokens per scenario across successful trials."""
    values = [t.input_tokens for t in trials if t.input_tokens is not None]
    return _average(values) if values else None


def mean_output_tokens(trials: list[PerformanceTrialResult]) -> float | None:
    """Mean output tokens per scenario across successful trials."""
    values = [t.output_tokens for t in trials if t.output_tokens is not None]
    return _average(values) if values else None


def mean_total_tokens(trials: list[PerformanceTrialResult]) -> float | None:
    """Mean total tokens per scenario across successful trials."""
    values = [t.total_tokens for t in trials if t.total_tokens is not None]
    return _average(values) if values else None


def total_tokens_across_successful(trials: list[PerformanceTrialResult]) -> int | None:
    """Total tokens across all successful trials."""
    values = [t.total_tokens for t in trials if t.total_tokens is not None]
    return sum(values) if values else None


# ---------------------------------------------------------------------------
# Cost metrics
# ---------------------------------------------------------------------------

def mean_estimated_cost_usd(trials: list[PerformanceTrialResult]) -> float | None:
    """Mean estimated cost in USD per scenario."""
    values = [t.estimated_cost_usd for t in trials if t.estimated_cost_usd is not None]
    return _average(values) if values else None


def aggregate_estimated_cost_usd(trials: list[PerformanceTrialResult]) -> float | None:
    """Aggregate estimated cost across all successful trials."""
    values = [t.estimated_cost_usd for t in trials if t.estimated_cost_usd is not None]
    return sum(values) if values else None


# ---------------------------------------------------------------------------
# Reliability context
# ---------------------------------------------------------------------------

def success_rate(trials: list[PerformanceTrialResult]) -> float:
    """Success rate = successful trials / total trials.

    Denominator: all trials executed for this implementation/scenario.
    """
    if not trials:
        return 0.0
    return _average(1.0 if t.status == "success" else 0.0 for t in trials)


def failure_count(trials: list[PerformanceTrialResult]) -> int:
    """Number of failed trials."""
    return sum(1 for t in trials if t.status != "success")


def provider_call_count(trials: list[PerformanceTrialResult]) -> int:
    """Total provider calls across all trials."""
    return sum(t.provider_calls for t in trials)


def tool_call_count(trials: list[PerformanceTrialResult]) -> int:
    """Total tool calls across all trials."""
    return sum(t.tool_calls_count for t in trials)


# ---------------------------------------------------------------------------
# Aggregate computation
# ---------------------------------------------------------------------------

def compute_performance_aggregate(
    custom_trials: list[PerformanceTrialResult],
    framework_trials: list[PerformanceTrialResult],
) -> dict[str, Any]:
    """Compute normalized aggregate metrics for a side-by-side comparison.

    Denominators are documented per metric. Missing metrics are reported as
    None (not measurable) rather than estimated.
    """
    return {
        "custom": {
            "trial_count": len(custom_trials),
            "success_count": sum(1 for t in custom_trials if t.status == "success"),
            "failure_count": failure_count(custom_trials),
            "success_rate": success_rate(custom_trials),
            "median_total_latency_ms": median_total_latency_ms(custom_trials),
            "mean_total_latency_ms": mean_total_latency_ms(custom_trials),
            "median_provider_latency_ms": median_provider_latency_ms(custom_trials),
            "median_tool_latency_ms": median_tool_latency_ms(custom_trials),
            "execution_overhead_ms": execution_overhead_ms(custom_trials),
            "mean_input_tokens": mean_input_tokens(custom_trials),
            "mean_output_tokens": mean_output_tokens(custom_trials),
            "mean_total_tokens": mean_total_tokens(custom_trials),
            "total_tokens_across_successful": total_tokens_across_successful(
                custom_trials
            ),
            "mean_estimated_cost_usd": mean_estimated_cost_usd(custom_trials),
            "aggregate_estimated_cost_usd": aggregate_estimated_cost_usd(
                custom_trials
            ),
            "provider_call_count": provider_call_count(custom_trials),
            "tool_call_count": tool_call_count(custom_trials),
        },
        "framework": {
            "trial_count": len(framework_trials),
            "success_count": sum(1 for t in framework_trials if t.status == "success"),
            "failure_count": failure_count(framework_trials),
            "success_rate": success_rate(framework_trials),
            "median_total_latency_ms": median_total_latency_ms(framework_trials),
            "mean_total_latency_ms": mean_total_latency_ms(framework_trials),
            "median_provider_latency_ms": median_provider_latency_ms(framework_trials),
            "median_tool_latency_ms": median_tool_latency_ms(framework_trials),
            "execution_overhead_ms": execution_overhead_ms(framework_trials),
            "mean_input_tokens": mean_input_tokens(framework_trials),
            "mean_output_tokens": mean_output_tokens(framework_trials),
            "mean_total_tokens": mean_total_tokens(framework_trials),
            "total_tokens_across_successful": total_tokens_across_successful(
                framework_trials
            ),
            "mean_estimated_cost_usd": mean_estimated_cost_usd(framework_trials),
            "aggregate_estimated_cost_usd": aggregate_estimated_cost_usd(
                framework_trials
            ),
            "provider_call_count": provider_call_count(framework_trials),
            "tool_call_count": tool_call_count(framework_trials),
        },
        "comparison": {
            "relative_latency_difference": relative_latency_difference(
                custom_trials, framework_trials
            ),
            "custom_slower": (
                relative_latency_difference(custom_trials, framework_trials) or 0
            )
            > 0,
        },
    }
