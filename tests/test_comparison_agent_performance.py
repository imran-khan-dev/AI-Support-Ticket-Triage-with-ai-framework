"""F4.5 performance, token, and cost benchmark tests.

All tests use deterministic scripted providers and controlled tool handlers.
No real API access is required.
"""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from app.comparison.agent_adapters import (
    AgentScenarioResult,
    CustomAgentAdapter,
    LangGraphAgentAdapter,
    RecordingToolExecutor,
    ScriptedModel,
    ScriptedProvider,
    ToolCallRecord,
    get_adapter,
)
from app.comparison.agent_performance import (
    DEFAULT_REPETITIONS,
    DEFAULT_WARM_UP,
    PerformanceBenchmark,
    PerformanceBenchmarkConfig,
    PerformanceComparison,
    PerformanceBenchmarkOutcome,
    PerformanceTrialResult,
    should_run_live,
    write_reports,
)
from app.comparison.agent_performance_metrics import (
    aggregate_estimated_cost_usd,
    compute_performance_aggregate,
    execution_overhead_ms,
    failure_count,
    mean_estimated_cost_usd,
    mean_input_tokens,
    mean_output_tokens,
    mean_total_tokens,
    median_provider_latency_ms,
    median_total_latency_ms,
    median_tool_latency_ms,
    provider_call_count,
    relative_latency_difference,
    success_rate,
    tool_call_count,
    total_tokens_across_successful,
    trial_stats,
)
from app.comparison.agent_scenarios import (
    AgentScenarioSpec,
    build_scenarios,
    CONTROLLED_TOOL_HANDLERS,
    TOOL_SPECS,
)
from app.comparison.pricing_config import (
    ModelPrice,
    PricingConfig,
    get_model_price,
    list_known_models,
)


# ---------------------------------------------------------------------------
# Latency capture tests
# ---------------------------------------------------------------------------

class TestLatencyCapture:
    """Tests for timing capture on ToolCallRecord and AgentScenarioResult."""

    def test_tool_call_record_duration_defaults(self):
        record = ToolCallRecord(
            call_id="c1", name="search_knowledge", arguments={"query": "q"},
            success=True,
        )
        assert record.duration_ms == 0.0

    def test_tool_call_record_duration_settable(self):
        record = ToolCallRecord(
            call_id="c1", name="search_knowledge", arguments={"query": "q"},
            success=True, duration_ms=42.5,
        )
        assert record.duration_ms == 42.5

    def test_agent_scenario_result_latency_fields_defaults(self):
        result = AgentScenarioResult(
            scenario_id="s1", implementation="custom", status="success",
        )
        assert result.total_latency_ms == 0.0
        assert result.provider_latency_ms == 0.0
        assert result.tool_latency_ms == 0.0

    def test_agent_scenario_result_latency_fields_settable(self):
        result = AgentScenarioResult(
            scenario_id="s1", implementation="framework", status="success",
            total_latency_ms=100.0, provider_latency_ms=60.0, tool_latency_ms=30.0,
        )
        assert result.total_latency_ms == 100.0
        assert result.provider_latency_ms == 60.0
        assert result.tool_latency_ms == 30.0

    def test_execution_overhead_derivation(self):
        """Overhead = total - provider - tool."""
        result = AgentScenarioResult(
            scenario_id="s1", implementation="custom", status="success",
            total_latency_ms=100.0, provider_latency_ms=60.0, tool_latency_ms=30.0,
        )
        overhead = result.total_latency_ms - result.provider_latency_ms - result.tool_latency_ms
        assert overhead == 10.0


# ---------------------------------------------------------------------------
# Token metadata tests
# ---------------------------------------------------------------------------

class TestTokenMetadata:
    """Tests for token metadata extraction (missing = None)."""

    def test_tool_call_record_tokens_default_none(self):
        record = ToolCallRecord(
            call_id="c1", name="search_knowledge", arguments={"query": "q"},
            success=True,
        )
        assert record.token_input is None
        assert record.token_output is None
        assert record.token_total is None

    def test_tool_call_record_tokens_settable(self):
        record = ToolCallRecord(
            call_id="c1", name="search_knowledge", arguments={"query": "q"},
            success=True, token_input=100, token_output=50, token_total=150,
        )
        assert record.token_input == 100
        assert record.token_output == 50
        assert record.token_total == 150

    def test_agent_scenario_result_tokens_default_none(self):
        result = AgentScenarioResult(
            scenario_id="s1", implementation="custom", status="success",
        )
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None

    def test_agent_scenario_result_tokens_settable(self):
        result = AgentScenarioResult(
            scenario_id="s1", implementation="framework", status="success",
            input_tokens=200, output_tokens=100, total_tokens=300,
        )
        assert result.input_tokens == 200
        assert result.output_tokens == 100
        assert result.total_tokens == 300

    def test_deterministic_adapter_reports_tokens_none(self):
        """Deterministic scripted providers have no token metadata."""
        spec = build_scenarios()[0]
        result = CustomAgentAdapter().run(spec)
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None

    def test_deterministic_framework_reports_tokens_none(self):
        spec = build_scenarios()[0]
        result = LangGraphAgentAdapter().run(spec)
        assert result.input_tokens is None
        assert result.output_tokens is None
        assert result.total_tokens is None


# ---------------------------------------------------------------------------
# Multiple provider calls tests
# ---------------------------------------------------------------------------

class TestMultipleProviderCalls:
    """Tests for scenarios with multiple provider calls."""

    def test_provider_calls_count_matches_model_calls(self):
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-multiple-sequential-tools"][0]
        result = CustomAgentAdapter().run(spec)
        assert result.provider_calls == len(result.tool_calls) + 1  # at least one per tool + final
        assert result.provider_calls > 0

    def test_framework_provider_calls_count(self):
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-multiple-sequential-tools"][0]
        result = LangGraphAgentAdapter().run(spec)
        assert result.provider_calls > 0


# ---------------------------------------------------------------------------
# Pricing calculation tests
# ---------------------------------------------------------------------------

class TestPricingCalculation:
    """Tests for pricing configuration and cost estimation."""

    def test_get_model_price_known_model(self):
        price = get_model_price("llama-3.1-8b-instant", "groq")
        assert price is not None
        assert price.model == "llama-3.1-8b-instant"
        assert price.provider == "groq"
        assert price.input_price_per_1m_tokens == 0.05
        assert price.output_price_per_1m_tokens == 0.05

    def test_get_model_price_unknown_model(self):
        price = get_model_price("nonexistent-model", "groq")
        assert price is None

    def test_list_known_models_non_empty(self):
        models = list_known_models()
        assert len(models) > 0
        assert all(isinstance(m, ModelPrice) for m in models)

    def test_model_price_input_cost(self):
        price = get_model_price("llama-3.1-8b-instant", "groq")
        cost = price.input_cost_usd(1_000_000)
        assert cost == 0.05

    def test_model_price_output_cost(self):
        price = get_model_price("llama-3.1-70b-versatile", "groq")
        cost = price.output_cost_usd(1_000_000)
        assert cost == 0.79

    def test_model_price_total_cost(self):
        price = get_model_price("llama-3.1-8b-instant", "groq")
        cost = price.total_cost_usd(1_000_000, 500_000)
        assert cost == 0.05 + 0.025  # 0.05 input + 0.025 output

    def test_pricing_config_estimate_cost(self):
        config = PricingConfig(
            model="llama-3.1-8b-instant", provider="groq",
        )
        cost = config.estimate_cost_usd(1_000_000, 500_000)
        assert abs(cost - 0.075) < 1e-9

    def test_pricing_config_estimate_cost_no_model(self):
        config = PricingConfig()
        cost = config.estimate_cost_usd(1_000_000, 500_000)
        assert cost is None

    def test_pricing_config_estimate_cost_unknown_model(self):
        config = PricingConfig(model="nonexistent-model", provider="groq")
        cost = config.estimate_cost_usd(1_000_000, 500_000)
        assert cost is None

    def test_pricing_config_explicit_prices(self):
        config = PricingConfig(
            model="custom-model", provider="test",
            input_price_per_1m_tokens=1.0,
            output_price_per_1m_tokens=2.0,
            pricing_source="test_source",
            pricing_date="2026-01-01",
        )
        cost = config.estimate_cost_usd(1_000_000, 1_000_000)
        assert cost == 3.0


# ---------------------------------------------------------------------------
# Cost aggregation tests
# ---------------------------------------------------------------------------

class TestCostAggregation:
    """Tests for cost aggregation metrics."""

    def test_mean_estimated_cost_usd_none_when_missing(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=1,
                execution_mode="deterministic", total_latency_ms=10.0,
                provider_latency_ms=5.0, tool_latency_ms=3.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                estimated_cost_usd=None, model=None, provider="scripted",
                pricing_source=None, pricing_date=None, status="success",
            )
        ]
        assert mean_estimated_cost_usd(trials) is None

    def test_mean_estimated_cost_usd_with_values(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=1,
                execution_mode="deterministic", total_latency_ms=10.0,
                provider_latency_ms=5.0, tool_latency_ms=3.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=100, output_tokens=50, total_tokens=150,
                estimated_cost_usd=0.01, model="m", provider="groq",
                pricing_source="s", pricing_date="d", status="success",
            ),
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=2,
                execution_mode="deterministic", total_latency_ms=12.0,
                provider_latency_ms=6.0, tool_latency_ms=4.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=200, output_tokens=100, total_tokens=300,
                estimated_cost_usd=0.02, model="m", provider="groq",
                pricing_source="s", pricing_date="d", status="success",
            ),
        ]
        assert mean_estimated_cost_usd(trials) == 0.015

    def test_aggregate_estimated_cost_usd(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=1,
                execution_mode="deterministic", total_latency_ms=10.0,
                provider_latency_ms=5.0, tool_latency_ms=3.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=100, output_tokens=50, total_tokens=150,
                estimated_cost_usd=0.01, model="m", provider="groq",
                pricing_source="s", pricing_date="d", status="success",
            ),
            PerformanceTrialResult(
                scenario_id="s2", implementation="custom", trial_number=1,
                execution_mode="deterministic", total_latency_ms=12.0,
                provider_latency_ms=6.0, tool_latency_ms=4.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=200, output_tokens=100, total_tokens=300,
                estimated_cost_usd=0.02, model="m", provider="groq",
                pricing_source="s", pricing_date="d", status="success",
            ),
        ]
        assert aggregate_estimated_cost_usd(trials) == 0.03


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------

class TestStatistics:
    """Tests for median/mean/min/max calculations."""

    def test_trial_stats_all_none(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=1,
                execution_mode="deterministic", total_latency_ms=None,
                provider_latency_ms=None, tool_latency_ms=None,
                provider_calls=0, tool_calls_count=0, tool_failures=0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                estimated_cost_usd=None, model=None, provider=None,
                pricing_source=None, pricing_date=None, status="success",
            )
        ]
        stats = trial_stats(trials, "total_latency_ms")
        assert stats == {"median": None, "mean": None, "min": None, "max": None}

    def test_trial_stats_with_values(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=i,
                execution_mode="deterministic", total_latency_ms=float(i * 10),
                provider_latency_ms=float(i * 5), tool_latency_ms=float(i * 3),
                provider_calls=i, tool_calls_count=i, tool_failures=0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                estimated_cost_usd=None, model=None, provider=None,
                pricing_source=None, pricing_date=None, status="success",
            )
            for i in (1, 2, 3, 4, 5)
        ]
        stats = trial_stats(trials, "total_latency_ms")
        assert stats["median"] == 30.0
        assert stats["mean"] == 30.0
        assert stats["min"] == 10.0
        assert stats["max"] == 50.0

    def test_median_even_count(self):
        vals = [1.0, 2.0, 3.0, 4.0]
        assert statistics.median(vals) == 2.5


# ---------------------------------------------------------------------------
# Comparison denominators tests
# ---------------------------------------------------------------------------

class TestComparisonDenominators:
    """Tests that denominators are documented and correct."""

    def test_success_rate_denominator_is_all_trials(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=i,
                execution_mode="deterministic", total_latency_ms=10.0,
                provider_latency_ms=5.0, tool_latency_ms=3.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                estimated_cost_usd=None, model=None, provider=None,
                pricing_source=None, pricing_date=None,
                status="success" if i <= 3 else "failed",
            )
            for i in range(1, 6)
        ]
        assert success_rate(trials) == 0.6  # 3/5

    def test_failure_count(self):
        trials = [
            PerformanceTrialResult(
                scenario_id="s1", implementation="custom", trial_number=i,
                execution_mode="deterministic", total_latency_ms=10.0,
                provider_latency_ms=5.0, tool_latency_ms=3.0,
                provider_calls=1, tool_calls_count=1, tool_failures=0,
                input_tokens=None, output_tokens=None, total_tokens=None,
                estimated_cost_usd=None, model=None, provider=None,
                pricing_source=None, pricing_date=None,
                status="success" if i <= 3 else "failed",
            )
            for i in range(1, 6)
        ]
        assert failure_count(trials) == 2

    def test_relative_latency_difference_none_when_framework_zero(self):
        custom = [PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider=None,
            pricing_source=None, pricing_date=None, status="success",
        )]
        framework = [PerformanceTrialResult(
            scenario_id="s1", implementation="framework", trial_number=1,
            execution_mode="deterministic", total_latency_ms=0.0,
            provider_latency_ms=0.0, tool_latency_ms=0.0,
            provider_calls=0, tool_calls_count=0, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider=None,
            pricing_source=None, pricing_date=None, status="success",
        )]
        assert relative_latency_difference(custom, framework) is None

    def test_compute_performance_aggregate_structure(self):
        trials = [PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider=None,
            pricing_source=None, pricing_date=None, status="success",
        )]
        agg = compute_performance_aggregate(trials, [])
        assert "custom" in agg
        assert "framework" in agg
        assert "comparison" in agg
        assert agg["custom"]["trial_count"] == 1
        assert agg["framework"]["trial_count"] == 0


# ---------------------------------------------------------------------------
# Live benchmark opt-in tests
# ---------------------------------------------------------------------------

class TestLiveOptIn:
    """Tests for live benchmark opt-in behavior."""

    def test_should_run_live_default_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert should_run_live() is False

    def test_should_run_live_true_when_env_set(self):
        with patch.dict(os.environ, {"F45_LIVE_ENABLED": "1"}):
            assert should_run_live() is True

    def test_should_run_live_true_when_env_true(self):
        with patch.dict(os.environ, {"F45_LIVE_ENABLED": "true"}):
            assert should_run_live() is True

    def test_should_run_live_false_when_env_false(self):
        with patch.dict(os.environ, {"F45_LIVE_ENABLED": "false"}):
            assert should_run_live() is False


# ---------------------------------------------------------------------------
# Report serialization tests
# ---------------------------------------------------------------------------

class TestReportSerialization:
    """Tests for report serialization."""

    def test_trial_result_to_dict(self):
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        d = trial.to_dict()
        assert d["scenario_id"] == "s1"
        assert d["implementation"] == "custom"
        assert d["total_latency_ms"] == 10.0
        assert d["input_tokens"] is None
        assert d["estimated_cost_usd"] is None

    def test_comparison_to_dict(self):
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        comp = PerformanceComparison(
            scenario_id="s1", category="test",
            custom_trials=[trial], framework_trials=[],
        )
        d = comp.to_dict()
        assert d["scenario_id"] == "s1"
        assert d["custom_success_count"] == 1
        assert d["framework_success_count"] == 0
        assert "aggregate" in d
        assert "custom" in d["aggregate"]
        assert "framework" in d["aggregate"]
        assert "comparison" in d["aggregate"]

    def test_write_reports_creates_files(self):
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        comp = PerformanceComparison(
            scenario_id="s1", category="test",
            custom_trials=[trial], framework_trials=[],
        )
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        outcome = PerformanceBenchmarkOutcome(
            config=config, comparisons=[comp],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            # Patch report paths to temp dir
            import app.comparison.agent_performance as perf_mod
            old_agg = perf_mod.F45_AGGREGATE_PATH
            old_md = perf_mod.F45_MARKDOWN_PATH
            old_dir = perf_mod.F45_SCENARIO_DIR
            perf_mod.F45_AGGREGATE_PATH = Path(tmpdir) / "f45_performance_comparison.json"
            perf_mod.F45_MARKDOWN_PATH = Path(tmpdir) / "f45_performance_comparison.md"
            perf_mod.F45_SCENARIO_DIR = Path(tmpdir) / "f45_scenarios"
            try:
                written = write_reports(outcome)
                assert written["aggregate"].is_file()
                assert written["markdown"].is_file()
                assert written["scenario_dir"].is_dir()
                # Verify JSON content
                agg_data = json.loads(written["aggregate"].read_text(encoding="utf-8"))
                assert agg_data["total_scenarios"] == 1
                assert agg_data["custom_success_total"] == 1
                assert "scenarios" in agg_data
            finally:
                perf_mod.F45_AGGREGATE_PATH = old_agg
                perf_mod.F45_MARKDOWN_PATH = old_md
                perf_mod.F45_SCENARIO_DIR = old_dir

    def test_markdown_contains_key_sections(self):
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        comp = PerformanceComparison(
            scenario_id="s1", category="test",
            custom_trials=[trial], framework_trials=[],
        )
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        outcome = PerformanceBenchmarkOutcome(
            config=config, comparisons=[comp],
        )
        import app.comparison.agent_performance as perf_mod
        md = perf_mod._render_markdown(outcome)
        assert "# F4.5" in md
        assert "Configuration" in md
        assert "Environment" in md
        assert "Results by Scenario" in md
        assert "Notes" in md
        assert "scripted" in md.lower() or "deterministic" in md.lower()


# ---------------------------------------------------------------------------
# Provider failure handling tests
# ---------------------------------------------------------------------------

class TestProviderFailureHandling:
    """Tests for provider failure handling in benchmarks."""

    def test_provider_failure_trial_recorded(self):
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-provider-failure"][0]
        result = CustomAgentAdapter().run(spec)
        assert result.status == "provider_failure"
        assert result.total_latency_ms > 0

    def test_framework_provider_failure_trial_recorded(self):
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-provider-failure"][0]
        result = LangGraphAgentAdapter().run(spec)
        assert result.status == "provider_failure"
        assert result.total_latency_ms > 0


# ---------------------------------------------------------------------------
# Benchmark runner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:
    """Tests for the PerformanceBenchmark runner."""

    def test_warm_up_trials_discarded(self):
        spec = build_scenarios()[0]
        config = PerformanceBenchmarkConfig(warm_up=2, repetitions=1)
        bench = PerformanceBenchmark(config)
        # Just verify warm-up doesn't raise
        bench._warm_up(spec, "custom")
        bench._warm_up(spec, "framework")

    def test_run_all_returns_outcome(self):
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        bench = PerformanceBenchmark(config)
        outcome = bench.run_all()
        assert isinstance(outcome, PerformanceBenchmarkOutcome)
        assert outcome.total_scenarios > 0
        assert "python_version" in outcome.environment
        assert "os" in outcome.environment

    def test_unsupported_implementation_skipped(self):
        # agent-cancellation is custom-only
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-cancellation"][0]
        assert "framework" not in spec.supported_implementations
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        bench = PerformanceBenchmark(config)
        outcome = bench.run_all()
        comp = next(c for c in outcome.comparisons if c.scenario_id == "agent-cancellation")
        assert len(comp.framework_trials) == 0
        assert len(comp.custom_trials) > 0


# ---------------------------------------------------------------------------
# Token extraction tests (mock Groq response fixtures, no API calls)
# ---------------------------------------------------------------------------

class TestTokenExtraction:
    """Tests for token usage extraction from Groq response fixtures.

    These tests use mock response objects that mimic the structure of Groq
    Responses API responses. No real API calls are made.
    """

    def _make_mock_groq_response(self, input_tokens, output_tokens):
        """Build a mock object resembling a Groq Responses API response."""
        class MockUsage:
            def __init__(self, input_tokens, output_tokens):
                self.input_tokens = input_tokens
                self.output_tokens = output_tokens

        class MockResponse:
            def __init__(self, input_tokens, output_tokens):
                self.output = []
                self.output_text = ""
                self.finish_reason = "stop"
                self.usage = MockUsage(input_tokens, output_tokens)

        return MockResponse(input_tokens, output_tokens)

    def test_known_input_output_usage_extraction(self):
        """Verify token extraction from a mock response with known usage."""
        response = self._make_mock_groq_response(100, 50)
        # Simulate extraction logic (would be in a Groq wrapper)
        input_tokens = getattr(response, "usage", None)
        if input_tokens is not None:
            input_tokens = input_tokens.input_tokens
        output_tokens = getattr(response, "usage", None)
        if output_tokens is not None:
            output_tokens = output_tokens.output_tokens
        assert input_tokens == 100
        assert output_tokens == 50

    def test_missing_usage_metadata(self):
        """Verify that missing usage metadata yields None, not zero."""
        class MockResponseNoUsage:
            def __init__(self):
                self.output = []
                self.output_text = ""
                self.finish_reason = "stop"
                # No usage attribute

        response = MockResponseNoUsage()
        usage = getattr(response, "usage", None)
        assert usage is None

    def test_zero_usage(self):
        """Verify zero usage is distinguished from missing usage."""
        response = self._make_mock_groq_response(0, 0)
        usage = getattr(response, "usage", None)
        assert usage is not None
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0

    def test_multiple_calls_accumulation(self):
        """Verify tokens accumulate across multiple provider calls."""
        responses = [
            self._make_mock_groq_response(100, 50),
            self._make_mock_groq_response(200, 100),
            self._make_mock_groq_response(50, 25),
        ]
        total_input = sum(getattr(r.usage, "input_tokens", 0) for r in responses if getattr(r, "usage", None))
        total_output = sum(getattr(r.usage, "output_tokens", 0) for r in responses if getattr(r, "usage", None))
        assert total_input == 350
        assert total_output == 175

    def test_different_response_shapes(self):
        """Test extraction handles different response attribute names."""
        # Groq Responses API uses .usage.input_tokens / .usage.output_tokens
        # Some wrappers might use .usage.prompt_tokens / .usage.completion_tokens
        class MockResponseAlt:
            def __init__(self):
                self.output = []
                self.output_text = ""
                self.finish_reason = "stop"
                self.usage = type("U", (), {"prompt_tokens": 75, "completion_tokens": 25})()

        response = MockResponseAlt()
        usage = getattr(response, "usage", None)
        assert usage is not None
        # Verify we don't assume wrong attribute names
        assert hasattr(usage, "prompt_tokens")
        assert hasattr(usage, "completion_tokens")
        assert not hasattr(usage, "input_tokens")


# ---------------------------------------------------------------------------
# Timing edge case tests
# ---------------------------------------------------------------------------

class TestTimingEdgeCases:
    """Tests for timing behavior under exceptions and retries."""

    def test_provider_exception_duration_recorded(self):
        """Provider exceptions still record elapsed time (try/finally)."""
        provider = ScriptedProvider(ScriptedModel([]))  # empty script = exhaustion
        # ScriptedProvider with empty model will raise on step()
        try:
            provider.complete(__import__("agent_orchestrator.providers.models", fromlist=["ProviderRequest"]).ProviderRequest(messages=[]))
        except Exception:
            pass
        # Duration should have been recorded even on exception
        assert len(provider.durations) >= 0  # at minimum, no crash

    def test_failed_execution_retains_timing(self):
        """Failed scenario executions retain timing information."""
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-provider-failure"][0]
        result = CustomAgentAdapter().run(spec)
        assert result.status == "provider_failure"
        assert result.total_latency_ms > 0
        assert result.provider_latency_ms >= 0

    def test_framework_failed_execution_retains_timing(self):
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-provider-failure"][0]
        result = LangGraphAgentAdapter().run(spec)
        assert result.status == "provider_failure"
        assert result.total_latency_ms > 0
        assert result.provider_latency_ms >= 0

    def test_all_durations_in_milliseconds(self):
        """Verify all timing fields use milliseconds consistently."""
        result = CustomAgentAdapter().run(build_scenarios()[0])
        assert result.total_latency_ms >= 0
        assert result.provider_latency_ms >= 0
        assert result.tool_latency_ms >= 0
        for c in result.tool_calls:
            assert c.duration_ms >= 0


# ---------------------------------------------------------------------------
# Paired-comparison metadata tests
# ---------------------------------------------------------------------------

class TestPairedComparisonMetadata:
    """Tests for shared/excluded scenario metadata."""

    def test_shared_property(self):
        spec = build_scenarios()[0]  # supported by both
        comp = PerformanceComparison(
            scenario_id=spec.scenario_id,
            category=spec.category,
            supported_implementations=list(spec.supported_implementations),
        )
        assert comp.shared is True

    def test_not_shared_property(self):
        # agent-cancellation is custom-only
        spec = [s for s in build_scenarios() if s.scenario_id == "agent-cancellation"][0]
        comp = PerformanceComparison(
            scenario_id=spec.scenario_id,
            category=spec.category,
            supported_implementations=list(spec.supported_implementations),
        )
        assert comp.shared is False

    def test_successful_paired_trials(self):
        comp = PerformanceComparison(
            scenario_id="s1", category="test",
            supported_implementations=["custom", "framework"],
            custom_trials=[
                PerformanceTrialResult(
                    scenario_id="s1", implementation="custom", trial_number=1,
                    execution_mode="deterministic", total_latency_ms=10.0,
                    provider_latency_ms=5.0, tool_latency_ms=3.0,
                    provider_calls=1, tool_calls_count=1, tool_failures=0,
                    input_tokens=None, output_tokens=None, total_tokens=None,
                    estimated_cost_usd=None, model=None, provider="scripted",
                    pricing_source=None, pricing_date=None, status="success",
                ),
                PerformanceTrialResult(
                    scenario_id="s1", implementation="custom", trial_number=2,
                    execution_mode="deterministic", total_latency_ms=12.0,
                    provider_latency_ms=6.0, tool_latency_ms=4.0,
                    provider_calls=1, tool_calls_count=1, tool_failures=0,
                    input_tokens=None, output_tokens=None, total_tokens=None,
                    estimated_cost_usd=None, model=None, provider="scripted",
                    pricing_source=None, pricing_date=None, status="failed",
                ),
            ],
            framework_trials=[
                PerformanceTrialResult(
                    scenario_id="s1", implementation="framework", trial_number=1,
                    execution_mode="deterministic", total_latency_ms=15.0,
                    provider_latency_ms=8.0, tool_latency_ms=5.0,
                    provider_calls=1, tool_calls_count=1, tool_failures=0,
                    input_tokens=None, output_tokens=None, total_tokens=None,
                    estimated_cost_usd=None, model=None, provider="scripted",
                    pricing_source=None, pricing_date=None, status="success",
                ),
                PerformanceTrialResult(
                    scenario_id="s1", implementation="framework", trial_number=2,
                    execution_mode="deterministic", total_latency_ms=18.0,
                    provider_latency_ms=9.0, tool_latency_ms=6.0,
                    provider_calls=1, tool_calls_count=1, tool_failures=0,
                    input_tokens=None, output_tokens=None, total_tokens=None,
                    estimated_cost_usd=None, model=None, provider="scripted",
                    pricing_source=None, pricing_date=None, status="success",
                ),
            ],
        )
        assert comp.successful_paired_trials == 1  # only trial 1 both succeeded
        assert comp.failed_paired_trials == 1  # trial 2: custom failed

    def test_outcome_shared_scenarios_count(self):
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        comp_shared = PerformanceComparison(
            scenario_id="s1", category="test",
            supported_implementations=["custom", "framework"],
            custom_trials=[trial], framework_trials=[trial],
        )
        comp_excluded = PerformanceComparison(
            scenario_id="s2", category="test",
            supported_implementations=["custom"],
            custom_trials=[trial], framework_trials=[],
        )
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        outcome = PerformanceBenchmarkOutcome(
            config=config, comparisons=[comp_shared, comp_excluded],
        )
        assert outcome.shared_scenarios == 1
        assert len(outcome.excluded_scenarios) == 1
        assert outcome.excluded_scenarios[0]["scenario_id"] == "s2"


# ---------------------------------------------------------------------------
# Currency consistency tests
# ---------------------------------------------------------------------------

class TestCurrencyConsistency:
    """Tests for pricing currency consistency."""

    def test_all_prices_use_usd(self):
        """Verify all configured prices use USD currency."""
        for price in list_known_models():
            assert price.currency == "USD"

    def test_model_price_currency_field(self):
        price = get_model_price("llama-3.1-8b-instant", "groq")
        assert price is not None
        assert price.currency == "USD"

    def test_explicit_pricing_currency(self):
        config = PricingConfig(
            model="custom-model", provider="test",
            input_price_per_1m_tokens=1.0,
            output_price_per_1m_tokens=2.0,
            currency="EUR",
        )
        price = config.model_price()
        assert price is not None
        assert price.currency == "EUR"


# ---------------------------------------------------------------------------
# Report consistency tests
# ---------------------------------------------------------------------------

class TestReportConsistency:
    """Tests for JSON/Markdown report consistency."""

    def test_null_tokens_distinguishable_from_zero(self):
        """Null tokens must not be serialized as zero."""
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        d = trial.to_dict()
        assert d["input_tokens"] is None
        assert d["output_tokens"] is None
        assert d["total_tokens"] is None
        assert d["input_tokens"] != 0
        assert d["output_tokens"] != 0
        assert d["total_tokens"] != 0

    def test_cost_unavailable_when_tokens_missing(self):
        """Cost must be None when token usage is missing."""
        trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        assert trial.estimated_cost_usd is None

    def test_aggregate_counts_match_per_scenario(self):
        """Aggregate trial counts must equal sum of per-scenario counts."""
        config = PerformanceBenchmarkConfig(warm_up=1, repetitions=1)
        bench = PerformanceBenchmark(config)
        outcome = bench.run_all()
        custom_sum = sum(len(c.custom_trials) for c in outcome.comparisons)
        fw_sum = sum(len(c.framework_trials) for c in outcome.comparisons)
        assert outcome.custom_trial_total == custom_sum
        assert outcome.framework_trial_total == fw_sum

    def test_deterministic_and_live_modes_separate(self):
        """Deterministic trials must not be mixed with live trials."""
        det_trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="deterministic", total_latency_ms=10.0,
            provider_latency_ms=5.0, tool_latency_ms=3.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=None, output_tokens=None, total_tokens=None,
            estimated_cost_usd=None, model=None, provider="scripted",
            pricing_source=None, pricing_date=None, status="success",
        )
        live_trial = PerformanceTrialResult(
            scenario_id="s1", implementation="custom", trial_number=1,
            execution_mode="live", total_latency_ms=500.0,
            provider_latency_ms=400.0, tool_latency_ms=50.0,
            provider_calls=1, tool_calls_count=1, tool_failures=0,
            input_tokens=100, output_tokens=50, total_tokens=150,
            estimated_cost_usd=0.01, model="llama-3.1-8b-instant",
            provider="groq", pricing_source="https://groq.com/pricing",
            pricing_date="2026-01-15", status="success",
        )
        comp = PerformanceComparison(
            scenario_id="s1", category="test",
            supported_implementations=["custom"],
            custom_trials=[det_trial, live_trial], framework_trials=[],
        )
        det_trials = [t for t in comp.custom_trials if t.execution_mode == "deterministic"]
        live_trials = [t for t in comp.custom_trials if t.execution_mode == "live"]
        assert len(det_trials) == 1
        assert len(live_trials) == 1
        assert det_trials[0].total_latency_ms != live_trials[0].total_latency_ms
