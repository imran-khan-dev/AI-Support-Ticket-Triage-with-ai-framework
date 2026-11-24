"""Tests for the agent comparison benchmark runner (app/comparison/agent_benchmark.py).

Executes scenarios against both implementations, verifies per-scenario
comparison results, aggregate metrics, and report generation (JSON + markdown).
No external LLM or paid provider is used.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.comparison.agent_adapters import AgentScenarioResult
from app.comparison.agent_benchmark import (
    AgentComparisonRunner,
    ComparisonOutcome,
    ScenarioComparison,
    write_reports,
)
from app.comparison.agent_scenarios import (
    AgentEvaluationDataset,
    AgentScenarioSpec,
    ScriptedStep,
    build_scenarios,
    load_agent_dataset,
)


# ---------------------------------------------------------------------------
# ScenarioComparison
# ---------------------------------------------------------------------------

def test_scenario_comparison_both_supported():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    comp = ScenarioComparison(spec, custom=custom, framework=framework)
    assert comp.scenario_id == "s1"
    assert comp.category == "cat"
    assert comp.both_supported is True
    assert comp.outcome_match is True
    assert comp.differences == []


def test_scenario_comparison_differences():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=3, provider_calls=3, outcome_category="success",
        final_answer="custom answer",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="success",
        tool_calls_count=5, provider_calls=2, outcome_category="success",
        final_answer="fw answer",
    )
    comp = ScenarioComparison(spec, custom=custom, framework=framework)
    assert comp.outcome_match is True
    diffs = comp.differences
    assert any("tool_calls" in d for d in diffs)
    assert any("provider_calls" in d for d in diffs)
    assert any("final_answer" in d for d in diffs)


def test_scenario_comparison_outcome_mismatch():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=1, outcome_category="success",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="max_iterations_reached",
        tool_calls_count=1, provider_calls=1, outcome_category="max_iterations_reached",
    )
    comp = ScenarioComparison(spec, custom=custom, framework=framework)
    assert comp.outcome_match is False


def test_scenario_comparison_unsupported():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=1, outcome_category="success",
    )
    comp = ScenarioComparison(spec, custom=custom, framework=None)
    assert comp.both_supported is False
    assert comp.framework_supported is False
    assert comp.outcome_match is False
    assert any("unsupported" in d for d in comp.differences)


def test_scenario_comparison_to_dict():
    spec = AgentScenarioSpec(
        scenario_id="s1", category="cat", user_request="hi",
        expected_outcome="success", expected_tool_names=["search_knowledge"],
        expected_tool_call_count=1, safety_constraints=["cross_tenant_access_prevention"],
        supported_implementations=["custom", "framework"],
    )
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    comp = ScenarioComparison(spec, custom=custom, framework=None)
    d = comp.to_dict()
    assert d["scenario_id"] == "s1"
    assert d["category"] == "cat"
    assert d["expected_outcome"] == "success"
    assert d["expected_tool_names"] == ["search_knowledge"]
    assert d["expected_tool_call_count"] == 1
    assert d["both_supported"] is False
    assert d["custom"] is not None
    assert d["framework"] is None


# ---------------------------------------------------------------------------
# ComparisonOutcome
# ---------------------------------------------------------------------------

def test_comparison_outcome_add_and_aggregates():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds)
    comp = ScenarioComparison(spec, custom=custom, framework=framework)
    outcome.add(comp)
    assert len(outcome.scenarios) == 1
    assert outcome.custom_scenarios == 1
    assert outcome.framework_scenarios == 1
    agg = outcome.custom_aggregate
    assert agg["scenario_success_rate"] == 1.0


def test_comparison_outcome_to_dict():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds, experiment_id="exp-123")
    comp = ScenarioComparison(spec, custom=custom, framework=None)
    outcome.add(comp)
    d = outcome.to_dict()
    assert d["experiment_id"] == "exp-123"
    assert d["dataset_version"] == "1.0.0"
    assert d["total_scenarios"] == 1
    assert d["custom_scenarios"] == 1
    assert d["framework_scenarios"] == 0
    assert "aggregate" in d
    assert "custom" in d["aggregate"]
    assert "framework" in d["aggregate"]
    assert len(d["scenarios"]) == 1


# ---------------------------------------------------------------------------
# AgentComparisonRunner
# ---------------------------------------------------------------------------

class TestAgentComparisonRunner:
    @staticmethod
    def test_run_all_executes_both_implementations():
        runner = AgentComparisonRunner(load_agent_dataset())
        outcome = runner.run_all()
        assert outcome.total_scenarios == 14
        assert outcome.custom_scenarios == 14
        # Cancellation is custom-only, so framework gets 13
        assert outcome.framework_scenarios == 13

    @staticmethod
    def test_run_all_every_scenario_has_custom_result():
        runner = AgentComparisonRunner(load_agent_dataset())
        outcome = runner.run_all()
        for comp in outcome.scenarios:
            assert comp.custom is not None
            assert comp.custom.scenario_id == comp.scenario_id

    @staticmethod
    def test_run_all_framework_skips_unsupported():
        runner = AgentComparisonRunner(load_agent_dataset())
        outcome = runner.run_all()
        for comp in outcome.scenarios:
            if comp.spec.supported_implementations == ["custom"]:
                assert comp.framework is None
            else:
                assert comp.framework is not None

    @staticmethod
    def test_run_all_no_exceptions_propagate():
        """Even if a scenario fails, run_all returns a ComparisonOutcome with
        a 'failed' result rather than raising."""
        runner = AgentComparisonRunner(load_agent_dataset())
        outcome = runner.run_all()
        # All results should be present, even if some are failed
        for comp in outcome.scenarios:
            assert comp.custom is not None
            assert comp.custom.status in {
                "success", "provider_failure", "max_iterations_reached",
                "tool_failure", "failed", "cancelled",
            }


# ---------------------------------------------------------------------------
# write_reports
# ---------------------------------------------------------------------------

def test_write_reports_creates_files(tmp_path):
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds, experiment_id="exp-1")
    comp = ScenarioComparison(spec, custom=custom, framework=framework)
    outcome.add(comp)

    reports_dir = tmp_path / "reports"
    scenario_dir = reports_dir / "scenarios"
    aggregate_path = reports_dir / "agg.json"
    markdown_path = reports_dir / "report.md"

    written = write_reports(outcome)

    # verify per-scenario JSON
    scenario_path = scenario_dir / "s1.json" if False else None
    # Instead, check the written paths dict
    assert "s1" in written
    assert written["s1"].is_file()
    data = json.loads(written["s1"].read_text(encoding="utf-8"))
    assert data["scenario_id"] == "s1"
    assert data["custom"]["status"] == "success"
    assert data["framework"]["status"] == "success"

    # verify aggregate JSON
    assert "aggregate" in written
    assert written["aggregate"].is_file()
    agg = json.loads(written["aggregate"].read_text(encoding="utf-8"))
    assert agg["total_scenarios"] == 1
    assert "custom" in agg["aggregate"]
    assert "framework" in agg["aggregate"]

    # verify markdown
    assert "markdown" in written
    assert written["markdown"].is_file()
    md = written["markdown"].read_text(encoding="utf-8")
    assert "# F4.4" in md
    assert "s1" in md


# ---------------------------------------------------------------------------
# Focused tests: scenario coverage and test results metadata
# ---------------------------------------------------------------------------

def test_scenario_coverage_breakdown():
    spec = AgentScenarioSpec(
        scenario_id="s1", category="cat", user_request="hi",
        supported_implementations=["custom", "framework"],
    )
    spec_custom_only = AgentScenarioSpec(
        scenario_id="s2", category="cat", user_request="hi",
        supported_implementations=["custom"],
    )
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    custom_failed = AgentScenarioResult(
        scenario_id="s2", implementation="custom", status="failed",
        tool_calls_count=0, provider_calls=1, outcome_category="failed",
    )
    framework = AgentScenarioResult(
        scenario_id="s1", implementation="framework", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec, spec_custom_only])
    outcome = ComparisonOutcome(ds)
    comp1 = ScenarioComparison(spec, custom=custom, framework=framework)
    comp2 = ScenarioComparison(spec_custom_only, custom=custom_failed, framework=None)
    outcome.add(comp1)
    outcome.add(comp2)

    coverage = outcome.scenario_coverage
    assert "custom" in coverage
    assert "framework" in coverage
    # Custom: 2 total, 2 supported, 0 unsupported, 2 executed, 1 passed, 1 failed
    assert coverage["custom"]["total"] == 2
    assert coverage["custom"]["supported"] == 2
    assert coverage["custom"]["unsupported"] == 0
    assert coverage["custom"]["executed"] == 2
    assert coverage["custom"]["passed"] == 1
    assert coverage["custom"]["failed"] == 1
    # Framework: 2 total, 1 supported, 1 unsupported, 1 executed, 1 passed
    assert coverage["framework"]["total"] == 2
    assert coverage["framework"]["supported"] == 1
    assert coverage["framework"]["unsupported"] == 1
    assert coverage["framework"]["executed"] == 1
    assert coverage["framework"]["passed"] == 1


def test_scenario_coverage_in_to_dict():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds)
    comp = ScenarioComparison(spec, custom=custom, framework=None)
    outcome.add(comp)
    d = outcome.to_dict()
    assert "scenario_coverage" in d
    assert "custom" in d["scenario_coverage"]
    assert "framework" in d["scenario_coverage"]


def test_test_results_metadata_defaults_none():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds)
    assert outcome.test_results_command is None
    assert outcome.test_results_recorded_at is None


def test_test_results_metadata_settable():
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds)
    outcome.test_results_command = "pytest tests/"
    outcome.test_results_recorded_at = "2026-11-23T17:38:00+00:00"
    d = outcome.to_dict()
    assert d["test_results_command"] == "pytest tests/"
    assert d["test_results_recorded_at"] == "2026-11-23T17:38:00+00:00"


def test_write_reports_sets_test_results_metadata(tmp_path):
    spec = AgentScenarioSpec(scenario_id="s1", category="cat", user_request="hi")
    custom = AgentScenarioResult(
        scenario_id="s1", implementation="custom", status="success",
        tool_calls_count=1, provider_calls=2, outcome_category="success",
    )
    ds = AgentEvaluationDataset(scenarios=[spec])
    outcome = ComparisonOutcome(ds)
    comp = ScenarioComparison(spec, custom=custom, framework=None)
    outcome.add(comp)

    write_reports(outcome)

    assert outcome.test_results_command == "pytest tests/test_comparison_agent_*.py"
    assert outcome.test_results_recorded_at is not None
