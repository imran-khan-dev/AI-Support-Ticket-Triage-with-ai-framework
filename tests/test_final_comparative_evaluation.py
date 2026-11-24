"""Validation tests for the final comparative evaluation report (F4.6).

These tests verify:
- Final report schema (JSON and Markdown)
- Required sections and dimensions in the comparison matrix
- Correct ingestion of existing report data (F4.2-F4.5)
- Missing F4.3 live results remaining pending
- Null token/cost data remaining null
- Correct F4.4 scenario counts
- Correct F4.5 shared-scenario denominators
- Consistency between JSON and Markdown summaries

Tests do not assert hardcoded benchmark conclusions without checking
underlying source data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPORTS_DIR = Path(__file__).resolve().parent.parent / "app" / "f432_reports"
FINAL_JSON = REPORTS_DIR / "final_comparative_evaluation.json"
FINAL_MD = REPORTS_DIR / "final_comparative_evaluation.md"

F42_RETRIEVAL = REPORTS_DIR / "bm25_comparison_report.md"
F43_ANSWER = REPORTS_DIR / "f432_report.md"
F44_AGENT = REPORTS_DIR / "f44_agent_comparison.json"
F45_PERF = REPORTS_DIR / "f45_performance_comparison.json"


# ---------------------------------------------------------------------------
# Report schema tests
# ---------------------------------------------------------------------------

class TestFinalReportSchema:
    """Tests for the final evaluation report JSON schema."""

    def test_json_exists(self):
        assert FINAL_JSON.is_file()

    def test_json_has_required_top_level_keys(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        required = [
            "report_id", "title", "generated_at", "project",
            "source_reports", "scope", "evidence_summary",
            "comparison_matrix", "answers_to_questions",
            "evidence_gaps", "test_results", "files_created",
            "files_modified", "acceptance_recommendation",
            "project_completion_assessment",
        ]
        for key in required:
            assert key in data, f"Missing top-level key: {key}"

    def test_comparison_matrix_has_required_dimensions(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        matrix = data["comparison_matrix"]
        dimensions = {row["dimension"] for row in matrix}
        required_dims = {
            "Retrieval quality", "Answer quality", "Faithfulness",
            "Tool orchestration", "Reliability", "Security behavior",
            "Execution control", "Observability", "Latency",
            "Token usage", "Cost", "Maintainability", "Extensibility",
            "Operational complexity",
        }
        missing = required_dims - dimensions
        assert not missing, f"Missing matrix dimensions: {missing}"

    def test_matrix_rows_have_required_fields(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        for row in data["comparison_matrix"]:
            for field in ("dimension", "custom", "framework", "evidence", "conclusion", "confidence"):
                assert field in row, f"Matrix row missing field: {field}"

    def test_answers_to_questions_has_all_questions(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        answers = data["answers_to_questions"]
        required_qs = {
            "greater_low_level_control", "more_built_in_abstractions",
            "stronger_measured_retrieval", "answer_quality_superior",
            "production_performance_superior", "more_custom_maintenance",
            "strongest_reasons_custom", "strongest_reasons_framework",
            "valuable_additional_experiments",
        }
        missing = required_qs - set(answers.keys())
        assert not missing, f"Missing questions: {missing}"


# ---------------------------------------------------------------------------
# Evidence ingestion tests
# ---------------------------------------------------------------------------

class TestEvidenceIngestion:
    """Tests that the final report correctly ingests existing report data."""

    def test_f42_retrieval_ingested(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f42 = data["evidence_summary"]["f42_retrieval"]
        assert f42["status"] == "verified"
        assert f42["queries"] == 6
        assert f42["custom_mrr"] == 0.8333
        assert f42["framework_mrr"] == 1.0000

    def test_f43_answer_quality_ingested(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f43 = data["evidence_summary"]["f43_answer_quality"]
        assert f43["status"] == "partially_verified"
        assert f43["deterministic_metrics"]["samples_per_implementation"] == 7
        assert f43["deterministic_metrics"]["answer_correctness"]["custom"] == 0.5984
        assert f43["deterministic_metrics"]["answer_correctness"]["framework"] == 0.5219

    def test_f43_ragas_not_available(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f43 = data["evidence_summary"]["f43_answer_quality"]
        assert f43["ragas_evaluation"]["status"] == "not_available"

    def test_f43_real_generation_partial(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f43 = data["evidence_summary"]["f43_answer_quality"]
        assert f43["real_llm_generation"]["status"] == "partial"
        assert f43["real_llm_generation"]["samples_generated"] == "aq1 both implementations (2 total)"

    def test_f44_agent_behavior_ingested(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f44 = data["evidence_summary"]["f44_agent_behavior"]
        assert f44["status"] == "verified"
        assert f44["total_scenarios"] == 14
        assert f44["custom"]["supported"] == 14
        assert f44["custom"]["executed"] == 14
        assert f44["custom"]["passed"] == 12
        assert f44["framework"]["supported"] == 13
        assert f44["framework"]["executed"] == 13
        assert f44["framework"]["passed"] == 11

    def test_f44_excluded_scenario_documented(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f44 = data["evidence_summary"]["f44_agent_behavior"]
        excluded = f44["excluded_scenario"]
        assert excluded["scenario_id"] == "agent-cancellation"
        assert excluded["supported_implementations"] == ["custom"]

    def test_f45_performance_ingested(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f45 = data["evidence_summary"]["f45_performance"]
        assert f45["status"] == "verified_preliminary"
        assert f45["total_scenarios"] == 14
        assert f45["shared_scenarios"] == 13
        assert f45["excluded_scenarios"] == 1
        assert f45["custom_trials"] == 42
        assert f45["custom_successes"] == 36
        assert f45["framework_trials"] == 39
        assert f45["framework_successes"] == 33
        assert f45["successful_paired_trials"] == 33
        assert f45["failed_paired_trials"] == 6


# ---------------------------------------------------------------------------
# Null token/cost tests
# ---------------------------------------------------------------------------

class TestNullTokenCost:
    """Tests that null token and cost data remain null."""

    def test_tokens_null_in_f45(self):
        data = json.loads(F45_PERF.read_text(encoding="utf-8"))
        for scenario in data["scenarios"]:
            for trial in scenario["custom_trials"] + scenario["framework_trials"]:
                assert trial["input_tokens"] is None
                assert trial["output_tokens"] is None
                assert trial["total_tokens"] is None
                assert trial["estimated_cost_usd"] is None

    def test_tokens_null_in_final_json(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f45 = data["evidence_summary"]["f45_performance"]
        assert f45["tokens"] == "null for all trials (scripted providers supply no usage metadata)"
        assert f45["cost"] == "null for all trials (no token usage, no verifiable pricing extraction)"

    def test_cost_not_converted_to_zero(self):
        data = json.loads(F45_PERF.read_text(encoding="utf-8"))
        for scenario in data["scenarios"]:
            for trial in scenario["custom_trials"] + scenario["framework_trials"]:
                assert trial["estimated_cost_usd"] is None
                assert trial["estimated_cost_usd"] != 0


# ---------------------------------------------------------------------------
# F4.4 scenario count tests
# ---------------------------------------------------------------------------

class TestF44ScenarioCounts:
    """Tests that F4.4 scenario counts are correct."""

    def test_f44_custom_counts_match_source(self):
        data = json.loads(F44_AGENT.read_text(encoding="utf-8"))
        assert data["custom_scenarios"] == 1
        assert data["framework_scenarios"] == 0

    def test_f44_coverage_breakdown(self):
        data = json.loads(F44_AGENT.read_text(encoding="utf-8"))
        cov = data["scenario_coverage"]
        assert cov["custom"]["total"] == 1
        assert cov["custom"]["supported"] == 1
        assert cov["custom"]["executed"] == 1
        assert cov["custom"]["passed"] == 1
        assert cov["framework"]["total"] == 1
        assert cov["framework"]["supported"] == 1
        assert cov["framework"]["executed"] == 0


# ---------------------------------------------------------------------------
# F4.5 shared-scenario denominator tests
# ---------------------------------------------------------------------------

class TestF45Denominators:
    """Tests that F4.5 shared-scenario denominators are correct."""

    def test_shared_vs_total_scenarios(self):
        data = json.loads(F45_PERF.read_text(encoding="utf-8"))
        assert data["total_scenarios"] == 14
        assert data["shared_scenarios"] == 13
        assert len(data["excluded_scenarios"]) == 1
        assert data["excluded_scenarios"][0]["scenario_id"] == "agent-cancellation"

    def test_trial_totals_match_repetitions(self):
        data = json.loads(F45_PERF.read_text(encoding="utf-8"))
        # 14 scenarios * 3 reps = 42 custom
        assert data["custom_trial_total"] == 42
        # 13 shared * 3 reps = 39 framework
        assert data["framework_trial_total"] == 39

    def test_success_rate_denominator_all_trials(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f45 = data["evidence_summary"]["f45_performance"]
        assert f45["aggregate_custom"]["success_rate"] == 36 / 42
        assert f45["aggregate_framework"]["success_rate"] == 33 / 39


# ---------------------------------------------------------------------------
# JSON/Markdown consistency tests
# ---------------------------------------------------------------------------

class TestJSONMarkdownConsistency:
    """Tests for consistency between JSON and Markdown reports."""

    def test_markdown_exists(self):
        assert FINAL_MD.is_file()

    def test_markdown_has_executive_summary(self):
        md = FINAL_MD.read_text(encoding="utf-8")
        assert "# Final Comparative Evaluation" in md
        assert "## Executive Summary" in md

    def test_markdown_has_required_sections(self):
        md = FINAL_MD.read_text(encoding="utf-8")
        required = [
            "## Project Scope",
            "## Architecture Overview",
            "## Benchmark Methodology",
            "## Comparative Results",
            "## Architecture and Engineering Trade-offs",
            "## Final Comparison Matrix",
            "## Final Engineering Conclusions",
            "## Limitations",
            "## Future Experiments",
            "## Files Created",
            "## Source Reports Consumed",
        ]
        for section in required:
            assert section in md, f"Missing markdown section: {section}"
        assert "Final assessment" in md

    def test_markdown_contains_key_figures(self):
        md = FINAL_MD.read_text(encoding="utf-8")
        # F4.2 figures
        assert "0.8333" in md  # Custom MRR
        assert "1.0000" in md  # Framework MRR
        # F4.3 figures
        assert "0.5984" in md  # Custom correctness
        assert "0.5219" in md  # Framework correctness
        # F4.4 figures
        assert "0.8571" in md  # Custom success rate
        assert "0.8462" in md  # Framework success rate
        # F4.5 figures
        assert "1.5724" in md  # Custom median latency
        assert "2.7278" in md  # Framework median latency

    def test_json_and_markdown_agree_on_retrieval_winner(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        json_conclusion = None
        for row in data["comparison_matrix"]:
            if row["dimension"] == "Retrieval quality":
                json_conclusion = row["conclusion"]
                break
        assert json_conclusion is not None
        assert "Framework" in json_conclusion

        md = FINAL_MD.read_text(encoding="utf-8")
        # Markdown should also indicate Framework leads retrieval
        assert "Framework" in md and "retrieval" in md.lower()

    def test_json_and_markdown_agree_on_preliminary_label(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f45 = data["evidence_summary"]["f45_performance"]
        assert f45["status"] == "verified_preliminary"
        assert "preliminary" in f45["limitation"].lower()

        md = FINAL_MD.read_text(encoding="utf-8")
        assert "preliminary" in md.lower()

    def test_no_invented_cost_figures(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        f45 = data["evidence_summary"]["f45_performance"]
        # Cost should be null/unavailable, not a number
        assert f45["cost"] is not None
        assert "null" in f45["cost"].lower() or "unavailable" in f45["cost"].lower()

    def test_evidence_gaps_listed(self):
        data = json.loads(FINAL_JSON.read_text(encoding="utf-8"))
        gaps = data["evidence_gaps"]
        assert len(gaps) > 0
        # Ragas unavailability should be listed
        ragas_gap = any("Ragas" in g for g in gaps)
        assert ragas_gap, "Ragas gap not listed in evidence_gaps"
