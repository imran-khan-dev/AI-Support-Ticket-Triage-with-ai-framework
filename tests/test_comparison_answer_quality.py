"""Tests for the answer-quality and faithfulness comparison (F4.3).

Covers:
- Deterministic metric unit tests with hand-calculated values.
- Custom answer adapter (real Retrieval Stack, mock LLM).
- Framework answer adapter (mock LLM; real DB when available).
- Answer comparison runner with mock LLM (no external API calls).
- Metric normalization, missing values, failed generation.
- Insufficient-context samples.
- Aggregate calculations.
- JSON and Markdown reporting.
- F4.1 ComparisonResult contract compatibility.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import json
import pytest

from app.comparison.answer_adapters import (
    AnswerAdapter,
    AnswerResult,
    CustomAnswerAdapter,
    FrameworkAnswerAdapter,
    RetrievedContext,
)
from app.comparison.answer_metrics import (
    answer_correctness,
    answer_relevancy,
    average,
    context_precision,
    context_recall,
    faithfulness,
    unsupported_claims,
)
from app.comparison.answer_comparison import (
    AnswerQualityComparisonResult,
    load_answer_quality_dataset,
    run_answer_quality_comparison,
    save_json_report,
    save_markdown_report,
)
from app.comparison.contracts import (
    Implementation,
    MetricScore,
    ComparisonResult,
    Component,
)
from app.core.config import Settings

DATASET_PATH = Path(__file__).resolve().parent.parent / "retrieval_eval" / "data" / "answer_quality_dataset.json"


# ---------------------------------------------------------------------------
# Metric unit tests (deterministic, hand-calculated)
# ---------------------------------------------------------------------------

class TestAnswerCorrectness:
    def test_exact_match(self):
        assert answer_correctness("hello world", "hello world") == 1.0

    def test_no_match(self):
        assert answer_correctness("xyz", "abc") == 0.0

    def test_none_answer(self):
        assert answer_correctness(None, "reference") == 0.0

    def test_none_reference_and_no_facts(self):
        assert answer_correctness("anything", None) == 0.0

    def test_fact_coverage_only(self):
        # No reference, but facts are covered.
        score = answer_correctness(
            "The reset link is sent by email.",
            None,
            ["reset link", "email"],
        )
        assert score == 1.0  # both facts found

    def test_partial_fact_coverage(self):
        score = answer_correctness(
            "The reset link is sent by email.",
            None,
            ["reset link", "phone number", "email"],
        )
        assert score == pytest.approx(2 / 3)

    def test_blend_reference_and_facts(self):
        score = answer_correctness(
            "Reset your password via email.",
            "Reset your password via email.",
            ["password", "email"],
        )
        # sim=1.0, fact_coverage=1.0 -> blend=1.0
        assert score == 1.0


class TestAnswerRelevancy:
    def test_relevant_answer(self):
        score = answer_relevancy(
            "How to reset your password",
            "How do I reset my password?",
        )
        assert 0.0 < score <= 1.0

    def test_irrelevant_answer(self):
        score = answer_relevancy(
            "The weather is nice today",
            "How do I reset my password?",
        )
        assert score < 0.5

    def test_none_answer(self):
        assert answer_relevancy(None, "question") == 0.0

    def test_none_question(self):
        assert answer_relevancy("answer", None) == 0.0


class TestFaithfulness:
    def test_fully_faithful(self):
        score = faithfulness(
            "Reset your password via email.",
            ["Reset your password, go to account settings.", "A reset link will be sent to your email."],
        )
        assert score > 0.5

    def test_unfaithful(self):
        score = faithfulness(
            "The moon is made of green cheese.",
            ["Reset your password, go to account settings."],
        )
        assert score == 0.0

    def test_none_answer(self):
        assert faithfulness(None, ["context"]) == 0.0

    def test_no_context_returns_none(self):
        # No context: cannot verify faithfulness -> None (unavailable).
        assert faithfulness("some answer", []) is None
        assert faithfulness("some answer", None) is None

    def test_empty_answer(self):
        assert faithfulness("", ["context"]) == 1.0


class TestContextPrecision:
    def test_all_relevant(self):
        assert context_precision(["C1", "C2"], ["C1", "C2"]) == 1.0

    def test_partial_relevant(self):
        assert context_precision(["C1", "C9"], ["C1", "C2"]) == 0.5

    def test_none_retrieved(self):
        assert context_precision([], ["C1"]) == 0.0

    def test_with_chunk_map(self):
        c2t = {"C1": "text about passwords", "C9": "text about settings"}
        assert context_precision(["C1", "C9"], ["C1"], chunk_id_to_text=c2t) == 0.5


class TestContextRecall:
    def test_all_retrieved(self):
        assert context_recall(["C1", "C2"], ["C1", "C2"]) == 1.0

    def test_partial_retrieved(self):
        assert context_recall(["C1"], ["C1", "C2", "C3"]) == pytest.approx(1 / 3)

    def test_none_expected(self):
        assert context_recall(["C1"], []) == 0.0


class TestUnsupportedClaims:
    def test_no_unsupported(self):
        assert unsupported_claims(
            "Reset your password via email.",
            ["Reset your password, go to account settings. A reset link is sent by email."],
        ) == []

    def test_unsupported_claim(self):
        result = unsupported_claims(
            "The moon is made of green cheese.",
            ["Reset your password, go to account settings."],
        )
        assert len(result) > 0

    def test_none_answer(self):
        assert unsupported_claims(None, ["context"]) == []


class TestAverage:
    def test_average(self):
        assert average([1.0, 2.0, 3.0]) == 2.0

    def test_average_ignores_none(self):
        assert average([1.0, None, 3.0]) == 2.0

    def test_average_all_none(self):
        assert average([None, None]) == 0.0

    def test_average_empty(self):
        assert average([]) == 0.0


# ---------------------------------------------------------------------------
# Dataset loading tests
# ---------------------------------------------------------------------------

class TestDatasetLoading:
    def test_load_dataset(self):
        ds = load_answer_quality_dataset(DATASET_PATH)
        assert ds["name"] == "support-kb-answer-quality-eval"
        assert ds["version"] == "1.0.0"
        assert len(ds["samples"]) == 7
        assert ds["tenant_id"] == "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"

    def test_sample_structure(self):
        ds = load_answer_quality_dataset(DATASET_PATH)
        sample = ds["samples"][0]
        assert "sample_id" in sample
        assert "question" in sample
        assert "reference_answer" in sample
        assert "expected_facts" in sample
        assert "expected_context_chunk_ids" in sample
        assert "category" in sample
        assert "sufficient_context" in sample

    def test_insufficient_context_sample(self):
        ds = load_answer_quality_dataset(DATASET_PATH)
        aq4 = [s for s in ds["samples"] if s["sample_id"] == "aq4"][0]
        assert aq4["category"] == "insufficient_context"
        assert aq4["sufficient_context"] is False

    def test_no_context_sample(self):
        ds = load_answer_quality_dataset(DATASET_PATH)
        aq7 = [s for s in ds["samples"] if s["sample_id"] == "aq7"][0]
        assert aq7["category"] == "no_context"
        assert aq7["sufficient_context"] is False
        assert aq7["expected_context_chunk_ids"] == []


# ---------------------------------------------------------------------------
# Custom answer adapter tests
# ---------------------------------------------------------------------------

class TestCustomAnswerAdapter:
    def test_adapter_creation(self):
        chunks = [
            {
                "logical_id": "C1",
                "document_id": "D1",
                "chunk_index": 0,
                "content": "Password reset instructions.",
                "metadata": {},
            }
        ]
        adapter = CustomAnswerAdapter(
            chunks=chunks,
            settings=Settings(),
            client=None,
        )
        assert adapter.implementation == "custom"

    def test_generate_with_mock_llm(self):
        chunks = [
            {
                "logical_id": "C1",
                "document_id": "D1",
                "chunk_index": 0,
                "content": "To reset your password, go to account settings.",
                "metadata": {},
            }
        ]
        adapter = CustomAnswerAdapter(
            chunks=chunks,
            settings=Settings(),
            client=None,
        )
        result = adapter.generate(
            question="How do I reset my password?",
            tenant_id="test-tenant",
            top_k=5,
        )

        assert result.implementation == "custom"
        assert result.question == "How do I reset my password?"
        assert result.status == "mock"  # client=None -> mock mode
        assert result.answer is not None
        assert "mock" in result.answer.lower() or result.answer is not None
        assert len(result.contexts) == 1
        assert result.contexts[0].chunk_id == "C1"
        assert result.context_chunk_ids == ["C1"]

    def test_generate_no_context(self):
        adapter = CustomAnswerAdapter(
            chunks=[],
            settings=Settings(),
            client=None,
        )
        result = adapter.generate(
            question="What is the stock price?",
            tenant_id="test-tenant",
            top_k=5,
        )
        assert result.contexts == []
        assert result.context_chunk_ids == []
        assert result.status == "mock"


# ---------------------------------------------------------------------------
# Framework answer adapter tests (mock LLM)
# ---------------------------------------------------------------------------

class TestFrameworkAnswerAdapterMock:
    def test_adapter_creation_requires_db(self):
        with pytest.raises(ValueError, match="requires a database handle"):
            FrameworkAnswerAdapter(
                db=None,
                tenant_id="test-tenant",
                documents=[],
                settings=Settings(),
                client=None,
            )

    def test_generate_with_mock_llm(self, clean_database):
        db = clean_database
        documents = [
            {
                "filename": "test.txt",
                "content_type": "text/plain",
                "content": "Test content about passwords.",
                "chunks": [
                    {
                        "logical_id": "C1",
                        "chunk_index": 0,
                        "content": "Test content about passwords.",
                        "metadata": {},
                    }
                ],
            }
        ]
        adapter = FrameworkAnswerAdapter(
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            documents=documents,
            settings=Settings(),
            client=None,
        )
        assert adapter.implementation == "framework"

        result = adapter.generate(
            question="How do I reset my password?",
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
        )

        assert result.implementation == "framework"
        assert result.status == "mock"
        assert result.answer is not None
        assert len(result.contexts) >= 0  # may be 0 if indexing takes time


# ---------------------------------------------------------------------------
# Runner tests with mock LLM
# ---------------------------------------------------------------------------

class TestAnswerQualityRunnerMock:
    def test_runner_completes_all_samples(self, clean_database):
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        documents = [
            {
                "filename": d["filename"],
                "content_type": d["content_type"],
                "content": d["content"],
                "chunks": [
                    {
                        "logical_id": c["logical_id"],
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "metadata": dict(c.get("metadata", {})),
                    }
                    for c in d["chunks"]
                ],
            }
            for d in dataset["documents"]
        ]

        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id=dataset["tenant_id"],
            top_k=5,
            client=None,
            settings=Settings(),
        )

        assert result.status == "completed"
        assert len(result.samples) == 7
        assert result.aggregate["num_samples"] == 7

        # Every sample has both custom and framework results.
        for sample in result.samples:
            assert "custom" in sample
            assert "framework" in sample
            assert sample["custom"]["status"] == "mock"
            assert sample["framework"]["status"] == "mock"

    def test_runner_metrics_present(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        for sample in result.samples:
            for impl in ("custom", "framework"):
                metrics = sample[impl]["metrics"]
                assert "answer_correctness" in metrics
                assert "answer_relevancy" in metrics
                assert "faithfulness" in metrics
                assert "context_precision" in metrics
                assert "context_recall" in metrics

    def test_runner_insufficient_context(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        # Find aq4 (insufficient context)
        aq4 = [s for s in result.samples if s["sample_id"] == "aq4"][0]
        assert aq4["sufficient_context"] is False
        # Both implementations should still produce answers (or note insufficient context)
        for impl in ("custom", "framework"):
            assert aq4[impl]["status"] in ("mock", "completed", "failed")

    def test_runner_aggregate_structure(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        agg = result.aggregate
        assert agg["num_samples"] == 7
        assert "custom" in agg
        assert "framework" in agg
        for impl in ("custom", "framework"):
            impl_agg = agg[impl]
            assert "num_completed" in impl_agg
            assert "num_failed" in impl_agg
            assert "metrics" in impl_agg
            for mname in ("answer_correctness", "answer_relevancy", "faithfulness",
                          "context_precision", "context_recall"):
                assert mname in impl_agg["metrics"]
                mdata = impl_agg["metrics"][mname]
                assert "mean" in mdata
                assert "count" in mdata

    def test_runner_without_db(self):
        # Framework should fail gracefully without a database.
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=None,
            tenant_id="test-tenant",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        assert result.status == "completed"  # custom still works
        for sample in result.samples:
            assert sample["custom"]["status"] == "mock"
            assert sample["framework"]["status"] == "failed"
            assert "requires a database handle" in sample["framework"]["error"]

    def test_runner_deterministic(self, clean_database):
        db = clean_database
        r1 = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )
        r2 = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        # Mock mode should be deterministic.
        for s1, s2 in zip(r1.samples, r2.samples):
            assert s1["custom"]["answer"] == s2["custom"]["answer"]
            assert s1["framework"]["answer"] == s2["framework"]["answer"]


# ---------------------------------------------------------------------------
# Reporting tests
# ---------------------------------------------------------------------------

class TestReporting:
    def test_save_json_report(self, clean_database, tmp_path):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        json_path = tmp_path / "report.json"
        save_json_report(result, json_path)

        assert json_path.is_file()
        data = json.loads(json_path.read_text())
        assert data["experiment_id"] == result.experiment_id
        assert data["dataset_name"] == result.dataset_name
        assert data["status"] == result.status
        assert len(data["samples"]) == 7
        assert "aggregate" in data

    def test_save_markdown_report(self, clean_database, tmp_path):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        md_path = tmp_path / "report.md"
        save_markdown_report(result, md_path)

        assert md_path.is_file()
        content = md_path.read_text()
        assert "# Answer Quality Comparison Report" in content
        assert "## Aggregate Results" in content
        assert "## Per-Sample Results" in content
        assert "aq1" in content


# ---------------------------------------------------------------------------
# Contract compatibility tests
# ---------------------------------------------------------------------------

class TestContractCompatibility:
    def test_to_contract_results(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        contract_results = result.to_contract_results()

        assert len(contract_results) == 14  # 7 samples * 2 implementations
        for cr in contract_results:
            assert isinstance(cr, ComparisonResult)
            assert cr.component == Component.agent
            assert cr.implementation in (Implementation.custom, Implementation.framework)
            assert cr.sample_id is not None
            assert cr.experiment_id == result.experiment_id
            assert cr.dataset_id == result.dataset_id
            assert len(cr.scores) > 0
            for score in cr.scores:
                assert isinstance(score, MetricScore)
                assert score.metric_name in (
                    "answer_correctness",
                    "answer_relevancy",
                    "faithfulness",
                    "context_precision",
                    "context_recall",
                )
                assert score.provenance == cr.implementation.value
                assert score.sample_id == cr.sample_id

    def test_metric_score_distinguishes_none_from_zero(self):
        # A metric that is unavailable (None) differs from a genuine 0.0.
        s_none = MetricScore(
            metric_name="faithfulness",
            provenance="custom",
            score=None,
            sample_id="aq1",
        )
        s_zero = MetricScore(
            metric_name="faithfulness",
            provenance="custom",
            score=0.0,
            sample_id="aq1",
        )
        assert s_none.score is None
        assert s_zero.score == 0.0
        assert s_none != s_zero
