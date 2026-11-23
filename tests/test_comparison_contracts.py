import pytest

from app.comparison.contracts import (
    ComparisonDataset,
    ComparisonResult,
    ComparisonRunConfig,
    ComparisonSample,
    Implementation,
    Component,
    RetrievalSample,
    AnswerSample,
    AgentScenario,
    MetricScore,
)
from app.comparison.adapters import NoOpAdapter


def test_retrieval_sample_validation():
    s = RetrievalSample(sample_id="r1", query="q")
    assert s.sample_id == "r1"


def test_missing_query():
    with pytest.raises(Exception):
        RetrievalSample(sample_id="r1", query="")


def test_dataset_unique_ids():
    with pytest.raises(Exception):
        ComparisonDataset(
            name="d",
            samples=ComparisonSample(
                retrieval=[RetrievalSample(sample_id="dup", query="q"), RetrievalSample(sample_id="dup", query="q2")]
            ),
        )


def test_result_serialization():
    r = ComparisonResult(
        experiment_id="e1",
        implementation=Implementation.framework,
        component=Component.retrieval,
        dataset_id="d1",
        status="completed",
        scores=[MetricScore(metric_name="exact_match", provenance="ragas", score=1.0, sample_id="s1")],
    )
    d = r.model_dump()
    assert d["scores"][0]["score"] == 1.0


def test_missing_vs_zero_score():
    r = ComparisonResult(
        experiment_id="e1",
        implementation=Implementation.custom,
        component=Component.retrieval,
        dataset_id="d1",
        status="completed",
        scores=[MetricScore(metric_name="m", provenance="custom", score=None)],
    )
    assert r.scores[0].score is None


def test_adapter_noop():
    adapter = NoOpAdapter("custom")
    assert adapter.implementation == "custom"
    cfg = ComparisonRunConfig(implementation=Implementation.custom, component=Component.retrieval, dataset_id="d1")
    res = adapter.run(cfg, ComparisonDataset(name="d", samples=ComparisonSample(retrieval=[RetrievalSample(sample_id="r1", query="q")])))
    assert res.status == "completed"