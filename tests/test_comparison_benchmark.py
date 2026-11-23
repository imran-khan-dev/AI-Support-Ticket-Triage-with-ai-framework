"""Tests for the retrieval benchmark runner (app/comparison/retrieval_benchmark.py).

The custom adapter is exercised for real (in-memory BM25, no external services).
The framework adapter path is validated for argument handling and error
propagation without requiring llama_index or PostgreSQL.
"""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter, RetrievalRequest
from app.comparison.retrieval_benchmark import (
    RetrievalBenchmarkResult,
    RetrievalQueryResult,
    run_retrieval_benchmark,
)
from app.comparison.contracts import Implementation


class _MockChunk:
    def __init__(self, logical_id, chunk_index, content, metadata=None):
        self.logical_id = logical_id
        self.chunk_index = chunk_index
        self.content = content
        self.metadata = metadata or {}


class _MockDoc:
    def __init__(self, id, filename, content_type, content, chunks):
        self.id = id
        self.filename = filename
        self.content_type = content_type
        self.content = content
        self.chunks = chunks


class _MockQuery:
    def __init__(self, query_id, query_text, relevance_judgments):
        self.query_id = query_id
        self.query_text = query_text
        self.relevance_judgments = relevance_judgments


def _make_dataset():
    docs = [
        _MockDoc("D1", "password_reset.txt", "text/plain",
                 "To reset your password, go to account settings and click Forgot password.",
                 [_MockChunk("C1", 0, "To reset your password, go to account settings.", {}),
                  _MockChunk("C2", 1, "A reset link will be sent to your email address.", {})]),
        _MockDoc("D2", "billing_refunds.txt", "text/plain",
                 "Billing refunds are processed within 5 business days for eligible charges.",
                 [_MockChunk("C3", 0, "Billing refunds are processed within 5 business days.", {}),
                  _MockChunk("C4", 1, "Refunds are issued to the original payment method.", {})]),
    ]
    queries = [
        _MockQuery("q1", "How do I reset my password?", {"C1": 3, "C2": 2}),
        _MockQuery("q2", "billing refund processing time", {"C3": 3, "C4": 1}),
    ]
    return SimpleNamespace(
        name="test-dataset",
        tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
        documents=docs,
        queries=queries,
    )


class TestRunRetrievalBenchmarkCustom:
    def test_custom_benchmark_runs_and_returns_results(self):
        dataset = _make_dataset()
        result = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)

        assert isinstance(result, RetrievalBenchmarkResult)
        assert result.implementation == "custom"
        assert result.top_k == 5
        assert result.num_queries == 2
        assert result.status == "completed"
        assert result.error is None
        assert len(result.per_query) == 2

        for pq in result.per_query:
            assert pq.implementation == "custom"
            assert pq.query_id in ("q1", "q2")
            assert pq.query_text
            assert isinstance(pq.retrieved_chunk_ids, list)
            assert pq.status == "completed"
            assert pq.error is None
            assert 5 in pq.precision_at_k
            assert 5 in pq.recall_at_k
            assert 5 in pq.ndcg_at_k
            assert isinstance(pq.reciprocal_rank, float)

    def test_custom_benchmark_per_query_metrics(self):
        dataset = _make_dataset()
        result = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)

        by_id = {pq.query_id: pq for pq in result.per_query}
        q1 = by_id["q1"]
        q2 = by_id["q2"]

        assert q1.retrieved_chunk_ids, "q1 should retrieve at least one chunk"
        assert q1.precision_at_k[5] >= 0.0
        assert q1.recall_at_k[5] >= 0.0
        assert 0.0 <= q1.reciprocal_rank <= 1.0
        assert 0.0 <= q1.ndcg_at_k[5] <= 1.0
        assert q2.retrieved_chunk_ids
        assert q2.precision_at_k[5] >= 0.0

    def test_custom_benchmark_aggregate(self):
        dataset = _make_dataset()
        result = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)

        agg = result.aggregate
        assert agg["precision_at_k"][5] is not None
        assert agg["recall_at_k"][5] is not None
        assert agg["mrr"] is not None
        assert agg["ndcg_at_k"][5] is not None
        assert agg["num_completed"] == 2
        assert agg["num_failed"] == 0

    def test_custom_benchmark_top_k_enforced(self):
        dataset = _make_dataset()
        result = run_retrieval_benchmark(dataset, Implementation.custom, top_k=1)

        for pq in result.per_query:
            assert len(pq.retrieved_chunk_ids) <= 1

    def test_custom_benchmark_deterministic_repeat(self):
        dataset = _make_dataset()
        r1 = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)
        r2 = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)

        assert [pq.retrieved_chunk_ids for pq in r1.per_query] == \
            [pq.retrieved_chunk_ids for pq in r2.per_query]
        assert r1.aggregate == r2.aggregate

    def test_unsupported_implementation_raises(self):
        dataset = _make_dataset()
        with pytest.raises(ValueError):
            run_retrieval_benchmark(dataset, "not-an-implementation", top_k=5)

    def test_empty_dataset(self):
        empty = SimpleNamespace(name="empty", tenant_id="t1", documents=[], queries=[])
        result = run_retrieval_benchmark(empty, Implementation.custom, top_k=5)

        assert result.num_queries == 0
        assert result.status == "failed"
        assert "no queries" in result.error

    def test_result_is_serializable(self):
        dataset = _make_dataset()
        result = run_retrieval_benchmark(dataset, Implementation.custom, top_k=5)
        data = asdict(result)
        assert data["implementation"] == "custom"
        assert data["num_queries"] == 2
        assert len(data["per_query"]) == 2
        assert "aggregate" in data


class TestRunRetrievalBenchmarkFramework:
    def test_framework_requires_database(self):
        dataset = _make_dataset()
        with pytest.raises(ValueError, match="requires a database handle"):
            run_retrieval_benchmark(
                dataset, Implementation.framework, top_k=5, db=None,
            )

    def test_framework_adapter_error_is_recorded(self):
        dataset = _make_dataset()
        db = MagicMock()

        with patch("app.knowledge.repository.create_chunks",
                   side_effect=RuntimeError("db down")):
            result = run_retrieval_benchmark(
                dataset, Implementation.framework, top_k=5, db=db,
                tenant_id=dataset.tenant_id,
            )

        assert result.implementation == "framework"
        assert result.status in ("partial", "failed")
        failed = [pq for pq in result.per_query if pq.status == "failed"]
        assert len(failed) >= 1
        for pq in failed:
            assert pq.error is not None
            assert pq.retrieved_chunk_ids == []
            assert pq.precision_at_k[5] == 0.0
            assert pq.recall_at_k[5] == 0.0
            assert pq.reciprocal_rank == 0.0
            assert pq.ndcg_at_k[5] == 0.0
