"""Tests for the custom and framework retrieval adapters.

The custom adapter is exercised against the real Retrieval Stack (in-memory
BM25, no external services). The framework adapter requires llama_index and a
PostgreSQL database, which are not installed in this environment, so its
identifier-mapping and hit-normalization logic is tested with mocks instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.comparison.retrieval_adapters import (
    CustomBM25RetrievalAdapter,
    FrameworkBM25RetrievalAdapter,
    RetrievalHit,
    RetrievalRequest,
)


# ---------------------------------------------------------------------------
# Custom adapter (real Retrieval Stack, no external services)
# ---------------------------------------------------------------------------

class TestCustomBM25RetrievalAdapter:
    def test_search_returns_normalized_hits(self):
        chunks = [
            {
                "logical_id": "C1",
                "document_id": "D1",
                "chunk_index": 0,
                "content": "To reset your password, go to account settings.",
                "metadata": {"section": "account"},
            },
            {
                "logical_id": "C2",
                "document_id": "D1",
                "chunk_index": 1,
                "content": "A reset link will be sent to your email address.",
                "metadata": {"section": "email"},
            },
            {
                "logical_id": "C3",
                "document_id": "D2",
                "chunk_index": 0,
                "content": "Billing refunds are processed within 5 business days.",
                "metadata": {"section": "billing"},
            },
        ]
        adapter = CustomBM25RetrievalAdapter(chunks)
        assert adapter.implementation == "custom"

        hits = adapter.search(RetrievalRequest(query="reset password", top_k=2))

        assert len(hits) == 2
        assert all(isinstance(h, RetrievalHit) for h in hits)
        # Ranking order is preserved (highest score first).
        assert hits[0].score >= hits[1].score
        # Chunk and document identifiers are preserved.
        assert {h.chunk_id for h in hits} <= {"C1", "C2", "C3"}
        assert all(h.document_id in ("D1", "D2") for h in hits)
        # Content and score are populated.
        assert all(h.content for h in hits)
        assert all(h.score is not None for h in hits)

    def test_top_k_enforcement(self):
        chunks = [
            {"logical_id": f"C{i}", "document_id": "D1", "chunk_index": i - 1,
             "content": f"content {i}", "metadata": {}}
            for i in range(1, 6)
        ]
        adapter = CustomBM25RetrievalAdapter(chunks)

        hits = adapter.search(RetrievalRequest(query="content", top_k=3))
        assert len(hits) == 3

        hits = adapter.search(RetrievalRequest(query="content", top_k=10))
        assert len(hits) == 5

    def test_unknown_query_returns_empty(self):
        adapter = CustomBM25RetrievalAdapter([
            {"logical_id": "C1", "document_id": "D1", "chunk_index": 0,
             "content": "password reset instructions", "metadata": {}},
        ])
        hits = adapter.search(RetrievalRequest(query="zzznothingzzz", top_k=5))
        assert hits == []

    def test_preserves_source_identifiers(self):
        adapter = CustomBM25RetrievalAdapter([
            {"logical_id": "C1", "document_id": "D9", "chunk_index": 0,
             "content": "unique phrase here", "metadata": {"section": "x"}},
        ])
        hits = adapter.search(RetrievalRequest(query="unique phrase", top_k=1))
        assert len(hits) == 1
        assert hits[0].chunk_id == "C1"
        assert hits[0].document_id == "D9"


# ---------------------------------------------------------------------------
# Framework adapter (mocks: llama_index + PostgreSQL not installed)
# ---------------------------------------------------------------------------

class _FakeKnowledgeChunk:
    __slots__ = ("id", "document_id", "content")

    def __init__(self, id: str, document_id: str, content: str):
        self.id = id
        self.document_id = document_id
        self.content = content


class _FakeSearchResult:
    __slots__ = ("chunk", "score")

    def __init__(self, chunk, score: float):
        self.chunk = chunk
        self.score = score


class TestFrameworkBM25RetrievalAdapterMapping:
    """Identifier mapping and normalization logic, no real DB needed."""

    def _make_adapter(self, logical_to_actual, actual_to_logical, search_results):
        db = MagicMock()
        mock_conn = MagicMock()
        db.write_transaction.return_value.__enter__ = MagicMock(return_value=mock_conn)
        db.write_transaction.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("app.database.repository._canonical_uuid", return_value="tenant-canonical"),
            patch("app.knowledge.repository.create_document",
                  return_value=SimpleNamespace(id="doc-1")),
            patch("app.knowledge.repository.create_chunks",
                  return_value=[SimpleNamespace(id=aid) for aid in actual_to_logical.keys()]),
        ):
            adapter = FrameworkBM25RetrievalAdapter(
                db=db,
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                documents=[{"filename": "a.txt", "content_type": "text/plain",
                            "content": "x", "chunks": []}],
            )
            # Override the mapping built during construction.
            adapter._logical_to_actual = dict(logical_to_actual)
            adapter._actual_to_logical = dict(actual_to_logical)

        # Mock the lazily-created KnowledgeService.search.
        adapter._service = MagicMock()
        adapter._service.search.return_value = search_results

        return adapter

    def test_search_maps_actual_ids_back_to_logical_ids(self):
        logical_to_actual = {"C1": "actual-uuid-1", "C2": "actual-uuid-2"}
        actual_to_logical = {"actual-uuid-1": "C1", "actual-uuid-2": "C2"}
        results = [
            _FakeSearchResult(_FakeKnowledgeChunk("actual-uuid-1", "D1", "password reset"), 0.9),
            _FakeSearchResult(_FakeKnowledgeChunk("actual-uuid-2", "D1", "email link"), 0.5),
        ]
        adapter = self._make_adapter(logical_to_actual, actual_to_logical, results)

        hits = adapter.search(RetrievalRequest(query="reset password", top_k=5))

        assert len(hits) == 2
        assert hits[0].chunk_id == "C1"
        assert hits[0].document_id == "D1"
        assert hits[0].content == "password reset"
        assert hits[0].score == 0.9
        assert hits[1].chunk_id == "C2"
        assert hits[1].score == 0.5

    def test_search_skips_chunks_not_in_mapping(self):
        logical_to_actual = {"C1": "actual-uuid-1"}
        actual_to_logical = {"actual-uuid-1": "C1"}
        # One result whose chunk id is not in the mapping -> should be skipped.
        results = [
            _FakeSearchResult(_FakeKnowledgeChunk("actual-uuid-99", "D1", "orphan"), 0.8),
            _FakeSearchResult(_FakeKnowledgeChunk("actual-uuid-1", "D1", "password reset"), 0.9),
        ]
        adapter = self._make_adapter(logical_to_actual, actual_to_logical, results)

        hits = adapter.search(RetrievalRequest(query="reset", top_k=5))

        assert len(hits) == 1
        assert hits[0].chunk_id == "C1"

    def test_search_skips_results_without_chunk(self):
        logical_to_actual = {}
        actual_to_logical = {}
        # Result with no chunk at all.
        results = [SimpleNamespace(chunk=None, score=0.5)]
        adapter = self._make_adapter(logical_to_actual, actual_to_logical, results)

        hits = adapter.search(RetrievalRequest(query="reset", top_k=5))

        assert hits == []

    def test_search_preserves_ranking_order(self):
        logical_to_actual = {"C1": "a1", "C2": "a2", "C3": "a3"}
        actual_to_logical = {"a1": "C1", "a2": "C2", "a3": "C3"}
        results = [
            _FakeSearchResult(_FakeKnowledgeChunk("a1", "D1", "first"), 0.9),
            _FakeSearchResult(_FakeKnowledgeChunk("a2", "D1", "second"), 0.5),
            _FakeSearchResult(_FakeKnowledgeChunk("a3", "D1", "third"), 0.1),
        ]
        adapter = self._make_adapter(logical_to_actual, actual_to_logical, results)

        hits = adapter.search(RetrievalRequest(query="q", top_k=5))
        assert [h.chunk_id for h in hits] == ["C1", "C2", "C3"]
        assert hits[0].score >= hits[1].score >= hits[2].score

    def test_implementation_property(self):
        assert FrameworkBM25RetrievalAdapter.implementation == "framework"
