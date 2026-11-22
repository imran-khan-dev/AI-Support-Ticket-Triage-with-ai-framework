"""LlamaIndex-specific knowledge retrieval tests.

These tests exercise the framework-backed retrieval path with deterministic
embeddings and a deterministic reranker, so they do not download models.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import NodeWithScore, QueryBundle

from app.database import get_database
from app.database.repository import _canonical_uuid
from app.knowledge.llamaindex import FilteredBM25Retriever
from app.knowledge.service import IngestionError, KnowledgeService
from app.modules.integrations.service import issue_integration_credential


class DeterministicEmbedding(BaseEmbedding):
    """Small deterministic bag-of-words embedding for tests."""

    @classmethod
    def class_name(cls) -> str:
        return "DeterministicEmbedding"

    def _embed(self, text: str) -> list[float]:
        dimensions = 16
        vector = [0.0] * dimensions
        for token in re.findall(r"\w+", text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[digest[0] % dimensions] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)


class ReverseReranker(BaseNodePostprocessor):
    """Deterministic test reranker that reverses the candidate order."""

    top_n: int = 50

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        return list(reversed(nodes))[: self.top_n]


def _new_tenant(db) -> str:
    integration, _, _ = issue_integration_credential(
        db,
        integration_name=f"llamaindex-{uuid.uuid4().hex[:12]}",
    )
    tenant_id = str(uuid.uuid4())
    with db.write_transaction() as conn:
        conn.execute(
            "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE)",
            (_canonical_uuid(tenant_id), f"llamaindex-tenant-{tenant_id[:8]}"),
        )
        conn.execute(
            "UPDATE integrations SET tenant_id = %s WHERE id = %s",
            (_canonical_uuid(tenant_id), _canonical_uuid(integration.id)),
        )
    return tenant_id


def _service(db, *, reranker=None) -> KnowledgeService:
    return KnowledgeService(
        db,
        chunk_size=32,
        chunk_overlap=8,
        embed_model=DeterministicEmbedding(),
        reranker=reranker or ReverseReranker(),
        enable_reranking=True,
    )


def _ingest(service: KnowledgeService, tenant_id: str, text: str, filename: str):
    return service.ingest_document(
        tenant_id=tenant_id,
        file_content=text.encode("utf-8"),
        filename=filename,
        content_type="text/plain",
    )


def test_active_service_has_no_legacy_retrieval_stack_imports():
    service_path = Path(__file__).resolve().parent.parent / "app" / "knowledge" / "service.py"
    source = service_path.read_text(encoding="utf-8")
    assert "retrieval_stack" not in source
    assert "Retrieval-Stack" not in source


def test_ingestion_creates_lamaindex_chunks_with_metadata():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    result = _ingest(
        service,
        tenant_id,
        "First paragraph about billing refunds.\n\nSecond paragraph about account access.",
        "policy.txt",
    )

    assert result.document.status == "completed"
    assert result.chunks_created >= 1

    chunks = service.get_document_chunks(tenant_id, result.document.id)
    assert chunks
    for chunk in chunks:
        assert chunk.token_count > 0
        assert chunk.start_char >= 0
        assert chunk.end_char > chunk.start_char
        assert chunk.metadata == {}


def test_pdf_ingestion_extracts_text_and_page_metadata():
    import pymupdf

    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "PDF billing refund policy")
        pdf_bytes = document.tobytes()
    finally:
        document.close()

    result = service.ingest_document(
        tenant_id=tenant_id,
        file_content=pdf_bytes,
        filename="policy.pdf",
        content_type="application/pdf",
    )

    assert result.document.status == "completed"
    assert result.chunks_created >= 1
    chunks = service.get_document_chunks(tenant_id, result.document.id)
    assert any("billing refund policy" in chunk.content.lower() for chunk in chunks)


def test_all_retrieval_modes_are_tenant_scoped():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_a = _new_tenant(db)
    tenant_b = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_a, "Tenant A confidential billing data", "a.txt")
    _ingest(service, tenant_b, "Tenant B confidential shipping data", "b.txt")

    for search in (
        service.search,
        service.search_dense,
        service.search_hybrid,
    ):
        results_a = search(tenant_a, "confidential", top_k=5)
        results_b = search(tenant_b, "confidential", top_k=5)

        assert len(results_a) == 1
        assert len(results_b) == 1
        assert "tenant a" in results_a[0].chunk.content.lower()
        assert "tenant b" in results_b[0].chunk.content.lower()


def test_reranking_is_tenant_scoped_and_changes_score_type():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_a = _new_tenant(db)
    tenant_b = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_a, "Tenant A confidential billing data", "a.txt")
    _ingest(service, tenant_b, "Tenant B confidential shipping data", "b.txt")

    results_a = service.search(tenant_a, "confidential", top_k=5, rerank=True)
    results_b = service.search(tenant_b, "confidential", top_k=5, rerank=True)

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert all(result.score_type == "rerank" for result in results_a)
    assert all(result.score_type == "rerank" for result in results_b)
    assert "tenant a" in results_a[0].chunk.content.lower()
    assert "tenant b" in results_b[0].chunk.content.lower()


def test_reranking_changes_candidate_order():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "alpha shared content", "a.txt")
    _ingest(service, tenant_id, "beta shared content", "b.txt")

    bm25_results = service.search(tenant_id, "shared", top_k=5)
    reranked_results = service.search(tenant_id, "shared", top_k=5, rerank=True)

    assert len(bm25_results) >= 2
    assert len(reranked_results) >= 2
    assert bm25_results[0].chunk.id != reranked_results[0].chunk.id
    assert all(result.score_type == "rerank" for result in reranked_results)


def test_bm25_retriever_drops_zero_score_matches():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refunds", "billing.txt")
    retriever = service._get_bm25_retriever(tenant_id, 10)

    assert retriever is not None
    assert retriever.retrieve("nonexistent") == []
    assert retriever.retrieve("billing")


def test_delete_invalidates_bm25_dense_and_hybrid_indexes():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    result = _ingest(service, tenant_id, "billing refunds within five days", "billing.txt")

    for search in (
        service.search,
        service.search_dense,
        service.search_hybrid,
    ):
        assert search(tenant_id, "billing refunds", top_k=5)

    assert service.delete_document(tenant_id, result.document.id) is True

    for search in (
        service.search,
        service.search_dense,
        service.search_hybrid,
    ):
        assert search(tenant_id, "billing refunds", top_k=5) == []


def test_empty_and_whitespace_documents_fail_without_chunks():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    with pytest.raises(IngestionError):
        _ingest(service, tenant_id, "", "empty.txt")

    with pytest.raises(IngestionError):
        _ingest(service, tenant_id, "   \n\t  ", "whitespace.txt")

    failed_documents = service.list_documents(tenant_id, status="failed")
    assert len(failed_documents) == 2
    for document in failed_documents:
        assert service.get_document_chunks(tenant_id, document.id) == []


def test_hybrid_results_are_sorted_and_typed():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refund policy five days", "billing.txt")
    _ingest(service, tenant_id, "shipping policy twenty four hours", "shipping.txt")

    results = service.search_hybrid(tenant_id, "policy", top_k=5)

    assert results
    assert all(result.score_type == "hybrid" for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results),
        reverse=True,
    )


def test_hybrid_retriever_configuration_uses_rrf_without_query_generation():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refund policy five days", "billing.txt")
    hybrid = service._get_hybrid_retriever(tenant_id, 3)

    assert isinstance(hybrid, QueryFusionRetriever)
    assert hybrid.mode is FUSION_MODES.RECIPROCAL_RANK
    assert hybrid.num_queries == 1
    assert hybrid.similarity_top_k == 3
    assert len(hybrid._retrievers) == 2


def test_hybrid_executes_both_retrievers_without_calling_query_generation_llm(monkeypatch):
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refund policy five days", "billing.txt")

    calls: list[str] = []
    original_bm25_retrieve = FilteredBM25Retriever.retrieve
    original_dense_retrieve = VectorIndexRetriever.retrieve

    def bm25_retrieve(self, query):
        calls.append("bm25")
        return original_bm25_retrieve(self, query)

    def dense_retrieve(self, query):
        calls.append("dense")
        return original_dense_retrieve(self, query)

    monkeypatch.setattr(FilteredBM25Retriever, "retrieve", bm25_retrieve)
    monkeypatch.setattr(VectorIndexRetriever, "retrieve", dense_retrieve)

    def fail_llm(*args, **kwargs):
        raise AssertionError("hybrid retrieval must not call a query-generation LLM")

    monkeypatch.setattr(MockLLM, "complete", fail_llm)
    monkeypatch.setattr(MockLLM, "acomplete", fail_llm)

    results = service.search_hybrid(tenant_id, "billing refund", top_k=5)

    assert results
    assert calls.count("bm25") == 1
    assert calls.count("dense") == 1


def test_hybrid_applies_reciprocal_rank_fusion_and_deduplicates():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refund policy five days", "billing.txt")

    bm25_results = service.search(tenant_id, "billing refund", top_k=5)
    dense_results = service.search_dense(tenant_id, "billing refund", top_k=5)
    hybrid_results = service.search_hybrid(tenant_id, "billing refund", top_k=5)

    assert len(bm25_results) == 1
    assert len(dense_results) == 1
    assert len(hybrid_results) == 1
    assert bm25_results[0].chunk.id == dense_results[0].chunk.id
    assert hybrid_results[0].chunk.id == bm25_results[0].chunk.id
    assert hybrid_results[0].score == pytest.approx(2.0 / 60.0)


def test_hybrid_single_source_reciprocal_rank_score():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refund policy five days", "billing.txt")
    results = service.search_hybrid(tenant_id, "xyz123nonexistent", top_k=5)

    assert len(results) == 1
    assert results[0].score == pytest.approx(1.0 / 60.0)


def test_tenant_nodes_and_vector_indexes_are_cached_and_reused():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_a = _new_tenant(db)
    tenant_b = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_a, "Tenant A confidential billing data", "a.txt")
    _ingest(service, tenant_b, "Tenant B confidential shipping data", "b.txt")

    service.search(tenant_a, "confidential", top_k=5)
    nodes_a = service._nodes[tenant_a]
    service.search(tenant_a, "confidential", top_k=5)
    assert service._nodes[tenant_a] is nodes_a

    service.search_dense(tenant_b, "confidential", top_k=5)
    index_b = service._vector_indexes[tenant_b]
    service.search_dense(tenant_b, "confidential", top_k=5)
    assert service._vector_indexes[tenant_b] is index_b

    node_ids_a = {node.node_id for node in nodes_a}
    node_ids_b = {node.node_id for node in service._nodes[tenant_b]}
    assert node_ids_a.isdisjoint(node_ids_b)


def test_ingestion_invalidates_tenant_caches_and_rebuilds_them():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_id, "billing refunds", "billing.txt")
    service.search(tenant_id, "billing", top_k=5)
    service.search_dense(tenant_id, "billing", top_k=5)
    old_nodes = service._nodes[tenant_id]
    old_index = service._vector_indexes[tenant_id]

    _ingest(service, tenant_id, "shipping refunds", "shipping.txt")

    assert tenant_id not in service._nodes
    assert tenant_id not in service._vector_indexes

    results = service.search(tenant_id, "shipping", top_k=5)
    assert results
    assert service._nodes[tenant_id] is not old_nodes

    service.search_dense(tenant_id, "shipping", top_k=5)
    assert service._vector_indexes[tenant_id] is not old_index


def test_deletion_invalidates_tenant_caches_and_removes_stale_nodes():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    service = _service(db)

    result = _ingest(service, tenant_id, "billing refunds", "billing.txt")
    service.search(tenant_id, "billing", top_k=5)
    service.search_dense(tenant_id, "billing", top_k=5)
    stale_node_ids = {node.node_id for node in service._nodes[tenant_id]}
    assert stale_node_ids

    assert service.delete_document(tenant_id, result.document.id) is True

    assert tenant_id not in service._nodes
    assert tenant_id not in service._vector_indexes
    rebuilt_nodes = service._get_nodes(tenant_id)
    assert rebuilt_nodes == []
    assert stale_node_ids.isdisjoint({node.node_id for node in rebuilt_nodes})


def test_new_service_reconstructs_indexes_from_postgres():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_id = _new_tenant(db)
    first_service = _service(db)
    _ingest(first_service, tenant_id, "billing refunds within five days", "billing.txt")

    restarted_service = _service(db)

    assert restarted_service.search(tenant_id, "billing", top_k=5)
    assert restarted_service.search_dense(tenant_id, "billing", top_k=5)
    assert restarted_service.search_hybrid(tenant_id, "billing", top_k=5)


def test_concurrent_tenant_searches_do_not_cross_contaminate():
    db = get_database(os.environ["DATABASE_URL"])
    tenant_a = _new_tenant(db)
    tenant_b = _new_tenant(db)
    service = _service(db)

    _ingest(service, tenant_a, "Tenant A confidential billing data", "a.txt")
    _ingest(service, tenant_b, "Tenant B confidential shipping data", "b.txt")

    def search(tenant_id: str):
        return service.search_hybrid(tenant_id, "confidential", top_k=5)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(search, tenant_a if index % 2 == 0 else tenant_b)
            for index in range(24)
        ]
        results = [future.result() for future in futures]

    for index, result in enumerate(results):
        expected = "tenant a" if index % 2 == 0 else "tenant b"
        assert len(result) == 1
        assert expected in result[0].chunk.content.lower()