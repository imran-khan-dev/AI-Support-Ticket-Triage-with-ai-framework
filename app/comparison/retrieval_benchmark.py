"""Comparative retrieval benchmark runner.

Loads a shared dataset, executes one or both retrieval adapters with identical
inputs, computes framework-independent retrieval metrics, and produces a
normalized result structure suitable for later comparison.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from .contracts import Implementation
from .metrics import average, dcg_at_k, idcg_at_k, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .retrieval_adapters import (
    FrameworkBM25RetrievalAdapter,
    RetrievalAdapter,
    RetrievalHit,
    RetrievalRequest,
    CustomBM25RetrievalAdapter,
)


@dataclass
class RetrievalQueryResult:
    query_id: str
    query_text: str
    implementation: str
    retrieved_chunk_ids: list[str]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    ndcg_at_k: dict[int, float]
    status: str = "completed"
    error: str | None = None


@dataclass
class RetrievalBenchmarkResult:
    experiment_id: str
    dataset_id: str
    dataset_name: str
    implementation: str
    top_k: int
    num_queries: int
    per_query: list[RetrievalQueryResult] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None


def _build_documents_from_dataset(dataset: Any) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for doc in getattr(dataset, "documents", []):
        documents.append(
            {
                "id": doc.id,
                "filename": doc.filename,
                "content_type": doc.content_type,
                "content": doc.content,
                "chunks": [
                    {
                        "logical_id": c.logical_id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "metadata": dict(c.metadata),
                    }
                    for c in doc.chunks
                ],
            }
        )
    return documents


def _build_custom_chunks(dataset: Any) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for doc in getattr(dataset, "documents", []):
        for c in doc.chunks:
            chunks.append(
                {
                    "logical_id": c.logical_id,
                    "document_id": doc.id,
                    "chunk_index": c.chunk_index,
                    "content": c.content,
                    "metadata": dict(c.metadata),
                }
            )
    return chunks


def _run_one(
    adapter: RetrievalAdapter,
    queries: list[Any],
    top_k: int,
) -> list[RetrievalQueryResult]:
    results: list[RetrievalQueryResult] = []
    for q in queries:
        try:
            hits = adapter.search(RetrievalRequest(query=q.query_text, top_k=top_k))
            ids = [h.chunk_id for h in hits]
            relevant = getattr(q, "relevance_judgments", {}) or {}
            results.append(
                RetrievalQueryResult(
                    query_id=q.query_id,
                    query_text=q.query_text,
                    implementation=adapter.implementation,
                    retrieved_chunk_ids=ids,
                    precision_at_k={k: precision_at_k(ids, relevant, k) for k in [top_k]},
                    recall_at_k={k: recall_at_k(ids, relevant, k) for k in [top_k]},
                    reciprocal_rank=reciprocal_rank(ids, relevant),
                    ndcg_at_k={k: ndcg_at_k(ids, relevant, k) for k in [top_k]},
                )
            )
        except Exception as exc:
            results.append(
                RetrievalQueryResult(
                    query_id=q.query_id,
                    query_text=q.query_text,
                    implementation=adapter.implementation,
                    retrieved_chunk_ids=[],
                    precision_at_k={top_k: 0.0},
                    recall_at_k={top_k: 0.0},
                    reciprocal_rank=0.0,
                    ndcg_at_k={top_k: 0.0},
                    status="failed",
                    error=str(exc),
                )
            )
    return results


def run_retrieval_benchmark(
    dataset: Any,
    implementation: str,
    top_k: int = 5,
    db: Any = None,
    tenant_id: str | None = None,
    experiment_id: str | None = None,
) -> RetrievalBenchmarkResult:
    if implementation not in (Implementation.custom, Implementation.framework):
        raise ValueError(f"unsupported implementation: {implementation}")
    queries = list(getattr(dataset, "queries", []))
    if not queries:
        return RetrievalBenchmarkResult(
            experiment_id=experiment_id or str(uuid.uuid4()),
            dataset_id=getattr(dataset, "name", "unknown"),
            dataset_name=getattr(dataset, "name", "unknown"),
            implementation=implementation,
            top_k=top_k,
            num_queries=0,
            status="failed",
            error="dataset has no queries",
        )
    if implementation == Implementation.custom:
        try:
            chunks = _build_custom_chunks(dataset)
            adapter: RetrievalAdapter = CustomBM25RetrievalAdapter(chunks)
            adapter_error: str | None = None
        except Exception as exc:
            adapter = None
            adapter_error = str(exc)
    else:
        if db is None:
            raise ValueError("framework implementation requires a database handle")
        documents = _build_documents_from_dataset(dataset)
        try:
            adapter = FrameworkBM25RetrievalAdapter(
                db=db,
                tenant_id=tenant_id or getattr(dataset, "tenant_id", "default"),
                documents=documents,
            )
            adapter_error = None
        except Exception as exc:
            adapter = None
            adapter_error = str(exc)

    if adapter is None:
        # Adapter construction failed: mark every query as failed without
        # crashing the run, so the other implementation is never affected.
        per_query = [
            RetrievalQueryResult(
                query_id=q.query_id,
                query_text=q.query_text,
                implementation=implementation,
                retrieved_chunk_ids=[],
                precision_at_k={top_k: 0.0},
                recall_at_k={top_k: 0.0},
                reciprocal_rank=0.0,
                ndcg_at_k={top_k: 0.0},
                status="failed",
                error=adapter_error or "adapter construction failed",
            )
            for q in queries
        ]
    else:
        per_query = _run_one(adapter, queries, top_k)
    n = len(per_query)
    completed = [p for p in per_query if p.status == "completed"]
    aggregate = {
        "precision_at_k": {top_k: average([p.precision_at_k[top_k] for p in completed])} if completed else None,
        "recall_at_k": {top_k: average([p.recall_at_k[top_k] for p in completed])} if completed else None,
        "mrr": average([p.reciprocal_rank for p in completed]) if completed else None,
        "ndcg_at_k": {top_k: average([p.ndcg_at_k[top_k] for p in completed])} if completed else None,
        "num_completed": len(completed),
        "num_failed": n - len(completed),
    }
    status = "completed" if len(completed) == n else "partial"
    return RetrievalBenchmarkResult(
        experiment_id=experiment_id or str(uuid.uuid4()),
        dataset_id=getattr(dataset, "name", "unknown"),
        dataset_name=getattr(dataset, "name", "unknown"),
        implementation=implementation,
        top_k=top_k,
        num_queries=n,
        per_query=per_query,
        aggregate=aggregate,
        status=status,
    )