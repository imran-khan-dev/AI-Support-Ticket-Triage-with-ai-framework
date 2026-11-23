"""Retrieval adapters for comparative benchmarking.

These adapters translate the shared contract into each implementation's native
inputs and normalize outputs back. They intentionally avoid importing private
classes beyond the stable public entry points of each implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ComparisonRunConfig, Implementation, RetrievalSample


@dataclass
class RetrievalRequest:
    query: str
    top_k: int
    tenant_id: str | None = None


@dataclass
class RetrievalHit:
    chunk_id: str
    document_id: str | None = None
    content: str = ""
    score: float | None = None


class RetrievalAdapter:
    """Base adapter interface for retrieval implementations."""

    implementation: str

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        raise NotImplementedError


class CustomBM25RetrievalAdapter(RetrievalAdapter):
    """Adapter around the standalone Retrieval Stack BM25 searcher.

    Builds an in-memory inverted index directly from chunk content, so no
    database or embedding model is required. Chunk identifiers are the stable
    logical ids from the dataset.
    """

    implementation = Implementation.custom

    def __init__(self, chunks: list[dict[str, Any]]):
        from retrieval_stack.bm25 import BM25Searcher, InvertedIndex
        from retrieval_stack.bm25.tokenizer import Tokenizer
        from retrieval_stack.chunking.models import Chunk

        self._chunks = chunks
        index = InvertedIndex(tokenizer=Tokenizer())
        for chunk in chunks:
            index.add_chunk(
                Chunk(
                    id=chunk["logical_id"],
                    document_id=chunk.get("document_id", ""),
                    index=chunk.get("chunk_index", 0),
                    content=chunk["content"],
                    token_count=max(1, len(chunk["content"].split())),
                    metadata=chunk.get("metadata", {}),
                )
            )
        self._searcher = BM25Searcher(index)

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        results = self._searcher.search(request.query, top_k=request.top_k)
        hits: list[RetrievalHit] = []
        for r in results:
            chunk = getattr(r, "chunk", None)
            content = chunk.content if chunk is not None else ""
            hits.append(
                RetrievalHit(
                    chunk_id=r.chunk_id,
                    document_id=r.document_id,
                    content=content,
                    score=float(r.score),
                )
            )
        return hits


class FrameworkBM25RetrievalAdapter(RetrievalAdapter):
    """Adapter around the LlamaIndex-backed KnowledgeService BM25 search.

    Requires a database handle and ingests the supplied documents into the
    tenant's knowledge base before searching.
    """

    implementation = Implementation.framework

    def __init__(self, db: Any, tenant_id: str, documents: list[dict[str, Any]]):
        from app.knowledge.repository import create_chunks, create_document

        self._db = db
        self._tenant_id = tenant_id
        self._logical_to_actual: dict[str, str] = {}
        self._actual_to_logical: dict[str, str] = {}
        self._service = None
        self._create_document = create_document
        self._create_chunks = create_chunks
        self._setup(db, documents)

    def _setup(self, db: Any, documents: list[dict[str, Any]]) -> None:
        from app.database.repository import _canonical_uuid

        canonical = _canonical_uuid(self._tenant_id)
        if canonical is None:
            raise ValueError(f"tenant_id is not a valid UUID: {self._tenant_id}")
        with db.write_transaction() as conn:
            conn.execute(
                "INSERT INTO tenants (id, name, enabled) VALUES (%s, %s, TRUE) "
                "ON CONFLICT (id) DO NOTHING",
                (canonical, f"eval-{self._tenant_id[:8]}"),
            )
            conn.execute(
                "DELETE FROM knowledge_documents WHERE tenant_id = %s",
                (canonical,),
            )
        for doc in documents:
            document = self._create_document(
                self._db,
                tenant_id=self._tenant_id,
                filename=doc["filename"],
                content_type=doc["content_type"],
                byte_size=len(doc["content"].encode("utf-8")),
                checksum=doc.get("checksum", ""),
            )
            chunk_dicts = [
                {
                    "chunk_index": c["chunk_index"],
                    "content": c["content"],
                    "token_count": max(1, len(c["content"].split())),
                    "start_char": 0,
                    "end_char": len(c["content"]),
                    "metadata": c.get("metadata", {}),
                }
                for c in doc["chunks"]
            ]
            stored = self._create_chunks(
                self._db,
                tenant_id=self._tenant_id,
                document_id=document.id,
                chunks=chunk_dicts,
            )
            for logical, stored_chunk in zip(doc["chunks"], stored):
                self._logical_to_actual[logical["logical_id"]] = stored_chunk.id
                self._actual_to_logical[stored_chunk.id] = logical["logical_id"]

    @property
    def service(self) -> Any:
        if self._service is None:
            from app.knowledge.service import KnowledgeService

            self._service = KnowledgeService(self._db)
            self._service.invalidate_index(self._tenant_id)
        return self._service

    def search(self, request: RetrievalRequest) -> list[RetrievalHit]:
        results = self.service.search(
            self._tenant_id, request.query, top_k=request.top_k, rerank=False
        )
        hits: list[RetrievalHit] = []
        for result in results:
            chunk = getattr(result, "chunk", None)
            actual_id = getattr(chunk, "id", None) if chunk is not None else None
            if actual_id is None or actual_id not in self._actual_to_logical:
                continue
            logical_id = self._actual_to_logical[actual_id]
            hits.append(
                RetrievalHit(
                    chunk_id=logical_id,
                    document_id=getattr(chunk, "document_id", None) if chunk is not None else None,
                    content=chunk.content if chunk is not None else "",
                    score=float(getattr(result, "score", 0.0) or 0.0),
                )
            )
        return hits