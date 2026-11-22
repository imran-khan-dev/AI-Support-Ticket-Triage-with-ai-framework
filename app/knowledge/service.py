"""Knowledge base service layer.

The application owns tenant-scoped document and chunk persistence. LlamaIndex
provides the retrieval infrastructure: document/node representation, sentence
splitting, embeddings, BM25 retrieval, vector retrieval, hybrid fusion, and
reranking. Every index is built only from chunks belonging to one tenant.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

import pymupdf
import tiktoken
from llama_index.core import VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document, NodeWithScore, QueryBundle, TextNode
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank

from app.database import Database
from app.knowledge.llamaindex import (
    FilteredBM25Retriever,
    SentenceTransformerEmbedding,
)
from app.knowledge.repository import (
    KnowledgeChunk,
    KnowledgeDocument,
    create_chunks,
    create_document,
    delete_document,
    get_all_chunks_for_tenant,
    get_chunks_for_document,
    get_document,
    list_documents,
    update_document_status,
)


class IngestionError(Exception):
    """Raised when a document cannot be extracted or split into chunks."""


@dataclass(frozen=True)
class IngestionResult:
    document: KnowledgeDocument
    chunks_created: int


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float
    score_type: Literal["bm25", "dense", "hybrid", "rerank"] = "bm25"


class KnowledgeService:
    """Tenant-isolated knowledge base service backed by LlamaIndex."""

    def __init__(
        self,
        db: Database,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        dense_model_name: str = "all-MiniLM-L6-v2",
        reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_reranking: bool = False,
        *,
        embed_model: BaseEmbedding | None = None,
        reranker: BaseNodePostprocessor | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self._db = db
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self._tokenizer.encode,
            include_metadata=False,
            include_prev_next_rel=False,
        )
        self._dense_model_name = dense_model_name
        self._reranker_model_name = reranker_model_name
        self._enable_reranking = enable_reranking
        self._embed_model = embed_model
        self._reranker = reranker
        self._nodes: dict[str, list[TextNode]] = {}
        self._vector_indexes: dict[str, VectorStoreIndex] = {}

    def _get_embed_model(self) -> BaseEmbedding:
        if self._embed_model is None:
            self._embed_model = SentenceTransformerEmbedding(
                model_name=self._dense_model_name
            )
        return self._embed_model

    def _get_reranker(self) -> BaseNodePostprocessor:
        if self._reranker is None:
            self._reranker = SentenceTransformerRerank(
                model=self._reranker_model_name,
                top_n=50,
            )
        return self._reranker

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def _extract(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> str:
        is_pdf = filename.lower().endswith(".pdf") or content_type == "application/pdf"

        if is_pdf:
            try:
                with pymupdf.open(stream=file_content, filetype="pdf") as document:
                    text = "\n\n".join(page.get_text() for page in document)
            except Exception as exc:
                raise IngestionError(f"Failed to extract text from PDF: {filename}") from exc
        else:
            try:
                text = file_content.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text = file_content.decode("latin-1")
                except Exception as exc:
                    raise IngestionError(
                        f"Failed to extract text from file: {filename}"
                    ) from exc

        normalized = self._normalize_text(text)
        if not normalized:
            raise IngestionError(f"No extractable text found in file: {filename}")
        return normalized

    def _chunk_text(self, text: str) -> list[dict]:
        document = Document(text=text)
        nodes = self._splitter.get_nodes_from_documents([document], show_progress=False)

        chunks: list[dict] = []
        for node in nodes:
            content = node.get_content()
            if not content.strip():
                continue

            start_char = node.start_char_idx if node.start_char_idx is not None else 0
            end_char = node.end_char_idx if node.end_char_idx is not None else len(text)
            if end_char <= start_char:
                end_char = min(len(text), start_char + len(content))

            chunks.append(
                {
                    "chunk_index": len(chunks),
                    "content": content,
                    "token_count": len(self._tokenizer.encode(content)),
                    "start_char": start_char,
                    "end_char": end_char,
                    "metadata": {},
                }
            )

        return chunks

    @staticmethod
    def _chunk_to_node(chunk: KnowledgeChunk) -> TextNode:
        metadata = {
            "tenant_id": chunk.tenant_id,
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
        }
        metadata.update(chunk.metadata or {})
        return TextNode(id_=chunk.id, text=chunk.content, metadata=metadata)

    def invalidate_index(self, tenant_id: str) -> None:
        self._nodes.pop(tenant_id, None)
        self._vector_indexes.pop(tenant_id, None)

    def _get_nodes(self, tenant_id: str) -> list[TextNode]:
        if tenant_id not in self._nodes:
            chunks = get_all_chunks_for_tenant(self._db, tenant_id)
            self._nodes[tenant_id] = [self._chunk_to_node(chunk) for chunk in chunks]
        return self._nodes[tenant_id]

    def _get_vector_index(self, tenant_id: str) -> VectorStoreIndex | None:
        if tenant_id not in self._vector_indexes:
            nodes = self._get_nodes(tenant_id)
            if not nodes:
                return None
            self._vector_indexes[tenant_id] = VectorStoreIndex(
                nodes=nodes,
                embed_model=self._get_embed_model(),
                show_progress=False,
            )
        return self._vector_indexes[tenant_id]

    def _get_bm25_retriever(
        self,
        tenant_id: str,
        similarity_top_k: int,
    ) -> FilteredBM25Retriever | None:
        nodes = self._get_nodes(tenant_id)
        if not nodes:
            return None
        return FilteredBM25Retriever(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
        )

    def _get_dense_retriever(self, tenant_id: str, similarity_top_k: int):
        index = self._get_vector_index(tenant_id)
        if index is None:
            return None
        return index.as_retriever(similarity_top_k=similarity_top_k)

    def _get_hybrid_retriever(self, tenant_id: str, similarity_top_k: int):
        nodes = self._get_nodes(tenant_id)
        if not nodes:
            return None

        bm25_retriever = self._get_bm25_retriever(tenant_id, similarity_top_k)
        dense_retriever = self._get_dense_retriever(tenant_id, similarity_top_k)
        if bm25_retriever is None or dense_retriever is None:
            return None

        return QueryFusionRetriever(
            retrievers=[bm25_retriever, dense_retriever],
            llm=MockLLM(),
            mode=FUSION_MODES.RECIPROCAL_RANK,
            similarity_top_k=similarity_top_k,
            num_queries=1,
            use_async=False,
            verbose=False,
        )

    def _search_nodes(
        self,
        mode: Literal["bm25", "dense", "hybrid"],
        tenant_id: str,
        query: str,
        top_k: int,
    ) -> list[NodeWithScore]:
        if top_k <= 0:
            return []

        if mode == "bm25":
            retriever = self._get_bm25_retriever(tenant_id, top_k)
        elif mode == "dense":
            retriever = self._get_dense_retriever(tenant_id, top_k)
        else:
            retriever = self._get_hybrid_retriever(tenant_id, top_k)

        if retriever is None:
            return []
        return retriever.retrieve(query)

    def _apply_reranking(
        self,
        query: str,
        nodes: list[NodeWithScore],
        top_k: int,
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        reranker = self._get_reranker()
        ranked = reranker.postprocess_nodes(
            nodes=nodes,
            query_bundle=QueryBundle(query_str=query),
        )
        return ranked[:top_k]

    def _to_search_results(
        self,
        tenant_id: str,
        nodes: list[NodeWithScore],
        score_type: Literal["bm25", "dense", "hybrid", "rerank"],
    ) -> list[SearchResult]:
        chunks = get_all_chunks_for_tenant(self._db, tenant_id)
        chunk_map = {chunk.id: chunk for chunk in chunks}

        results: list[SearchResult] = []
        seen: set[str] = set()
        for node_with_score in nodes:
            chunk_id = node_with_score.node.node_id
            if chunk_id in seen:
                continue
            chunk = chunk_map.get(chunk_id)
            if chunk is None:
                continue
            seen.add(chunk_id)
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(node_with_score.score or 0.0),
                    score_type=score_type,
                )
            )
        return results

    def _run_search(
        self,
        mode: Literal["bm25", "dense", "hybrid"],
        tenant_id: str,
        query: str,
        top_k: int,
        rerank: bool,
    ) -> list[SearchResult]:
        nodes = self._search_nodes(mode, tenant_id, query, top_k)
        if rerank:
            if not self._enable_reranking:
                raise RuntimeError(
                    "Reranking requested but not enabled in KnowledgeService. "
                    "Set enable_reranking=True."
                )
            nodes = self._apply_reranking(query, nodes, top_k)
            return self._to_search_results(tenant_id, nodes, "rerank")
        return self._to_search_results(tenant_id, nodes, mode)

    def ingest_document(
        self,
        tenant_id: str,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> IngestionResult:
        checksum = hashlib.sha256(file_content).hexdigest()
        byte_size = len(file_content)

        doc = create_document(
            self._db,
            tenant_id=tenant_id,
            filename=filename,
            content_type=content_type,
            byte_size=byte_size,
            checksum=checksum,
        )

        update_document_status(self._db, tenant_id, doc.id, "processing")

        try:
            text = self._extract(file_content, filename, content_type)
            chunk_dicts = self._chunk_text(text)

            stored_chunks = create_chunks(
                self._db,
                tenant_id=tenant_id,
                document_id=doc.id,
                chunks=chunk_dicts,
            )

            updated_doc = update_document_status(
                self._db, tenant_id, doc.id, "completed"
            )
            self.invalidate_index(tenant_id)

            return IngestionResult(
                document=updated_doc or doc,
                chunks_created=len(stored_chunks),
            )

        except IngestionError as exc:
            update_document_status(
                self._db,
                tenant_id,
                doc.id,
                "failed",
                error_message=str(exc),
            )
            raise
        except Exception as exc:
            update_document_status(
                self._db,
                tenant_id,
                doc.id,
                "failed",
                error_message=str(exc),
            )
            raise

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using BM25 retrieval."""
        return self._run_search("bm25", tenant_id, query, top_k, rerank)

    def search_dense(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using dense vector retrieval."""
        return self._run_search("dense", tenant_id, query, top_k, rerank)

    def search_hybrid(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Search the tenant's knowledge base using hybrid RRF retrieval."""
        return self._run_search("hybrid", tenant_id, query, top_k, rerank)

    def get_document(
        self, tenant_id: str, document_id: str
    ) -> KnowledgeDocument | None:
        return get_document(self._db, tenant_id, document_id)

    def list_documents(
        self,
        tenant_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeDocument]:
        return list_documents(
            self._db,
            tenant_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def get_document_chunks(
        self, tenant_id: str, document_id: str
    ) -> list[KnowledgeChunk]:
        return get_chunks_for_document(self._db, tenant_id, document_id)

    def delete_document(self, tenant_id: str, document_id: str) -> bool:
        result = delete_document(self._db, tenant_id, document_id)
        if result:
            self.invalidate_index(tenant_id)
        return result