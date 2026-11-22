"""LlamaIndex adapters used by the knowledge service.

The application owns tenant-scoped document and chunk storage. LlamaIndex is
used for the retrieval infrastructure only: node representation, sentence
splitting, embeddings, BM25 retrieval, vector retrieval, hybrid fusion, and
reranking. Tenant scoping is enforced before any index is built.
"""

from __future__ import annotations

from typing import Any, Sequence

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.retrievers.bm25 import BM25Retriever


class SentenceTransformerEmbedding(BaseEmbedding):
    """LlamaIndex embedding adapter backed by sentence-transformers."""

    _model: Any = PrivateAttr()

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    @classmethod
    def class_name(cls) -> str:
        return "SentenceTransformerEmbedding"

    def _embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, convert_to_numpy=True)
        return [float(value) for value in vector]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [[float(value) for value in vector] for vector in vectors]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)


class FilteredBM25Retriever(BaseRetriever):
    """BM25 retriever that drops non-positive matches.

    The LlamaIndex BM25 integration returns zero-score entries when the
    requested ``top_k`` is larger than the number of matching chunks. Those
    entries must not enter hybrid fusion as if they were real matches.
    """

    def __init__(
        self,
        nodes: Sequence[BaseNode],
        similarity_top_k: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._bm25 = BM25Retriever(
            nodes=list(nodes),
            similarity_top_k=similarity_top_k,
            language="en",
            verbose=False,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        results = self._bm25.retrieve(query_bundle.query_str)
        return [
            result
            for result in results
            if result.score is not None and result.score > 0
        ]