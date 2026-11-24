"""Answer-generation adapters for the custom and framework editions.

Each adapter accepts a normalized answer-quality request, executes its own
answer-generation path, and returns the generated answer plus the retrieved
context. No implementation is used as a fallback for the other.

Retrieval isolation:
- Custom adapter uses the Custom Retrieval Stack (in-memory BM25) for
  pre-retrieved context and passes ``db=None`` to the Custom Agent
  Orchestrator. The Custom agent's ``search_knowledge`` tool raises
  ``RuntimeError("Missing tenant context for knowledge search.")`` before
  reaching ``KnowledgeService`` (LlamaIndex), so LlamaIndex is never invoked
  for Custom answer generation.
- Framework adapter uses LlamaIndex BM25 (via ``FrameworkBM25RetrievalAdapter``
  backed by ``KnowledgeService``) for pre-retrieved context and passes
  ``db=self._db`` to the Framework LangGraph agent, so ``search_knowledge``
  uses LlamaIndex as intended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievedContext:
    """A single retrieved context chunk with its identifier and content."""

    chunk_id: str
    document_id: str | None
    content: str
    score: float | None = None


@dataclass
class AnswerResult:
    """Normalized answer-generation result for one implementation."""

    implementation: str
    question: str
    answer: str | None
    contexts: list[RetrievedContext] = field(default_factory=list)
    context_chunk_ids: list[str] = field(default_factory=list)
    status: str = "completed"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.content for c in self.contexts]


class AnswerAdapter:
    """Base interface for answer-generation adapters."""

    implementation: str = "base"

    def generate(self, question: str, tenant_id: str, top_k: int = 5) -> AnswerResult:
        raise NotImplementedError

    def retrieval_backend(self) -> str:
        """Return the name of the retrieval backend used by this adapter."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Custom edition answer adapter
# ---------------------------------------------------------------------------

class CustomAnswerAdapter(AnswerAdapter):
    """Generate answers using the Custom Edition's answer-generation path.

    Retrieves context using the Custom Retrieval Stack BM25 searcher
    (via CustomBM25RetrievalAdapter), then generates an answer using the
    Custom Agent Orchestrator's run_agent function with the retrieved
    context supplied as part of the user message.

    Retrieval isolation: passes ``db=None`` to ``run_agent`` so the Custom
    agent's ``search_knowledge`` tool cannot reach ``KnowledgeService``
    (LlamaIndex). It raises RuntimeError before the LlamaIndex path is
    entered, guaranteeing the Custom Retrieval Stack is the only retrieval
    backend used.
    """

    implementation = "custom"

    def __init__(
        self,
        chunks: list[dict[str, Any]],
        settings: Any,
        client: Any | None = None,
    ):
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter

        self._retriever = CustomBM25RetrievalAdapter(chunks)
        self._settings = settings
        self._client = client

    def retrieval_backend(self) -> str:
        return "custom-retrieval-stack"

    def generate(self, question: str, tenant_id: str, top_k: int = 5) -> AnswerResult:
        # Retrieve context using the Custom Retrieval Stack.
        from app.comparison.retrieval_adapters import RetrievalRequest

        hits = self._retriever.search(RetrievalRequest(query=question, top_k=top_k))

        contexts = [
            RetrievedContext(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                content=h.content,
                score=h.score,
            )
            for h in hits
        ]

        # Build a context-augmented prompt for the Custom agent.
        context_text = "\n\n".join(c.content for c in contexts)
        if context_text:
            augmented_question = (
                "Answer the user's question using only the following context. "
                "If the context does not contain enough information, say so. "
                "Do not invent facts.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )
        else:
            augmented_question = question

        # Generate answer using the Custom Edition's existing answer-generation path.
        # db=None guarantees the search_knowledge tool cannot reach KnowledgeService
        # (LlamaIndex). RuntimeError is raised before get_knowledge_service(db).
        try:
            from app.agent_core.agent import run_agent as custom_run_agent

            result = custom_run_agent(
                augmented_question,
                settings=self._settings,
                client=self._client,
                max_iterations=5,
                history=None,
                tenant_id=tenant_id,
                db=None,
                is_approved=False,
            )

            answer = result.answer if result.answer else None
            status = result.status
            error = result.error
        except Exception as exc:  # noqa: BLE001 - capture adapter boundary errors
            answer = None
            status = "failed"
            error = str(exc)

        return AnswerResult(
            implementation=self.implementation,
            question=question,
            answer=answer,
            contexts=contexts,
            context_chunk_ids=[c.chunk_id for c in contexts],
            status=status,
            error=error,
            metadata={"top_k": top_k, "num_contexts": len(contexts), "retrieval_backend": self.retrieval_backend()},
        )


# ---------------------------------------------------------------------------
# Framework edition answer adapter
# ---------------------------------------------------------------------------

class FrameworkAnswerAdapter(AnswerAdapter):
    """Generate answers using the Framework Edition's answer-generation path.

    Retrieves context using LlamaIndex BM25 (via FrameworkBM25RetrievalAdapter
    backed by KnowledgeService), then generates an answer using the
    Framework LangGraph agent's run_agent function with the retrieved
    context supplied as part of the user message.

    Retrieval isolation: passes ``db=self._db`` to ``run_agent`` so
    ``search_knowledge`` uses LlamaIndex as intended for the Framework edition.
    """

    implementation = "framework"

    def __init__(
        self,
        db: Any,
        tenant_id: str,
        documents: list[dict[str, Any]],
        settings: Any,
        client: Any | None = None,
    ):
        from app.comparison.retrieval_adapters import FrameworkBM25RetrievalAdapter

        if db is None:
            raise ValueError("FrameworkBM25RetrievalAdapter requires a database handle")

        self._db = db
        self._tenant_id = tenant_id
        self._retriever = FrameworkBM25RetrievalAdapter(
            db=db, tenant_id=tenant_id, documents=documents
        )
        self._settings = settings
        self._client = client

    def retrieval_backend(self) -> str:
        return "llamaindex"

    def generate(self, question: str, tenant_id: str, top_k: int = 5) -> AnswerResult:
        # Retrieve context using LlamaIndex BM25.
        from app.comparison.retrieval_adapters import RetrievalRequest

        hits = self._retriever.search(RetrievalRequest(query=question, top_k=top_k))

        contexts = [
            RetrievedContext(
                chunk_id=h.chunk_id,
                document_id=h.document_id,
                content=h.content,
                score=h.score,
            )
            for h in hits
        ]

        # Build a context-augmented prompt for the Framework agent.
        context_text = "\n\n".join(c.content for c in contexts)
        if context_text:
            augmented_question = (
                "Answer the user's question using only the following context. "
                "If the context does not contain enough information, say so. "
                "Do not invent facts.\n\n"
                f"Context:\n{context_text}\n\n"
                f"Question: {question}"
            )
        else:
            augmented_question = question

        # Generate answer using the Framework Edition's existing answer-generation path.
        # db=self._db allows search_knowledge to use KnowledgeService (LlamaIndex).
        try:
            from app.modules.agent.service import run_agent as framework_run_agent

            result = framework_run_agent(
                augmented_question,
                settings=self._settings,
                client=self._client,
                max_iterations=5,
                history=None,
                tenant_id=tenant_id,
                db=self._db,
                is_approved=False,
            )

            answer = result.answer if result.answer else None
            status = result.status
            error = result.error
        except Exception as exc:  # noqa: BLE001 - capture adapter boundary errors
            answer = None
            status = "failed"
            error = str(exc)

        return AnswerResult(
            implementation=self.implementation,
            question=question,
            answer=answer,
            contexts=contexts,
            context_chunk_ids=[c.chunk_id for c in contexts],
            status=status,
            error=error,
            metadata={"top_k": top_k, "num_contexts": len(contexts), "retrieval_backend": self.retrieval_backend()},
        )
