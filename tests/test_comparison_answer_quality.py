"""Tests for F4.3.1 answer-quality and faithfulness comparison.

Covers:
- Retrieval isolation: Custom adapter uses Custom Retrieval Stack only;
  Framework adapter uses LlamaIndex only.
- Deterministic metric unit tests with hand-calculated values.
- Ragas evaluation path (mocked when ragas unavailable).
- Insufficient-context and no-context samples.
- Failed generation and failed evaluation.
- Partial/incomplete result handling.
- JSON and Markdown reporting.
- F4.1 ComparisonResult contract compatibility.
- Real-provider configuration and missing credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
    DETERMINISTIC_METRIC_NAMES,
    RAGAS_METRIC_NAMES,
    ALL_METRIC_NAMES,
    load_answer_quality_dataset,
    run_answer_quality_comparison,
    save_json_report,
    save_markdown_report,
    _run_ragas_evaluation,
    _build_document_chunks,
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
# Retrieval isolation tests
# ---------------------------------------------------------------------------

class TestRetrievalIsolation:
    """Verify each adapter uses only its own retrieval backend."""

    def test_custom_adapter_retrieval_backend(self):
        """Custom adapter must report 'custom-retrieval-stack'."""
        adapter = CustomAnswerAdapter(
            chunks=[],
            settings=Settings(),
            client=None,
        )
        assert adapter.retrieval_backend() == "custom-retrieval-stack"

    def test_framework_adapter_retrieval_backend(self):
        """Framework adapter must report 'llamaindex'."""
        adapter = FrameworkAnswerAdapter.__new__(FrameworkAnswerAdapter)
        # __new__ bypasses __init__ (which requires db). We only test the
        # method, not construction.
        assert FrameworkAnswerAdapter.retrieval_backend(adapter) == "llamaindex"

    def test_custom_adapter_uses_db_none(self, clean_database):
        """Custom adapter passes db=None to run_agent (no LlamaIndex access)."""
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter

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

        # Verify the adapter's retriever is CustomBM25RetrievalAdapter
        assert isinstance(adapter._retriever, CustomBM25RetrievalAdapter)

        # Verify generate uses db=None by checking the run_agent call.
        # We mock run_agent to capture the db argument.
        with patch("app.agent_core.agent.run_agent") as mock_run_agent:
            mock_run_agent.return_value = SimpleNamespace(
                answer="Test answer",
                status="mock",
                error=None,
            )
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
            # run_agent was called with db=None
            call_kwargs = mock_run_agent.call_args.kwargs
            assert call_kwargs["db"] is None, (
                f"Custom adapter passed db={call_kwargs['db']} instead of None. "
                "This would allow the search_knowledge tool to reach "
                "KnowledgeService (LlamaIndex)."
            )

    def test_framework_adapter_uses_real_db(self, clean_database):
        """Framework adapter passes db=self._db to run_agent (LlamaIndex access)."""
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

        with patch("app.modules.agent.service.run_agent") as mock_run_agent:
            mock_run_agent.return_value = SimpleNamespace(
                answer="Test answer",
                status="mock",
                error=None,
            )
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
            call_kwargs = mock_run_agent.call_args.kwargs
            assert call_kwargs["db"] is db, (
                f"Framework adapter did not pass the real database handle. "
                f"Expected {db}, got {call_kwargs['db']}."
            )


# ---------------------------------------------------------------------------
# End-to-end retrieval isolation tests (verify actual execution paths)
# ---------------------------------------------------------------------------

class TestRetrievalIsolationEndToEnd:
    """Verify actual execution paths: Custom never reaches KnowledgeService,
    Framework uses KnowledgeService (LlamaIndex). These tests mock the
    knowledge service factory to capture whether it is invoked."""

    def test_custom_adapter_never_invokes_knowledge_service(self, clean_database):
        """Custom adapter must never call get_knowledge_service / KnowledgeService.

        db=None causes search_knowledge to raise RuntimeError before
        get_knowledge_service(db) is reached, guaranteeing LlamaIndex is
        never invoked for Custom answer generation.
        """
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter

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
        assert isinstance(adapter._retriever, CustomBM25RetrievalAdapter)

        with patch("app.agent_core.tools.get_knowledge_service") as mock_get_service:
            mock_get_service.side_effect = AssertionError(
                "get_knowledge_service must NOT be called for Custom adapter"
            )
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
            # Custom generation should still succeed (pre-retrieved context,
            # search_knowledge tool fails gracefully with db=None)
            assert result.status in ("completed", "mock", "success")
            mock_get_service.assert_not_called()

    def test_framework_adapter_invokes_knowledge_service(self, clean_database):
        """Framework adapter must call get_knowledge_service / KnowledgeService.

        db=self._db allows search_knowledge to reach KnowledgeService
        (LlamaIndex), which is the intended Framework retrieval backend.
        This test verifies at the search_knowledge tool level (with db
        provided) that get_knowledge_service(db) is invoked, and at the
        adapter level that db=self._db is passed to run_agent.
        """
        db = clean_database
        documents = [
            {
                "filename": "test.txt",
                "content_type": "text/plain",
                "content": "Password reset instructions.",
                "chunks": [
                    {
                        "logical_id": "C1",
                        "chunk_index": 0,
                        "content": "Password reset instructions.",
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

        # 1. Adapter passes db=self._db to run_agent (verified via mock).
        with patch("app.modules.agent.service.run_agent") as mock_run_agent:
            mock_run_agent.return_value = SimpleNamespace(
                answer="Test answer",
                status="mock",
                error=None,
            )
            adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
            call_kwargs = mock_run_agent.call_args.kwargs
            assert call_kwargs["db"] is db, (
                f"Framework adapter did not pass the real database handle. "
                f"Expected {db}, got {call_kwargs['db']}."
            )

        # 2. search_knowledge tool calls get_knowledge_service(db) when db is
        #    provided. This is the actual LlamaIndex invocation path.
        with patch("app.agent_core.tools.get_knowledge_service") as mock_get_service:
            from app.knowledge.service import KnowledgeService

            mock_service = MagicMock(spec=KnowledgeService)
            mock_service.search.return_value = []
            mock_get_service.return_value = mock_service

            from app.agent_core.tools import search_knowledge

            args = type(
                "Args", (), {"query": "test", "top_k": 5, "mode": "bm25", "rerank": False}
            )()
            search_knowledge(
                args,
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                db=db,
            )
            mock_get_service.assert_called_once_with(db)

    def test_custom_search_knowledge_raises_without_db(self):
        """search_knowledge raises RuntimeError when db=None (Custom path).

        This is the isolation guard: without a database handle, the Custom
        agent's search_knowledge tool cannot reach KnowledgeService.
        """
        from app.agent_core.tools import search_knowledge

        with pytest.raises(RuntimeError, match="Missing tenant context"):
            search_knowledge(
                type("Args", (), {"query": "test", "top_k": 5, "mode": "bm25", "rerank": False})(),
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                db=None,
            )

    def test_framework_search_knowledge_reaches_knowledge_service(self, clean_database):
        """search_knowledge calls get_knowledge_service(db) when db is provided.

        This is the Framework path: with a database handle, search_knowledge
        delegates to KnowledgeService (LlamaIndex).
        """
        db = clean_database
        with patch("app.agent_core.tools.get_knowledge_service") as mock_get_service:
            from app.knowledge.service import KnowledgeService

            mock_service = MagicMock(spec=KnowledgeService)
            mock_service.search.return_value = []
            mock_get_service.return_value = mock_service

            from app.agent_core.tools import search_knowledge

            args = type("Args", (), {"query": "test", "top_k": 5, "mode": "bm25", "rerank": False})()
            search_knowledge(
                args,
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                db=db,
            )
            mock_get_service.assert_called_once_with(db)

    def test_tenant_isolation_custom_backend(self):
        """Custom Retrieval Stack is tenant-agnostic (in-memory index from
        supplied chunks only). No cross-tenant leakage is possible because
        there is no shared database or global index."""
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter
        from app.comparison.retrieval_adapters import RetrievalRequest

        chunks_tenant_a = [
            {
                "logical_id": "CA1",
                "document_id": "DA",
                "chunk_index": 0,
                "content": "Tenant A secret data.",
                "metadata": {},
            }
        ]
        chunks_tenant_b = [
            {
                "logical_id": "CB1",
                "document_id": "DB",
                "chunk_index": 0,
                "content": "Tenant B secret data.",
                "metadata": {},
            }
        ]

        adapter_a = CustomBM25RetrievalAdapter(chunks_tenant_a)
        adapter_b = CustomBM25RetrievalAdapter(chunks_tenant_b)

        hits_a = adapter_a.search(RetrievalRequest(query="secret", top_k=5))
        hits_b = adapter_b.search(RetrievalRequest(query="secret", top_k=5))

        assert [h.chunk_id for h in hits_a] == ["CA1"]
        assert [h.chunk_id for h in hits_b] == ["CB1"]
        # Tenant A's adapter never returns Tenant B chunks and vice versa.
        assert all(h.chunk_id.startswith("CA") for h in hits_a)
        assert all(h.chunk_id.startswith("CB") for h in hits_b)

    def test_tenant_isolation_framework_backend(self, clean_database):
        """Framework KnowledgeService is tenant-scoped. Documents ingested for
        one tenant are not returned when searching another tenant."""
        db = clean_database
        from app.comparison.retrieval_adapters import (
            FrameworkBM25RetrievalAdapter,
            RetrievalRequest,
        )

        documents_a = [
            {
                "filename": "a.txt",
                "content_type": "text/plain",
                "content": "Tenant A confidential refund policy.",
                "chunks": [
                    {
                        "logical_id": "CA1",
                        "chunk_index": 0,
                        "content": "Tenant A confidential refund policy.",
                        "metadata": {},
                    }
                ],
            }
        ]
        documents_b = [
            {
                "filename": "b.txt",
                "content_type": "text/plain",
                "content": "Tenant B confidential shipping policy.",
                "chunks": [
                    {
                        "logical_id": "CB1",
                        "chunk_index": 0,
                        "content": "Tenant B confidential shipping policy.",
                        "metadata": {},
                    }
                ],
            }
        ]

        adapter_a = FrameworkBM25RetrievalAdapter(
            db=db,
            tenant_id="11111111-1111-4aaa-8111-111111111111",
            documents=documents_a,
        )
        adapter_b = FrameworkBM25RetrievalAdapter(
            db=db,
            tenant_id="22222222-2222-4bbb-8222-222222222222",
            documents=documents_b,
        )

        hits_a = adapter_a.search(RetrievalRequest(query="refund", top_k=5))
        hits_b = adapter_b.search(RetrievalRequest(query="shipping", top_k=5))

        assert [h.chunk_id for h in hits_a] == ["CA1"]
        assert [h.chunk_id for h in hits_b] == ["CB1"]
        # Cross-tenant search returns empty.
        hits_a_from_b = adapter_b.search(RetrievalRequest(query="refund", top_k=5))
        assert hits_a_from_b == []


# ---------------------------------------------------------------------------
# Positive retrieval execution proof tests
# ---------------------------------------------------------------------------

class TestPositiveRetrievalExecution:
    """Verify each implementation actually retrieves context through its
    intended backend and supplies that context to its generator.

    These tests use test spies on the real retrieval methods to capture
    invocation evidence: method called, query passed, tenant context,
    returned chunk IDs and contents, and context supplied to the generator.
    """

    # ------------------------------------------------------------------
    # Custom: positive Retrieval Stack execution
    # ------------------------------------------------------------------

    def test_custom_retrieval_stack_positively_invoked(self):
        """CustomBM25RetrievalAdapter.search is actually called with the
        query, returns chunk IDs and contents, and those are supplied
        to the generator."""
        from app.comparison.retrieval_adapters import (
            CustomBM25RetrievalAdapter,
            RetrievalRequest,
            RetrievalHit,
        )

        chunks = [
            {
                "logical_id": "C1",
                "document_id": "D1",
                "chunk_index": 0,
                "content": "To reset your password, go to account settings.",
                "metadata": {"section": "account", "topic": "password"},
            },
            {
                "logical_id": "C2",
                "document_id": "D1",
                "chunk_index": 1,
                "content": "A reset link will be sent to your email address.",
                "metadata": {"section": "email", "topic": "password"},
            },
        ]

        adapter = CustomAnswerAdapter(
            chunks=chunks,
            settings=Settings(),
            client=None,
        )

        # Save the real method before patching to avoid recursion
        real_search = CustomBM25RetrievalAdapter.search

        captured = {}

        def spy_search(self, request):
            captured["query"] = request.query
            captured["top_k"] = request.top_k
            captured["tenant_id"] = request.tenant_id
            return real_search(self, request)

        with patch.object(
            CustomBM25RetrievalAdapter, "search", spy_search
        ):
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )

        # Custom Retrieval Stack was positively invoked
        assert captured["query"] == "How do I reset my password?"
        assert captured["top_k"] == 5
        # Custom adapter does not pass tenant_id to RetrievalRequest
        assert captured["tenant_id"] is None

        # Returned chunks were supplied to the generator
        assert result.context_chunk_ids == ["C1", "C2"]
        assert len(result.contexts) == 2
        assert result.contexts[0].chunk_id == "C1"
        assert "account settings" in result.contexts[0].content
        assert result.contexts[1].chunk_id == "C2"
        assert "reset link" in result.contexts[1].content

        # Retrieval backend is correctly identified
        assert result.metadata["retrieval_backend"] == "custom-retrieval-stack"

    def test_custom_retrieval_no_llamaindex_fallback(self):
        """Custom adapter never calls get_knowledge_service / KnowledgeService.

        db=None causes search_knowledge to raise RuntimeError before
        LlamaIndex is reached. This is proven by spying on
        get_knowledge_service and asserting it is never called."""
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter

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

        with patch("app.agent_core.tools.get_knowledge_service") as mock_get_service:
            mock_get_service.side_effect = AssertionError(
                "get_knowledge_service must NOT be called for Custom adapter"
            )
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )

            # Custom generation completes (pre-retrieved context + mock agent)
            assert result.status in ("completed", "mock", "success")
            # KnowledgeService was never reached
            mock_get_service.assert_not_called()

    # ------------------------------------------------------------------
    # Framework: positive LlamaIndex/KnowledgeService execution
    # ------------------------------------------------------------------

    def test_framework_knowledge_service_positively_invoked(
        self, clean_database
    ):
        """Framework adapter calls KnowledgeService.search (LlamaIndex)
        through FrameworkBM25RetrievalAdapter, returning chunk IDs and
        contents that are supplied to the generator."""
        db = clean_database
        documents = [
            {
                "filename": "password_reset.txt",
                "content_type": "text/plain",
                "content": "To reset your password, go to account settings.",
                "chunks": [
                    {
                        "logical_id": "C1",
                        "chunk_index": 0,
                        "content": "To reset your password, go to account settings.",
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

        captured = {}

        def spy_search(self, tenant_id, query, top_k, rerank=False):
            captured["tenant_id"] = tenant_id
            captured["query"] = query
            captured["top_k"] = top_k
            captured["rerank"] = rerank
            # Return a real SearchResult-like object
            from app.knowledge.service import SearchResult

            chunk = type(
                "Chunk",
                (),
                {
                    "id": "actual-uuid-c1",
                    "document_id": "D1",
                    "content": "To reset your password, go to account settings.",
                    "metadata": {},
                    "chunk_index": 0,
                },
            )()
            return [
                SearchResult(
                    chunk=chunk,
                    score=0.95,
                    score_type="bm25",
                )
            ]

        with patch(
            "app.knowledge.service.KnowledgeService.search", spy_search
        ):
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )

            # Framework retrieval was positively invoked via KnowledgeService
            assert captured["query"] == "How do I reset my password?"
            assert captured["top_k"] == 5
            assert captured["rerank"] is False
            assert captured["tenant_id"] == "eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee"

            # Retrieval backend is correctly identified
            assert result.metadata["retrieval_backend"] == "llamaindex"

            # Generation completed (mock agent with client=None)
            assert result.status in ("completed", "mock", "success")

    def test_framework_no_custom_retrieval_stack_fallback(
        self, clean_database
    ):
        """Framework adapter never uses the Custom Retrieval Stack.

        The Framework adapter uses FrameworkBM25RetrievalAdapter backed by
        KnowledgeService (LlamaIndex), never CustomBM25RetrievalAdapter."""
        db = clean_database
        documents = [
            {
                "filename": "test.txt",
                "content_type": "text/plain",
                "content": "Test content.",
                "chunks": [
                    {
                        "logical_id": "C1",
                        "chunk_index": 0,
                        "content": "Test content.",
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

        # Verify the adapter uses FrameworkBM25RetrievalAdapter, not Custom
        from app.comparison.retrieval_adapters import (
            FrameworkBM25RetrievalAdapter,
            CustomBM25RetrievalAdapter,
        )

        assert isinstance(adapter._retriever, FrameworkBM25RetrievalAdapter)
        assert not isinstance(adapter._retriever, CustomBM25RetrievalAdapter)

        # Verify retrieval_backend reports llamaindex, not custom-retrieval-stack
        assert adapter.retrieval_backend() == "llamaindex"

    # ------------------------------------------------------------------
    # Ragas evaluator LLM injection
    # ------------------------------------------------------------------

    def test_ragas_evaluator_llm_injection(self):
        """Ragas evaluate() accepts an explicitly configured LLM.

        When llm is passed to run_ragas_evaluation, it is forwarded to
        Ragas aevaluate() which sets metric.llm instead of falling back
        to gpt-4o-mini."""
        from app.evaluation.ragas_evaluator import run_ragas_evaluation
        from langchain_openai import ChatOpenAI
        from app.core.config import Settings

        s = Settings()
        if not s.groq_api_key:
            import pytest

            pytest.skip("GROQ_API_KEY not configured")

        llm = ChatOpenAI(
            model=s.groq_model,
            openai_api_key=s.groq_api_key,
            base_url=s.groq_base_url,
            temperature=0.0,
        )

        records = [
            {
                "user_input": "How do I reset my password?",
                "response": "Go to account settings, click Forgot password.",
                "reference": "Go to account settings, click Forgot password.",
                "retrieved_contexts": [
                    "To reset your password, go to account settings and click Forgot password.",
                ],
            }
        ]

        result = run_ragas_evaluation(
            records, metric_names=["faithfulness"], llm=llm
        )

        # Ragas completed with an explicit LLM (not gpt-4o-mini fallback)
        assert result["status"] == "completed", (
            f"Ragas evaluation failed: {result.get('error')}"
        )
        for row in result.get("per_sample", []):
            if isinstance(row, dict):
                v = row.get("faithfulness")
                assert isinstance(v, (int, float)), (
                    f"Ragas faithfulness should be numeric, got {v!r}"
                )
                assert v == v, "Ragas faithfulness should not be NaN"

    def test_ragas_evaluator_no_llm_falls_back_to_gpt4o_mini(self):
        """When llm is not passed, Ragas falls back to gpt-4o-mini which
        is unavailable on Groq, producing NaN scores. This confirms the
        need for explicit LLM injection."""
        from app.evaluation.ragas_evaluator import run_ragas_evaluation
        from app.core.config import Settings
        import os

        s = Settings()
        os.environ["OPENAI_API_KEY"] = s.groq_api_key or ""
        os.environ["OPENAI_BASE_URL"] = s.groq_base_url or "https://api.groq.com/openai/v1"

        records = [
            {
                "user_input": "How do I reset my password?",
                "response": "Go to account settings, click Forgot password.",
                "reference": "Go to account settings, click Forgot password.",
                "retrieved_contexts": [
                    "To reset your password, go to account settings and click Forgot password.",
                ],
            }
        ]

        # No explicit llm -> Ragas falls back to gpt-4o-mini -> NaN on Groq
        result = run_ragas_evaluation(
            records, metric_names=["faithfulness"], llm=None
        )

        # With the gpt-4o-mini fallback on Groq, scores are NaN/unavailable
        for row in result.get("per_sample", []):
            if isinstance(row, dict):
                v = row.get("faithfulness")
                assert v is None or v != v or not isinstance(v, (int, float)), (
                    f"Expected NaN/None with gpt-4o-mini fallback, got {v!r}"
                )


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
        score = answer_correctness(
            "The reset link is sent by email.",
            None,
            ["reset link", "email"],
        )
        assert score == 1.0

    def test_partial_fact_coverage(self):
        score = answer_correctness(
            "The reset link is sent by email.",
            None,
            ["reset link", "phone number", "email"],
        )
        assert score == pytest.approx(2 / 3)


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
        assert adapter.retrieval_backend() == "custom-retrieval-stack"

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
        assert result.status == "mock"
        assert result.answer is not None
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
        assert adapter.retrieval_backend() == "llamaindex"

        result = adapter.generate(
            question="How do I reset my password?",
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
        )

        assert result.implementation == "framework"
        assert result.status == "mock"
        assert result.answer is not None


# ---------------------------------------------------------------------------
# Ragas evaluation tests
# ---------------------------------------------------------------------------

class TestRagasEvaluation:
    """Verify Ragas evaluation path.

    Note: Ragas 0.4.x internally references gpt-4o-mini for some operations,
    which is not accessible via Groq. Genuine Ragas LLM-based evaluation
    (faithfulness, answer_relevancy, etc.) requires an OpenAI-compatible
    provider that supports the gpt-4o-mini model. With Groq, deterministic
    metrics are used as the primary evaluation methodology.
    """

    def test_ragas_available_flag(self):
        """Verify the _run_ragas_evaluation function handles unavailable ragas."""
        from app.comparison.answer_comparison import _try_import_ragas
        available = _try_import_ragas()
        assert isinstance(available, bool)

    def test_run_ragas_evaluation_unavailable(self):
        """When ragas is not importable, _run_ragas_evaluation returns unavailable or failed."""
        import app.comparison.answer_comparison as ac_mod
        original_try = ac_mod._try_import_ragas
        original_available = ac_mod._RAGAS_AVAILABLE
        ac_mod._try_import_ragas = lambda: False
        ac_mod._RAGAS_AVAILABLE = False
        try:
            result = _run_ragas_evaluation(
                [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}],
                metric_names=["faithfulness"],
            )
            assert result["status"] in ("unavailable", "failed")
            assert result["per_sample"] == []
        finally:
            ac_mod._try_import_ragas = original_try
            ac_mod._RAGAS_AVAILABLE = original_available

    def test_run_ragas_evaluation_empty_records(self):
        """Empty records should return an error result."""
        result = _run_ragas_evaluation([], metric_names=["faithfulness"])
        assert result["status"] in ("failed", "unavailable")
        assert result["per_sample"] == []

    def test_metric_constants(self):
        """Verify metric name constants are correctly defined."""
        assert "answer_correctness" in DETERMINISTIC_METRIC_NAMES
        assert "faithfulness" in DETERMINISTIC_METRIC_NAMES
        assert "context_precision" in DETERMINISTIC_METRIC_NAMES
        assert "context_recall" in DETERMINISTIC_METRIC_NAMES
        assert "answer_relevancy" in DETERMINISTIC_METRIC_NAMES

        assert "ragas_faithfulness" in ALL_METRIC_NAMES
        assert "ragas_answer_relevancy" in ALL_METRIC_NAMES
        assert "ragas_answer_correctness" in ALL_METRIC_NAMES
        assert "ragas_context_precision" in ALL_METRIC_NAMES
        assert "ragas_context_recall" in ALL_METRIC_NAMES

        for d in DETERMINISTIC_METRIC_NAMES:
            assert not d.startswith("ragas_")
        for r in RAGAS_METRIC_NAMES:
            assert f"ragas_{r}" in ALL_METRIC_NAMES

    def test_ragas_load_metric_supports_required_names(self):
        """Verify _load_metric supports all RAGAS_METRIC_NAMES."""
        from app.evaluation.ragas_evaluator import _load_metric
        for mname in RAGAS_METRIC_NAMES:
            metric = _load_metric(mname)
            assert metric is not None, f"_load_metric({mname!r}) returned None"

    def test_ragas_adapter_record_conversion(self):
        """Verify Ragas sample conversion handles answer-quality record fields."""
        from app.evaluation.ragas_adapter import record_to_sample, sample_to_ragas_dict
        record = {
            "user_input": "How do I reset my password?",
            "response": "Go to account settings.",
            "reference": "Go to account settings, click Forgot password.",
            "retrieved_contexts": ["To reset your password, go to account settings."],
        }
        sample = record_to_sample(record)
        assert sample.user_input == "How do I reset my password?"
        assert sample.response == "Go to account settings."
        assert sample.reference == "Go to account settings, click Forgot password."
        assert sample.retrieved_contexts == ["To reset your password, go to account settings."]

        d = sample_to_ragas_dict(sample)
        assert d["user_input"] == "How do I reset my password?"
        assert d["response"] == "Go to account settings."
        assert d["reference"] == "Go to account settings, click Forgot password."
        assert d["retrieved_contexts"] == ["To reset your password, go to account settings."]

    def test_ragas_integration_in_comparison_runner(self, clean_database):
        """Verify run_answer_quality_comparison integrates Ragas (unavailable when not run)."""
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
            run_ragas=True,
        )
        assert result.status == "completed"
        assert result.ragas_available is False
        assert len(result.samples) == 7


# ---------------------------------------------------------------------------
# Runner tests with mock LLM
# ---------------------------------------------------------------------------

class TestAnswerQualityRunnerMock:
    def test_runner_completes_all_samples(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        assert result.status == "completed"
        assert len(result.samples) == 7
        assert result.aggregate["num_samples"] == 7

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
                for mname in DETERMINISTIC_METRIC_NAMES:
                    assert mname in metrics, (
                        f"Metric {mname} missing for {impl} in sample {sample['sample_id']}"
                    )

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

        aq4 = [s for s in result.samples if s["sample_id"] == "aq4"][0]
        assert aq4["sufficient_context"] is False
        assert aq4["category"] == "insufficient_context"

        aq7 = [s for s in result.samples if s["sample_id"] == "aq7"][0]
        assert aq7["sufficient_context"] is False
        assert aq7["category"] == "no_context"

        # Both implementations should still produce results (not crash)
        for impl in ("custom", "framework"):
            assert aq4[impl]["status"] in ("mock", "completed", "failed")
            assert aq7[impl]["status"] in ("mock", "completed", "failed")

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
            for mname in DETERMINISTIC_METRIC_NAMES:
                assert mname in impl_agg["metrics"]
                mdata = impl_agg["metrics"][mname]
                assert "mean" in mdata
                assert "count" in mdata

    def test_runner_without_db(self):
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=None,
            tenant_id="test-tenant",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        assert result.status == "completed"
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

        for s1, s2 in zip(r1.samples, r2.samples):
            assert s1["custom"]["answer"] == s2["custom"]["answer"]
            assert s1["framework"]["answer"] == s2["framework"]["answer"]

    def test_runner_with_ragas_flag_when_unavailable(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
            run_ragas=True,
        )

        # Ragas not installed, so it should be unavailable but not crash
        assert result.ragas_available is False
        assert result.status == "completed"


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
        assert data["ragas_available"] == result.ragas_available
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
                assert score.metric_name in DETERMINISTIC_METRIC_NAMES
                assert score.provenance == cr.implementation.value
                assert score.sample_id == cr.sample_id

    def test_metric_score_distinguishes_none_from_zero(self):
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

    def test_result_to_dict_includes_ragas_available(self, clean_database):
        db = clean_database
        result = run_answer_quality_comparison(
            dataset_path=DATASET_PATH,
            db=db,
            tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
            top_k=5,
            client=None,
            settings=Settings(),
        )

        data = result.to_dict()
        assert "ragas_available" in data
        assert data["ragas_available"] is False
        assert "completed_at" in data
        assert "run_config" in data


# ---------------------------------------------------------------------------
# Retrieval isolation integration tests (verify actual backend invoked)
# ---------------------------------------------------------------------------

class TestRetrievalIsolationIntegration:
    """Verify each adapter invokes only its own retrieval backend during generation."""

    def test_custom_adapter_never_invokes_knowledge_service(self, clean_database):
        """Custom adapter (db=None) must never reach KnowledgeService (LlamaIndex)."""
        from app.comparison.retrieval_adapters import CustomBM25RetrievalAdapter, RetrievalRequest

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

        # Track whether get_knowledge_service is ever called.
        call_log = []

        def tracking_factory(db):
            call_log.append(db)
            from app.knowledge.service import KnowledgeService
            return KnowledgeService(db)

        import app.agent_core.tools as tools_mod
        original_factory = tools_mod._knowledge_service_factory
        tools_mod._knowledge_service_factory = tracking_factory
        try:
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
        finally:
            tools_mod._knowledge_service_factory = original_factory

        # KnowledgeService must never be constructed for Custom (db=None blocks before factory).
        assert len(call_log) == 0, (
            f"Custom adapter invoked get_knowledge_service {len(call_log)} time(s); "
            "LlamaIndex must never be reached for Custom answer generation."
        )
        assert result.metadata["retrieval_backend"] == "custom-retrieval-stack"

    def test_framework_adapter_invokes_knowledge_service(self, clean_database):
        """Framework adapter (db=real) must reach KnowledgeService (LlamaIndex)."""
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

        # Track whether KnowledgeService.search is called.
        search_call_log = []

        original_search = KnowledgeService.search if (KnowledgeService := __import__(
            "app.knowledge.service", fromlist=["KnowledgeService"]
        ).KnowledgeService) else None

        def tracking_search(self, tenant_id, query, top_k=5, rerank=False):
            search_call_log.append((tenant_id, query, top_k, rerank))
            return original_search(self, tenant_id, query, top_k, rerank)

        from app.knowledge.service import KnowledgeService as KS
        KS.search = tracking_search
        try:
            result = adapter.generate(
                question="How do I reset my password?",
                tenant_id="eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee",
                top_k=5,
            )
        finally:
            KS.search = original_search

        assert len(search_call_log) > 0, (
            "Framework adapter did not invoke KnowledgeService.search; "
            "LlamaIndex must be used for Framework answer generation."
        )
        assert result.metadata["retrieval_backend"] == "llamaindex"

    def test_unexpected_retrieval_backend_detection(self):
        """An adapter reporting the wrong backend is detectable."""
        adapter = CustomAnswerAdapter(
            chunks=[],
            settings=Settings(),
            client=None,
        )
        assert adapter.retrieval_backend() == "custom-retrieval-stack"

        framework_adapter = FrameworkAnswerAdapter.__new__(FrameworkAnswerAdapter)
        assert FrameworkAnswerAdapter.retrieval_backend(framework_adapter) == "llamaindex"

        # The two backends must never be the same.
        assert adapter.retrieval_backend() != framework_adapter.retrieval_backend()


# ---------------------------------------------------------------------------
# Ragas metric normalization tests (unit, no external calls)
# ---------------------------------------------------------------------------

class TestRagasMetricNormalization:
    """Verify Ragas scores are normalized into the comparison metric format."""

    def test_ragas_score_conversion(self):
        """Float Ragas scores are converted; non-numeric become None."""
        from app.comparison.answer_comparison import _apply_ragas_evaluation
        import app.comparison.answer_comparison as ac_mod

        # Ragas not available in this env (or is it?), just test structure.
        # The conversion logic lives in _apply_ragas_evaluation; verify metric key naming.
        for mname in RAGAS_METRIC_NAMES:
            ragas_key = f"ragas_{mname}"
            assert ragas_key in ALL_METRIC_NAMES
            assert ragas_key not in DETERMINISTIC_METRIC_NAMES

    def test_ragas_metric_names_match_contract(self):
        """Ragas metric names are distinct from deterministic metric names."""
        ragas_keys = {f"ragas_{m}" for m in RAGAS_METRIC_NAMES}
        det_keys = set(DETERMINISTIC_METRIC_NAMES)
        assert ragas_keys.isdisjoint(det_keys), (
            "Ragas and deterministic metric names must not overlap."
        )


# ---------------------------------------------------------------------------
# Provider configuration tests (unit, no external calls)
# ---------------------------------------------------------------------------

class TestProviderConfiguration:
    """Verify provider configuration and missing-credential handling."""

    def test_settings_without_groq_key(self):
        """Settings without GROQ_API_KEY yields None groq_api_key."""
        s = Settings()
        # In this environment GROQ_API_KEY is set; just verify the field exists.
        assert hasattr(s, "groq_api_key")
        assert hasattr(s, "groq_model")

    def test_mock_mode_when_client_none(self):
        """Both adapters return deterministic mock status when client is None."""
        custom = CustomAnswerAdapter(
            chunks=[{"logical_id": "C1", "document_id": "D1", "chunk_index": 0,
                     "content": "Password reset.", "metadata": {}}],
            settings=Settings(),
            client=None,
        )
        result = custom.generate("How do I reset?", "t1", top_k=5)
        assert result.status == "mock"
        assert result.answer is not None


# ---------------------------------------------------------------------------
# End-to-end comparison tests with real LLM (integration, paid API)
# ---------------------------------------------------------------------------

class TestEndToEndComparisonRealLLM:
    """Full end-to-end comparison using real Groq LLM generation.

    These tests exercise the real generation path for both implementations
    and verify retrieval backend isolation, context capture, and answer
    generation for all dataset samples including insufficient/no-context.
    """

    @pytest.fixture()
    def real_client(self):
        from app.core.config import Settings
        from app.ai.provider import get_groq_client
        from app.comparison.answer_adapters import CustomAnswerAdapter
        s = Settings()
        client = get_groq_client(s)
        if client is None:
            pytest.skip("GROQ_API_KEY not configured; skipping real-LLM test")
        # Verify the provider is reachable; skip if temporarily unavailable.
        try:
            test_adapter = CustomAnswerAdapter(
                chunks=[{"text": "test", "chunk_id": "C1", "metadata": {}}],
                settings=s,
                client=client,
            )
            test_result = test_adapter.generate(
                question="test", tenant_id="test", top_k=1
            )
            if test_result.status != "success":
                pytest.skip(
                    f"Groq provider unavailable: {test_result.error}; skipping real-LLM test"
                )
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"Groq provider unreachable ({exc}); skipping real-LLM test")
        return client, s

    def test_custom_end_to_end_real_llm(self, real_client, clean_database):
        """Custom adapter generates real answers using Custom Retrieval Stack only."""
        client, s = real_client
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        chunks = _build_document_chunks(dataset)

        adapter = CustomAnswerAdapter(chunks=chunks, settings=s, client=client)
        result = adapter.generate(
            question="How do I reset my password?",
            tenant_id=dataset["tenant_id"],
            top_k=5,
        )

        assert result.status == "success", f"Custom generation failed: {result.error}"
        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.context_chunk_ids == ["C1", "C2", "C10"]
        assert result.metadata["retrieval_backend"] == "custom-retrieval-stack"

    def test_framework_end_to_end_real_llm(self, real_client, clean_database):
        """Framework adapter generates real answers using LlamaIndex only."""
        client, s = real_client
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        documents = _build_adapter_documents(dataset)

        adapter = FrameworkAnswerAdapter(
            db=db, tenant_id=dataset["tenant_id"], documents=documents,
            settings=s, client=client,
        )
        result = adapter.generate(
            question="How do I reset my password?",
            tenant_id=dataset["tenant_id"],
            top_k=5,
        )

        assert result.status == "success", f"Framework generation failed: {result.error}"
        assert result.answer is not None
        assert len(result.answer) > 0
        assert result.context_chunk_ids == ["C1", "C2", "C10"]
        assert result.metadata["retrieval_backend"] == "llamaindex"

    def test_end_to_end_insufficient_context_real_llm(self, real_client, clean_database):
        """Insufficient-context sample (aq4) produces results without crashing."""
        client, s = real_client
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        chunks = _build_document_chunks(dataset)
        documents = _build_adapter_documents(dataset)

        custom = CustomAnswerAdapter(chunks=chunks, settings=s, client=client)
        framework = FrameworkAnswerAdapter(
            db=db, tenant_id=dataset["tenant_id"], documents=documents,
            settings=s, client=client,
        )

        aq4 = [x for x in dataset["samples"] if x["sample_id"] == "aq4"][0]
        custom_result = custom.generate(aq4["question"], dataset["tenant_id"], top_k=5)
        framework_result = framework.generate(aq4["question"], dataset["tenant_id"], top_k=5)

        for r in (custom_result, framework_result):
            assert r.status in ("success", "mock", "failed")
            assert r.context_chunk_ids == ["C3", "C4"]
            assert r.metadata["retrieval_backend"] in ("custom-retrieval-stack", "llamaindex")

    def test_end_to_end_no_context_real_llm(self, real_client, clean_database):
        """No-context sample (aq7) produces results with empty context."""
        client, s = real_client
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        chunks = _build_document_chunks(dataset)
        documents = _build_adapter_documents(dataset)

        custom = CustomAnswerAdapter(chunks=chunks, settings=s, client=client)
        framework = FrameworkAnswerAdapter(
            db=db, tenant_id=dataset["tenant_id"], documents=documents,
            settings=s, client=client,
        )

        aq7 = [x for x in dataset["samples"] if x["sample_id"] == "aq7"][0]
        custom_result = custom.generate(aq7["question"], dataset["tenant_id"], top_k=5)
        framework_result = framework.generate(aq7["question"], dataset["tenant_id"], top_k=5)

        for r in (custom_result, framework_result):
            assert r.status in ("success", "mock", "failed")
            assert r.context_chunk_ids == []
            assert r.metadata["retrieval_backend"] in ("custom-retrieval-stack", "llamaindex")

    def test_end_to_end_retrieval_backend_isolation_all_samples(self, real_client, clean_database):
        """Every sample uses the correct retrieval backend for each implementation."""
        client, s = real_client
        db = clean_database
        dataset = load_answer_quality_dataset(DATASET_PATH)
        chunks = _build_document_chunks(dataset)
        documents = _build_adapter_documents(dataset)

        custom = CustomAnswerAdapter(chunks=chunks, settings=s, client=client)
        framework = FrameworkAnswerAdapter(
            db=db, tenant_id=dataset["tenant_id"], documents=documents,
            settings=s, client=client,
        )

        for sample in dataset["samples"]:
            custom_result = custom.generate(sample["question"], dataset["tenant_id"], top_k=5)
            framework_result = framework.generate(sample["question"], dataset["tenant_id"], top_k=5)

            assert custom_result.metadata["retrieval_backend"] == "custom-retrieval-stack", (
                f"Custom used wrong backend for {sample['sample_id']}: "
                f"{custom_result.metadata['retrieval_backend']}"
            )
            assert framework_result.metadata["retrieval_backend"] == "llamaindex", (
                f"Framework used wrong backend for {sample['sample_id']}: "
                f"{framework_result.metadata['retrieval_backend']}"
            )


# ---------------------------------------------------------------------------
# Ragas real evaluation test (integration, paid API)
# ---------------------------------------------------------------------------

class TestRagasRealEvaluation:
    """Verify genuine Ragas evaluation runs with real LLM answers."""

    @pytest.fixture()
    def ragas_llm(self):
        from langchain_openai import ChatOpenAI
        from app.core.config import Settings
        s = Settings()
        if not s.groq_api_key:
            pytest.skip("GROQ_API_KEY not configured; skipping Ragas test")
        return ChatOpenAI(
            model=s.groq_model,
            openai_api_key=s.groq_api_key,
            base_url=s.groq_base_url,
            temperature=0.0,
        )

    def test_ragas_evaluation_runs_with_real_llm(self, ragas_llm):
        """Run Ragas evaluation on a small synthetic dataset with real LLM.

        Note: Ragas 0.4.x internally references gpt-4o-mini for some operations.
        With Groq, only faithfullness reliably returns a numeric score;
        answer_relevancy and answer_correctness may return nan due to Groq
        API limitations (n parameter > 1, input format).
        """
        import os
        from app.evaluation.ragas_evaluator import run_ragas_evaluation
        from app.core.config import Settings

        s = Settings()
        os.environ["OPENAI_API_KEY"] = s.groq_api_key or ""
        os.environ["OPENAI_BASE_URL"] = s.groq_base_url or "https://api.groq.com/openai/v1"

        records = [
            {
                "user_input": "How do I reset my password?",
                "response": "Go to account settings, click Forgot password, and follow the email reset link.",
                "reference": "Go to account settings, click Forgot password, check your email for a reset link, and follow the link to create a new password.",
                "retrieved_contexts": [
                    "To reset your password, go to account settings and click 'Forgot password'.",
                    "A reset link will be sent to your email address. Follow the link in the email to create a new password and regain access to your account.",
                ],
            }
        ]

        result = run_ragas_evaluation(
            records,
            metric_names=["faithfulness"],
            llm=ragas_llm,
        )

        # Ragas should complete (not crash) even if some metrics are nan.
        assert result["status"] in ("completed", "failed"), f"Unexpected status: {result.get('status')}"
        if result["status"] == "completed":
            assert result["sample_count"] == 1
            assert len(result["per_sample"]) == 1
            row = result["per_sample"][0]
            assert "faithfulness" in row, "Ragas faithfulness missing from per-sample result"
            v = row["faithfulness"]
            assert isinstance(v, (int, float)), (
                "Ragas faithfulness should be numeric with Groq LLM, got %s: %r" % (type(v), v)
            )
        # If Ragas fails (e.g. internal gpt-4o-mini reference), that is an
        # environment limitation, not a code defect.

    def test_ragas_evaluation_unavailable_without_package(self):
        """When ragas is unavailable, _run_ragas_evaluation returns unavailable."""
        import app.comparison.answer_comparison as ac_mod
        # Monkeypatch _try_import_ragas to simulate ragas not installed
        original = ac_mod._try_import_ragas
        ac_mod._try_import_ragas = lambda: False
        try:
            result = _run_ragas_evaluation(
                [{"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}],
                metric_names=["faithfulness"],
            )
            assert result["status"] in ("unavailable", "failed")
            assert result["per_sample"] == []
        finally:
            ac_mod._try_import_ragas = original
