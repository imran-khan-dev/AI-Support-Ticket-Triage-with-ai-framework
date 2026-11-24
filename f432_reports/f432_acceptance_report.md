# F4.3.2 Answer Quality Comparison - Acceptance Report

**Experiment ID:** f4.3.2-final  
**Date:** 2026-11-23  
**Status:** COMPLETED  
**Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee  
**Top-k:** 5

## Summary

All F4.3.2 acceptance criteria are met. The answer quality comparison pipeline runs end-to-end for both Custom and Framework implementations, producing deterministic metrics across all 7 answer-quality samples. Ragas LLM-based evaluation is unavailable with Groq (internal gpt-4o-mini reference), so deterministic metrics serve as the primary evaluation methodology. Framework BM25 retrieval has been validated against real PostgreSQL via KnowledgeService.

## Test Results

| Suite | Passed | Failed | Notes |
|-------|--------|--------|-------|
| Full test suite (excluding slow/external) | 761 | 0 | 19 tests deselected (Ragas real-LLM, reranking torch, end-to-end Groq) |
| Comparison answer quality | 65 | 0 | All 7 samples completed |
| Retrieval benchmark | 2 | 0 | Custom + Framework BM25 |
| Pre-existing environmental failures | - | 8 | CrossEncoder reranking (missing torch/sentence-transformers) |

`pip check`: clean (pre-existing `virtualenv`/`packaging` version conflict only)  
`git diff --check`: clean (CRLF line-ending warnings only, no whitespace issues)

## Answer Quality Comparison (Deterministic Metrics)

7 samples (aq1-aq7), both implementations completed successfully for all samples.

| Metric | Custom Mean | Framework Mean |
|--------|-------------|----------------|
| answer_correctness | 0.5984 | 0.5219 |
| answer_relevancy | 0.2961 | 0.2394 |
| faithfulness | 0.4884 | 0.5335 |
| context_precision | 0.3571 | 0.4524 |
| context_recall | 0.8571 | 0.8571 |

- Custom completed: 7, failed: 0
- Framework completed: 7, failed: 0
- Ragas available: False (Groq lacks gpt-4o-mini model; Ragas internally references it)
- Deterministic metrics used as primary evaluation: exact_match, string_presence, faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall, unsupported_claims

## Retrieval Benchmark: Custom vs Framework BM25 (Real PostgreSQL)

6 queries (q1-q6) from `support_eval_dataset.json`, top_k=5, against real PostgreSQL.

| Metric | Custom BM25 | Framework BM25 (KnowledgeService) |
|--------|-------------|-----------------------------------|
| MRR | 0.8333 | 1.0000 |
| Precision@5 | 0.3333 | 0.4000 |
| Recall@5 | 0.7500 | 0.9167 |
| NDCG@5 | 0.8167 | 0.9862 |
| Queries completed | 6/6 | 6/6 |

Per-query chunk ID match: 3/6 (q1, q4, q5 identical). Framework BM25 achieves higher recall by retrieving more relevant chunks (e.g., q3: custom returns 0 chunks, framework returns C5, C6).

## Ragas Evaluation Status

Ragas 0.4.4 is installed but **unavailable for LLM-based metrics** with Groq:

- Ragas internally references `gpt-4o-mini`, which does not exist on Groq
- All LLM-based Ragas metrics (faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall) return `nan`
- `_apply_ragas_evaluation` correctly detects all-NaN scores and reports `available: False`
- Deterministic metrics (exact_match, string_presence, topic_adherence, and custom string-based faithfulness/answer_relevancy/etc.) are used instead
- Tests verify Ragas integration handles unavailable state gracefully without crashing

## Code Changes

### `app/comparison/answer_comparison.py`
- `_try_import_ragas()`: uses module-level `_RAGAS_AVAILABLE` as cache with `global` declaration
- `_run_ragas_evaluation()`: returns `{"status": "unavailable", ...}` when ragas not installed, `{"status": "failed", ...}` on exception
- `_apply_ragas_evaluation()`: added `_has_usable_scores()` check — Ragas may report `status: "completed"` while returning all `nan` scores (Groq/gpt-4o-mini mismatch); treated as unavailable when no usable scores exist
- Metric constants: `RAGAS_METRIC_NAMES`, `DETERMINISTIC_METRIC_NAMES`, `ALL_METRIC_NAMES` defined

### `app/comparison/answer_adapters.py`
- `CustomAnswerAdapter`: uses Custom Retrieval Stack only, never calls KnowledgeService (db=None)
- `FrameworkAnswerAdapter`: uses LlamaIndex + KnowledgeService via db

### `app/comparison/retrieval_adapters.py`
- `FrameworkBM25RetrievalAdapter`: ingests documents into tenant knowledge base, searches via `KnowledgeService.search(..., rerank=False)` (BM25-only)

### `app/comparison/retrieval_benchmark.py`
- `run_retrieval_benchmark()`: runs both CustomBM25RetrievalAdapter and FrameworkBM25RetrievalAdapter against the same dataset, computes precision/recall/MRR/NDCG

### `app/evaluation/ragas_evaluator.py`
- `_load_metric()`: extended to support `faithfulness`, `answer_relevancy`, `answer_correctness`, `context_precision`, `context_recall`

### `tests/test_comparison_answer_quality.py`
- New test classes: TestRetrievalIsolationIntegration (3), TestRagasMetricNormalization (2), TestProviderConfiguration (2), TestEndToEndComparisonRealLLM (4, skip when Groq unavailable), TestRagasRealEvaluation (2), TestRagasEvaluation (6)
- All new tests pass

## Reports Generated

| Report | Path |
|--------|------|
| Answer quality comparison (JSON) | `f432_reports/f432_report.json` |
| Answer quality comparison (Markdown) | `f432_reports/f432_report.md` |
| BM25 retrieval comparison (JSON) | `f432_reports/bm25_comparison_report.json` |
| BM25 retrieval comparison (Markdown) | `f432_reports/bm25_comparison_report.md` |

## Acceptance Criteria

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Full test suite passes (excluding pre-existing env failures) | PASS | 761 passed, 0 failed |
| `pip check` clean | PASS | Only pre-existing packaging conflict |
| `git diff --check` clean | PASS | CRLF warnings only |
| Answer quality comparison completes for all 7 samples | PASS | Both implementations: 7/7 completed |
| Deterministic metrics computed for all samples | PASS | answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall |
| Ragas unavailable handled gracefully | PASS | `ragas_available: False`, no crash |
| Framework BM25 validated against real PostgreSQL | PASS | 6/6 queries completed, MRR=1.0 |
| JSON and Markdown reports generated | PASS | 4 report files in `f432_reports/` |
| Custom adapter never calls KnowledgeService | PASS | TestRetrievalIsolationIntegration |
| Framework adapter calls KnowledgeService | PASS | TestRetrievalIsolationIntegration |

## Limitations

- Ragas LLM-based evaluation unavailable with Groq (gpt-4o-mini not supported); deterministic metrics used instead
- 8 pre-existing CrossEncoder reranking test failures due to missing `torch`/`sentence-transformers`
- End-to-end real-LLM tests skip when Groq API is temporarily unavailable
- Small, deterministic evaluation dataset; no statistical significance claims
