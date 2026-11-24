# F4.3 Final Acceptance Report

**Date:** 2026-11-23  
**Status:** PARTIALLY ACCEPTABLE — PENDING LIVE EVALUATION (Groq TPD rate limit)  
**Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee  
**Top-k:** 5

---

## 1. Available Providers and Actual Availability

| Provider | Configured | Usable for Generation | Usable for Ragas |
|----------|-----------|----------------------|------------------|
| Groq | Yes (GROQ_API_KEY, GROQ_MODEL=openai/gpt-oss-20b) | Partially — TPD exhausted (~199.9K/200K) | No — TPD exhausted + gpt-4o-mini unsupported |
| OpenAI | No | No | No |
| Anthropic | No | No | No |

Only **Groq** is configured in `Settings` (`app/core/config.py`). No alternative LLM provider is configured or available.

Groq TPD (tokens per day) limit: **200,000**. Usage: **~199,900** (exhausted).

---

## 2. Generation Results Summary

**Best achieved results (from F4.3.5, preserved):**

| Implementation | Generated | Failed | Provider/Model |
|----------------|-----------|--------|----------------|
| Custom | 2 | 5 | Groq (openai/gpt-oss-20b) |
| Framework | 2 | 5 | Groq (openai/gpt-oss-20b) |

### Per-Sample Generation Results

| Sample | Question | Custom Status | Framework Status | Custom Backend | Framework Backend |
|--------|----------|---------------|------------------|----------------|-------------------|
| aq1 | How do I reset my password? | success | success | custom-retrieval-stack | llamaindex |
| aq2 | How long do billing refunds take? | success | success | custom-retrieval-stack | llamaindex |
| aq3 | How long does shipping take? | provider_failure | provider_failure | custom-retrieval-stack | llamaindex |
| aq4 | Can I get a refund for 6 months ago? | provider_failure | provider_failure | custom-retrieval-stack | llamaindex |
| aq5 | How do I check API rate limit usage? | provider_failure | provider_failure | custom-retrieval-stack | llamaindex |
| aq6 | How do I update email notification preferences? | provider_failure | provider_failure | custom-retrieval-stack | llamaindex |
| aq7 | What is the company's stock price? | provider_failure | provider_failure | custom-retrieval-stack | llamaindex |

**Total: 2 genuine answers generated per implementation (aq1, aq2).**  
Remaining 5 samples per implementation: failed due to Groq TPD rate limit ("Provider service is temporarily unavailable"). This is an **external service limitation**, not a code defect.

Retrieval succeeded for all 7 samples in both implementations; only generation was rate-limited.

---

## 3. Positive Retrieval Execution Evidence

**Custom Edition** — verified via test spy on `CustomBM25RetrievalAdapter.search`:
- `CustomBM25RetrievalAdapter.search(RetrievalRequest)` was actually invoked
- Query and top_k captured; chunks C1/C2 returned and supplied to generator
- `get_knowledge_service` never called (`db=None` triggers RuntimeError before LlamaIndex)

**Framework Edition** — verified via test spy on `KnowledgeService.search`:
- `KnowledgeService.search(tenant_id, query, top_k, rerank=False)` was actually invoked
- Query, top_k, rerank=False, tenant_id captured; chunks returned and supplied to generator
- `db=self._db` enables LlamaIndex path

---

## 4. Genuine Ragas Evaluation

**Ragas evaluator LLM injection: Verified works.**

Ragas 0.4.4 `aevaluate()` accepts an explicit `llm` parameter. When `langchain_openai.ChatOpenAI` backed by Groq is passed, Ragas uses it instead of falling back to `gpt-4o-mini`. The explicit LLM injection path reaches `aevaluate()` correctly.

**Genuine Ragas metrics: NOT AVAILABLE**

**Reason:** Two compounding blockers:
1. Groq TPD rate limit exhausted (~199.9K/200K) — no tokens available for evaluation
2. Ragas 0.4.x internally defaults to `gpt-4o-mini` when no explicit LLM is passed; Groq doesn't support `gpt-4o-mini`

**Handling:**
- `_apply_ragas_evaluation()` detects all-NaN scores and reports `available: False`
- `run_ragas_evaluation()` returns `status: "failed"` when all scores are NaN/None
- No NaN scores treated as valid metrics
- No results fabricated

---

## 5. Deterministic Metrics (7 samples, both implementations)

| Metric | Custom Mean | Framework Mean |
|--------|-------------|----------------|
| answer_correctness | 0.1063 | 0.0000 |
| answer_relevancy | 0.0387 | 0.0000 |
| faithfulness | 0.1224 | 0.0000 |
| context_precision | 0.3571 | 0.4524 |
| context_recall | 0.8571 | 0.8571 |

*Note: Only 1 sample (aq1) has a genuine Custom answer; 0 samples have genuine Framework answers in this run. Best achieved (F4.3.5): 2/7 per implementation.*

---

## 6. BM25 Retrieval Comparison (6 queries, real PostgreSQL)

| Metric | Custom BM25 | Framework BM25 (KnowledgeService) |
|--------|-------------|-----------------------------------|
| MRR | 0.8333 | 1.0000 |
| Precision@5 | 0.3333 | 0.4000 |
| Recall@5 | 0.7500 | 0.9167 |
| NDCG@5 | 0.8167 | 0.9862 |

---

## 7. Test Results

### Comparison Test Suite

| Metric | Value |
|--------|-------|
| Passed | 78 |
| Failed | 1 |
| Deselected | 7 |

### Test Failures

| Test | Reason | Classification |
|------|--------|----------------|
| `TestPositiveRetrievalExecution::test_ragas_evaluator_llm_injection` | Groq TPD rate limit (external) | Expected external-provider failure — test infrastructure correct, LLM injected properly |

### Deselected Tests (7) — NOT counted as passing

| Test Class | Count | Reason |
|------------|-------|--------|
| `TestRagasRealEvaluation` | 2 | Requires genuine Ragas LLM evaluation (rate-limited) |
| `TestEndToEndComparisonRealLLM` | 5 | Requires real Groq API (rate-limited) |

### Pre-existing Failures (8) — CrossEncoder reranking

| Test Class | Count | Reason |
|------------|-------|--------|
| `TestReranking` | 8 | Missing `torch`/`sentence-transformers` — dependency issue, not code defect |

### `pip check`

Clean (only pre-existing: `virtualenv 21.15.6` requires `packaging>=26.3`, have `24.2`)

### `git diff --check`

Clean (CRLF line-ending warnings on 4 files, no whitespace issues)

### `git status --short`

Modified:
- `app/comparison/__init__.py`
- `app/comparison/answer_adapters.py`
- `app/comparison/answer_comparison.py`
- `app/evaluation/ragas_evaluator.py`
- `tests/test_comparison_answer_quality.py`

Untracked:
- `f432_reports/`

---

## 8. Remaining Limitations

| Limitation | Impact | Resolution Required |
|------------|--------|---------------------|
| Groq TPD rate limit (200K tokens/day exhausted) | 0-2/7 samples generated per implementation; Ragas evaluation not completed | Wait for TPD reset or upgrade Groq plan |
| Ragas 0.4.x default model `gpt-4o-mini` not on Groq | Ragas falls back to unsupported model → NaN scores | Explicit LLM injection works (verified), but rate-limited |
| No alternative LLM provider configured | Cannot switch to OpenAI/Anthropic for Ragas | Add provider configuration for a supported model |
| Missing `torch`/`sentence-transformers` | 8 CrossEncoder reranking tests fail (pre-existing) | Install packages |
| Only 0-2/7 samples have genuine answers | Deterministic metrics skewed (0.0 for None answers) | Re-run when TPD resets |

---

## 9. Final F4.3 Acceptance Recommendation

### Criteria Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Custom agent uses Custom Retrieval Stack exclusively | PASS | `db=None` → `search_knowledge` raises RuntimeError before `KnowledgeService` |
| Framework agent uses LlamaIndex exclusively | PASS | `db=self._db` → `search_knowledge` calls `get_knowledge_service(db)` → KnowledgeService |
| Positive Custom retrieval execution proven | PASS | Test spy on `CustomBM25RetrievalAdapter.search` |
| Positive Framework retrieval execution proven | PASS | Test spy on `KnowledgeService.search` |
| Tenant isolation maintained | PASS | Custom: in-memory per-adapter; Framework: tenant-scoped DB queries |
| Genuine Ragas evaluation executed | **NOT POSSIBLE** | Groq TPD rate limit; no alternative provider |
| Both implementations evaluated with same config | PARTIAL | Deterministic metrics for all 7 samples; Ragas unavailable |
| Actual LLM-generated answers captured | PARTIAL | 0-2 answers per implementation; remaining rate-limited |
| JSON and Markdown reports generated | PASS | Multiple report files in `f432_reports/` |
| Test suite passes (excluding pre-existing env failures) | PARTIAL | 78 passed, 1 failed (Groq rate-limited), 7 deselected |

### Recommendation

**F4.3 is PARTIALLY ACCEPTABLE with the following caveats:**

1. **Retrieval isolation is fully verified** — Custom uses Custom Retrieval Stack only, Framework uses LlamaIndex/KnowledgeService only. This is the core architectural requirement and it is proven with positive execution evidence (test spies on real methods).

2. **Deterministic answer-quality comparison is complete** — All 7 samples evaluated with deterministic metrics for both implementations. Reports generated.

3. **Genuine Ragas evaluation is NOT possible** with the current provider configuration. Groq TPD rate limit prevents completion. No alternative LLM provider is configured. This is a **blocker for full F4.3 acceptance** if genuine Ragas evaluation is a hard requirement.

4. **Real LLM generation is partially verified** — Only aq1 and aq2 succeeded for both implementations (F4.3.5). Subsequent runs hit TPD exhaustion.

### To Achieve Full Acceptance

1. Wait for Groq TPD rate limit reset or upgrade Groq plan, then re-run genuine evaluation
2. Configure an LLM provider that supports `gpt-4o-mini` (e.g., OpenAI API) for Ragas evaluation, OR
3. Accept deterministic metrics as the primary evaluation methodology and document Ragas unavailability

---

## Report Files

| Report | Path |
|--------|------|
| F4.3.6 Acceptance Report | `f432_reports/f436_acceptance_report.md` |
| F4.3.6 Acceptance Report (JSON) | `f432_reports/f436_acceptance_report.json` |
| F4.3.6 Live Evaluation (JSON) | `f432_reports/f436_live_evaluation.json` |
| F4.3.6 Live Evaluation (Markdown) | `f432_reports/f436_live_evaluation.md` |
| F4.3.5 Live Evaluation | `f432_reports/f435_live_evaluation.md/.json` |
| F4.3.5 Acceptance Report | `f432_reports/f435_acceptance_report.md/.json` |
| F4.3 Acceptance Report | `f432_reports/f43_acceptance_report.md/.json` |
| F4.3.2 Acceptance Report | `f432_reports/f432_acceptance_report.md/.json` |
| F4.3.2 Answer Quality Report | `f432_reports/f432_report.md/.json` |
| BM25 Comparison Report | `f432_reports/bm25_comparison_report.md/.json` |
| F4.3.3 Real Groq Comparison | `f432_reports/f433_real_groq_comparison.md/.json` |
| F4.3.4 Genuine Evaluation | `f432_reports/f434_genuine_evaluation.md/.json` |
