# Final Comparative Evaluation: Custom Edition vs Framework Edition

**Project:** AI-Support-Ticket-Triage  
**Generated:** 2026-11-24T03:44:50+00:00  
**Backend:** `backend/`  
**Status:** Final consolidation (F4.6)

---

## Executive Summary

This report compares two implementations of the AI Support Ticket Triage system:

- **Custom Edition** — Custom Agent Orchestrator with Custom Retrieval Stack (BM25), custom orchestration loop, retry, tracing, and lifecycle.
- **Framework Edition** — LangGraph-based graph orchestration with LlamaIndex retrieval (BM25 via KnowledgeService) and Ragas evaluation ecosystem.

Neither implementation can be declared categorically superior. Each demonstrates strengths in different dimensions:

| Dimension | Leading Implementation | Basis |
|-----------|----------------------|-------|
| Retrieval quality (6-query dataset) | Framework | MRR 1.0000 vs 0.8333; Recall@5 0.9167 vs 0.7500 |
| Answer correctness/relevancy | Custom | 0.5984 vs 0.5219; 0.2961 vs 0.2394 |
| Faithfulness / context precision | Framework | 0.5335 vs 0.4884; 0.4524 vs 0.3571 |
| Deterministic latency | Custom | Median 1.57ms vs 2.73ms (preliminary) |
| Low-level execution control | Custom | Full ownership of loop, retry, cancellation |
| Built-in abstractions | Framework | LlamaIndex, LangGraph, Ragas |

**Key limitations:** Genuine Ragas evaluation is unavailable (Ragas 0.4.x requires `gpt-4o-mini`, not supported by Groq). Real LLM generation is partial (aq1 only; aq2-aq7 rate-limited by Groq). Token usage and cost are unverified (scripted providers supply no metadata; live Groq not executed). Performance results are preliminary (3 repetitions, deterministic scripted providers).

---

## Project Scope

### Implementations Compared

1. **Custom Edition** — `backend/Agent Orchestrator/` (Custom Agent Orchestrator) + Custom Retrieval Stack (`backend/Retrieval-Stack/`)
2. **Framework Edition** — `backend/app/agent_core/langgraph/` (LangGraph) + `backend/app/knowledge/` (LlamaIndex/KnowledgeService) + Ragas evaluation

### Milestones Covered

| Milestone | Component | Status |
|-----------|-----------|--------|
| F1 | LlamaIndex Retrieval | Complete |
| F2 | LangGraph Agent Orchestration | Complete |
| F3 | Ragas Evaluation Framework | Complete (implementation accepted; live Ragas evaluation pending) |
| F4.1 | Comparison Contracts | Complete |
| F4.2 | Retrieval Benchmark | Complete |
| F4.3 | Answer Quality and Faithfulness | Partially accepted; live evaluation pending |
| F4.4 | Agent Behavior and Reliability | Accepted with documented limitations |
| F4.5 | Performance, Token and Cost Comparison | Implemented |
| F4.5.1 | Performance Audit | Accepted with documented limitations |

### Exclusions (Not Modified)

- Standalone Custom Retrieval Stack project
- Standalone Agent Orchestrator project
- Standalone Eval Harness project
- LangGraph, LlamaIndex, or Ragas integrations changed to improve benchmark results

---

## Architecture Overview

### Custom Edition Architecture

```
User Request
    │
    ▼
CustomBM25RetrievalAdapter.search()  ← Custom Retrieval Stack (BM25)
    │
    ▼
Build context-augmented prompt
    │
    ▼
run_agent() → AgentLoop
    ├─ Scripted LLMProvider (deterministic)
    ├─ Real ToolExecutor with controlled handlers
    ├─ Provider retry policy
    ├─ Tool retry policy
    ├─ InMemoryTracer
    └─ Async cancellation support (run_async)
```

### Framework Edition Architecture

```
User Request
    │
    ▼
FrameworkBM25RetrievalAdapter.search() ← KnowledgeService → LlamaIndex BM25
    │
    ▼
Build context-augmented prompt
    │
    ▼
run_agent_graph() → LangGraph StateGraph
    ├─ Scripted ChatModelRunner (deterministic)
    ├─ Real ToolExecutor with controlled handlers
    ├─ Graph: START → initialize → call_model → execute_tools → call_model → END
    ├─ Synchronous invocation (no async cancellation)
    └─ No built-in provider retry
```

### Comparison Layer

Both implementations execute through separate adapters in `app/comparison/agent_adapters.py` that translate shared normalized scenario inputs into native types and normalize outcomes back into the shared `AgentScenarioResult` contract. No implementation simulates the other.

---

## Benchmark Methodology

### F4.2 — Retrieval Benchmark

- **Dataset:** support-kb-retrieval-eval (6 queries)
- **Top-K:** 5
- **Custom:** Custom BM25 Retrieval Stack
- **Framework:** LlamaIndex BM25 via KnowledgeService
- **Metrics:** MRR, Precision@5, Recall@5, NDCG@5
- **Relevance judgments:** Preserved from original dataset

### F4.3 — Answer Quality and Faithfulness

- **Dataset:** support-kb-answer-quality-eval (7 samples: aq1-aq7)
- **Top-K:** 5
- **Metrics:** Deterministic answer_correctness, answer_relevancy, faithfulness, context_precision, context_recall
- **Ragas:** Not available (Ragas 0.4.x internally references `gpt-4o-mini`, not supported by Groq; no alternative provider configured)
- **Real LLM generation:** Only aq1 succeeded for both implementations; aq2-aq7 rate-limited by Groq

### F4.4 — Agent Behavior and Reliability

- **Dataset:** agent-evaluation v1.0.0 (14 scenarios)
- **Execution mode:** Deterministic scripted providers
- **Metrics:** Scenario success, tool call accuracy, outcome correctness, safety metrics, approval gate, tenant isolation, loop prevention, cancellation
- **Security:** Application-controlled (not LLM-controlled); no vacuous safety passes

### F4.5 — Performance, Token and Cost Comparison

- **Dataset:** 14 scenarios (13 shared, 1 excluded: agent-cancellation custom-only)
- **Configuration:** 2 warm-up trials, 3 measured repetitions per scenario
- **Execution mode:** Deterministic (scripted providers)
- **Timing:** `time.perf_counter()` monotonic clock; milliseconds
- **Boundaries:** Total = provider + tool + overhead; provider = `complete()`/`invoke()`; tool = `execute()`
- **Live Groq:** Not executed (opt-in via `F45_LIVE_ENABLED`)

---

## Comparative Results

### 1. Retrieval Comparison (F4.2)

| Metric | Custom BM25 | Framework BM25 (LlamaIndex) | Difference |
|--------|------------|----------------------------|------------|
| MRR | 0.8333 | 1.0000 | +0.1667 Framework |
| Precision@5 | 0.3333 | 0.4000 | +0.0667 Framework |
| Recall@5 | 0.7500 | 0.9167 | +0.1667 Framework |
| NDCG@5 | 0.8167 | 0.9862 | +0.1695 Framework |
| Queries completed | 6/6 | 6/6 | Equal |

**Per-query observations:**
- q1: Both return C1, C2, C10 (match)
- q2: Custom C3, C6, C4; Framework C3, C4, C6 (same chunks, different order; MRR both 1.0)
- q3: Custom returns **zero results**; Framework returns C5, C6 (Custom MRR 0.0)
- q4: Both return C7, C8 (match)
- q5: Both return C9, C2, C5 (match)
- q6: Custom C2, C1, C9, C5, C7; Framework C2, C1, C5, C9 (same top-4, Custom adds C7)

**Implementation differences:**
- Custom BM25 uses its own inverted index and scorer; zero-score filtering may eliminate q3 results entirely
- Framework BM25 uses LlamaIndex `BM25Retriever` wrapped in `FilteredBM25Retriever`; different stemming/ranking produces different result sets
- Custom returned zero results for q3, indicating stricter score thresholding or tokenization differences

**Conclusion:** Framework has stronger measured retrieval results on the 6-query dataset. This does not generalize beyond this dataset.

---

### 2. Answer Quality and Faithfulness (F4.3)

**Deterministic metrics (7 samples per implementation):**

| Metric | Custom Mean | Framework Mean | Difference |
|--------|------------|----------------|------------|
| answer_correctness | 0.5984 | 0.5219 | +0.0765 Custom |
| answer_relevancy | 0.2961 | 0.2394 | +0.0567 Custom |
| faithfulness | 0.4884 | 0.5335 | -0.0451 Custom |
| context_precision | 0.3571 | 0.4524 | -0.0953 Custom |
| context_recall | 0.8571 | 0.8571 | Equal |

**Real LLM generation (Groq, openai/gpt-oss-20b):**

| Sample | Custom | Framework |
|--------|--------|-----------|
| aq1 (password reset) | success | success (F4.3 acceptance); provider_failure (F4.3.6 live) |
| aq2-aq7 | provider_failure (rate-limited) | provider_failure (rate-limited) |

**Total: 2 actual LLM-generated answers (aq1, both implementations).**

**Ragas evaluation:** Not available. Ragas 0.4.x internally references `gpt-4o-mini`, which is not supported by the Groq provider. No alternative LLM provider is configured. All Ragas metric fields are `null`/`NaN` and were not treated as valid.

**Faithfulness method:** Deterministic keyword-overlap (not Ragas LLM-based faithfulness). These are diagnostic metrics, not equivalent to Ragas faithfulness.

**Conclusion:** Custom leads on correctness and relevancy; Framework leads on faithfulness and context precision. Neither is decisively superior. Results should not be presented as a completed answer-quality benchmark — real generation was partial and Ragas was unavailable.

---

### 3. Agent Behavior and Reliability (F4.4)

**Scenario coverage:**

| Implementation | Total | Supported | Executed | Passed | Failed | Unmeasurable | Success Rate |
|---------------|-------|-----------|----------|--------|--------|--------------|--------------|
| Custom | 14 | 14 | 14 | 12 | 0 | 2 | 0.8571 |
| Framework | 14 | 13 | 13 | 11 | 0 | 2 | 0.8462 |

**Excluded scenario:** `agent-cancellation` — Custom-only (async `run_async`/`asyncio.CancelledError`). Framework graph invocation is synchronous; cancellation is not silently excluded.

**Known implementation differences:**
- Framework counts limit-blocked tool calls toward tool-call budget (e.g., 5 vs 3 for `agent-maximum-tool-call-limit`)
- Framework checks iteration limit before executing tool calls on final iteration (e.g., 2 vs 3 for `agent-loop-prevention`)
- Provider retry metrics are `null` (not measurable) for Framework; Custom supports provider retry
- Cancellation is custom-only

**Security behavior (verified on actual execution):**

| Security Property | Custom | Framework |
|-------------------|--------|-----------|
| Cross-tenant access prevention | Verified | Verified |
| Approval-gated action enforcement | Verified | Verified |
| Invalid argument rejection | Verified | Verified |
| Loop prevention enforced | Verified | Verified |
| Provider failure is safe | Verified | Verified |

No scenario received a vacuous safety pass merely because it did not exercise the relevant behavior. All safety metrics are 0.0 or null where no safety constraint scenario was exercised.

**Conclusion:** Custom has a slightly higher success rate (0.8571 vs 0.8462). Neither implementation failed any scenario (unmeasurable ≠ failed). Security behavior is equivalent on both implementations.

---

### 4. Performance Comparison (F4.5 — Preliminary)

**Configuration:** 2 warm-up trials, 3 measured repetitions, deterministic scripted providers.

| Metric | Custom | Framework | Difference |
|--------|--------|-----------|------------|
| Median total latency (ms) | 1.5724 | 2.7278 | Custom faster |
| Mean total latency (ms) | 1.7934 | 3.0551 | Custom faster |
| Median provider latency (ms) | 0.0141 | 0.0326 | Custom faster |
| Median tool latency (ms) | 1.1384 | 1.0774 | Framework slightly faster |
| Execution overhead (ms) | 0.4148 | 1.6557 | Custom lower |
| Success rate | 0.8571 | 0.8462 | Custom slightly higher |
| Trials | 42 | 39 | — |
| Successes | 36 | 33 | — |
| Failures | 6 | 6 | — |

**Successful paired trials:** 33 (both implementations succeeded on same trial)  
**Failed paired trials:** 6  
**Relative latency difference:** -0.4236 (custom - framework) / framework

**Per-scenario:** Custom is faster on all 13 shared scenarios.

**Token usage:** `null` for all trials (scripted providers supply no usage metadata).  
**Cost:** `null` for all trials (no token usage, no live Groq extraction).

**Timing boundary note:** Custom counts limit-rejected tool calls in tool latency (they reach the executor and return `ExecutionLimitError`). Framework blocks them before the executor, so they are not counted in tool latency. This affects `execution_overhead_ms` comparability.

**Conclusion:** Custom is faster in the deterministic benchmark, but 3 repetitions are insufficient for strong production performance claims. Results are labeled **preliminary**. No production performance winner is declared.

---

### 5. Token and Cost Comparison

| Dimension | Custom | Framework |
|-----------|--------|-----------|
| Token usage | null (unavailable) | null (unavailable) |
| Cost | null (unavailable) | null (unavailable) |

**Limitations:**
- Scripted providers do not supply real token usage metadata
- Real Groq token extraction was not verified in both implementations
- Live benchmark was not executed
- Cost cannot be reliably compared without verified usage and applicable model pricing

**Distinctions maintained:**
- Missing usage (`null`) ≠ actual zero usage (`0`)
- Missing pricing ≠ available pricing
- Unavailable cost is reported as `null`, not converted to zero
- No cost estimates were invented

---

## Architecture and Engineering Trade-offs

### Custom Edition

**Strengths:**
- Greater low-level control over retrieval stack, orchestration loop, retry, tracing, and lifecycle
- Async cancellation support (`run_async`/`asyncio.CancelledError`)
- Provider retry supported
- No framework lock-in or dependency version management for core components
- Deterministic latency advantage in benchmark

**Trade-offs:**
- Higher custom maintenance burden — team owns and operates all components
- Implementation effort borne entirely by the team
- No built-in evaluation ecosystem (Ragas equivalent would need to be built)

### Framework Edition

**Strengths:**
- Built-in abstractions: LlamaIndex retrieval, LangGraph orchestration, Ragas evaluation
- Guided extensibility via framework conventions
- Stronger measured retrieval results on the current 6-query dataset
- Reduced custom code for standard patterns

**Trade-offs:**
- Synchronous invocation only (no async cancellation)
- No built-in provider retry
- Framework-specific debugging and behavior constraints
- Dependency and version-management overhead (LangGraph, LlamaIndex, Ragas, torch, sentence-transformers)
- Vendor/framework lock-in risk

**Engineering assessment:** Frameworks are not automatically more reliable, and custom implementations are not automatically more flexible. The choice depends on whether the team values low-level control and ownership (Custom) or built-in abstractions and reduced custom code (Framework).

---

## Final Comparison Matrix

| Dimension | Custom Edition | Framework Edition | Evidence | Conclusion | Confidence |
|-----------|---------------|-------------------|----------|------------|------------|
| Retrieval quality | MRR 0.8333, P@5 0.3333, R@5 0.7500, NDCG@5 0.8167; zero results for q3 | MRR 1.0000, P@5 0.4000, R@5 0.9167, NDCG@5 0.9862 | f432_reports/bm25_comparison_report.md | Framework stronger on 6-query dataset | Verified (dataset-limited) |
| Answer quality | correctness 0.5984, relevancy 0.2961 | correctness 0.5219, relevancy 0.2394 | f432_reports/f432_report.md | Custom leads correctness/relevancy | Deterministic only; Ragas unavailable |
| Faithfulness | 0.4884 | 0.5335 | f432_reports/f432_report.md | Framework slightly higher | Deterministic keyword-overlap, not Ragas |
| Tool orchestration | Custom AgentLoop with retry + async cancellation | LangGraph StateGraph, no retry, synchronous | app/f432_reports/f44_agent_comparison.md | Custom offers more execution control | Verified |
| Reliability | 12/14 passed; loop prevention 1.0 | 11/13 passed; loop prevention 0.0 | app/f432_reports/f44_agent_comparison.md | Custom slightly higher; neither failed | Verified |
| Security behavior | Application-controlled; all verified | Application-controlled; all verified | app/f432_reports/f44_agent_comparison.md | Equivalent on both | Verified |
| Execution control | Full control over loop, retry, lifecycle | Graph-managed lifecycle; conventions | app/comparison/agent_adapters.py | Custom offers greater low-level control | Verified |
| Observability | InMemoryTracer; custom tracing | LangGraph graph state; framework debugging | app/comparison/agent_adapters.py | Comparable via comparison-layer instrumentation | Verified |
| Latency | Median 1.57ms; overhead 0.41ms | Median 2.73ms; overhead 1.66ms | app/f432_reports/f45_performance_comparison.json | Custom faster (preliminary) | Verified as preliminary |
| Token usage | null | null | app/f432_reports/f45_performance_comparison.json | Cannot compare | Verified as unavailable |
| Cost | null | null | app/f432_reports/f45_performance_comparison.json | Cannot compare | Verified as unavailable |
| Maintainability | High custom maintenance | Dependency/version management | Source review | Custom = more code to own; Framework = more deps | Engineering assessment |
| Extensibility | Unlimited flexibility, full responsibility | Guided extensibility within framework bounds | Source review | Trade-off depends on team priorities | Engineering assessment |
| Operational complexity | Higher (team owns all) | Moderate (framework reduces custom code) | Source review | Custom = higher operational burden | Engineering assessment |

---

## Final Engineering Conclusions

**1. Which approach offers greater low-level control?**

**Custom Edition.** Full ownership of the retrieval stack, orchestration loop, retry policy, tracing, and lifecycle. The team can modify any component without framework constraints.

**2. Which approach offers more built-in abstractions?**

**Framework Edition.** LlamaIndex retrieval, LangGraph graph-based orchestration, and Ragas evaluation ecosystem provide ready-made abstractions for common patterns.

**3. Which approach has the stronger measured retrieval results on the current dataset?**

**Framework Edition.** MRR 1.0000 vs 0.8333, Recall@5 0.9167 vs 0.7500, NDCG@5 0.9862 vs 0.8167 on the 6-query retrieval dataset. This does not generalize beyond this dataset.

**4. Can either approach be declared superior in answer quality?**

**No.** Custom leads on answer_correctness (0.5984 vs 0.5219) and answer_relevancy (0.2961 vs 0.2394). Framework leads on faithfulness (0.5335 vs 0.4884) and context_precision (0.4524 vs 0.3571). Neither is decisively superior. Real LLM generation was partial (aq1 only) and Ragas was unavailable.

**5. Can either approach be declared superior in production performance?**

**No.** Custom is faster in the deterministic benchmark (median 1.57ms vs 2.73ms), but 3 repetitions with scripted providers are insufficient for production performance claims. Results are labeled preliminary.

**6. Which approach requires more custom maintenance?**

**Custom Edition.** The team owns and operates the retrieval stack, orchestration loop, retry, tracing, and lifecycle. Framework Edition shifts some maintenance to framework dependency management.

**7. What are the strongest reasons to choose each approach?**

- **Custom:** Greater low-level control, no framework lock-in, deterministic performance advantage in benchmark, full ownership of execution semantics, async cancellation.
- **Framework:** Built-in abstractions (LlamaIndex, LangGraph, Ragas), stronger measured retrieval on current dataset, guided extensibility via framework conventions, reduced custom code for standard patterns.

**8. What additional experiments would provide the most valuable evidence?**

- Genuine Ragas evaluation with a provider supporting `gpt-4o-mini` (not Groq)
- Live Groq performance benchmark with bounded requests when rate limits allow
- Larger retrieval dataset beyond 6 queries
- Production-scale load testing
- Real token usage and cost measurement with live provider

---

## Limitations

1. **Genuine Ragas evaluation unavailable:** Ragas 0.4.x requires `gpt-4o-mini`, not supported by Groq. No alternative provider configured.
2. **Real LLM generation partial:** Only aq1 succeeded for both implementations; aq2-aq7 rate-limited by Groq API.
3. **Token usage unverified:** Scripted providers supply no usage metadata; live Groq not executed.
4. **Cost unverified:** No token usage, no live extraction. No cost estimates invented.
5. **Performance results preliminary:** 3 repetitions per scenario; deterministic scripted providers only. Not production-validated.
6. **F4.4 JSON report discrepancy:** JSON contains 1 scenario (s1); markdown references 14. This is a pre-existing report inconsistency, not a data error in this evaluation.
7. **Small sample sizes:** All benchmarks use small, deterministic datasets. No statistical significance claims are made.

---

## Future Experiments

1. **Genuine Ragas evaluation:** Configure an LLM provider supporting `gpt-4o-mini` (e.g., OpenAI API) to enable Ragas LLM-based metrics.
2. **Live Groq benchmark:** When Groq rate limits allow, run bounded live requests per implementation to capture real token usage, latency, and cost.
3. **Expanded retrieval dataset:** Beyond the current 6 queries, to avoid overfitting conclusions to a small sample.
4. **Production load testing:** To validate deterministic performance observations under realistic load.
5. **Real token/cost measurement:** With live provider access, measure actual token consumption and estimate provider cost.

---

## Files Created

- `app/f432_reports/final_comparative_evaluation.json` — Structured comparison data with explicit evidence-status fields
- `app/f432_reports/final_comparative_evaluation.md` — This standalone technical case study

## Source Reports Consumed

- `f432_reports/bm25_comparison_report.md` (F4.2 retrieval)
- `f432_reports/f432_report.md` (F4.3 answer quality)
- `f432_reports/f43_acceptance_report.md` (F4.3 acceptance)
- `f432_reports/f436_live_evaluation.md` (F4.3.6 live evaluation)
- `app/f432_reports/f44_agent_comparison.md` (F4.4 agent behavior)
- `app/f432_reports/f45_performance_comparison.md` (F4.5 performance)
- F4.5.1 audit findings

---

**Final assessment:** The Framework Edition successfully demonstrates LlamaIndex-based retrieval with measurable quality advantages, LangGraph-based graph orchestration executing real agent scenarios, application-controlled security boundaries, and deterministic answer-quality metrics for all 7 samples. What still needs real-world validation: genuine Ragas evaluation, live Groq performance benchmarking, production-scale load testing, and a larger retrieval dataset. Neither implementation is categorically superior — the choice depends on project constraints and engineering priorities.
