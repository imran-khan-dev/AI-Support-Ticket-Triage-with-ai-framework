# BM25 Retrieval Comparison Report

- **Experiment ID:** 91990c59-05e7-4f01-994c-216aca8e29b4
- **Dataset:** support-kb-retrieval-eval
- **Tenant:** eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee
- **Top-K:** 5
- **Custom Implementation:** Custom BM25 Retrieval Stack
- **Framework Implementation:** LlamaIndex + KnowledgeService (BM25)

## Aggregate Metrics

| Metric | Custom | Framework |
|--------|--------|-----------|
| mrr | 0.8333 | 1.0000 |
| precision_at_k.5 | 0.3333 | 0.4000 |
| recall_at_k.5 | 0.7500 | 0.9167 |
| ndcg_at_k.5 | 0.8167 | 0.9862 |

## Per-Query Results

| Query | Custom Chunks | Framework Chunks | Match | Custom MRR | Framework MRR |
|-------|---------------|------------------|-------|------------|---------------|
| q1 | C1, C2, C10 | C1, C2, C10 | yes | 1.0000 | 1.0000 |
| q2 | C3, C6, C4 | C3, C4, C6 | no | 1.0000 | 1.0000 |
| q3 |  | C5, C6 | no | 0.0000 | 1.0000 |
| q4 | C7, C8 | C7, C8 | yes | 1.0000 | 1.0000 |
| q5 | C9, C2, C5 | C9, C2, C5 | yes | 1.0000 | 1.0000 |
| q6 | C2, C1, C9, C5, C7 | C2, C1, C5, C9 | no | 1.0000 | 1.0000 |