"""Framework-independent comparative evaluation contracts.

This package defines shared dataset and result contracts used by both the
custom and framework editions. It deliberately avoids importing any
implementation internals so it can be used by either edition.
"""

from .contracts import (
    ComparisonDataset,
    ComparisonResult,
    ComparisonRunConfig,
    ComparisonSample,
    ComparisonScenario,
    Implementation,
    MetricScore,
    RetrievalSample,
    AnswerSample,
    AgentScenario,
)
from .metrics import (
    average,
    dcg_at_k,
    idcg_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .retrieval_adapters import (
    CustomBM25RetrievalAdapter,
    FrameworkBM25RetrievalAdapter,
    RetrievalAdapter,
    RetrievalHit,
    RetrievalRequest,
)
from .retrieval_benchmark import (
    RetrievalBenchmarkResult,
    RetrievalQueryResult,
    run_retrieval_benchmark,
)

__all__ = [
    "ComparisonDataset",
    "ComparisonResult",
    "ComparisonRunConfig",
    "ComparisonSample",
    "ComparisonScenario",
    "Implementation",
    "MetricScore",
    "RetrievalSample",
    "AnswerSample",
    "AgentScenario",
    "average",
    "dcg_at_k",
    "idcg_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "CustomBM25RetrievalAdapter",
    "FrameworkBM25RetrievalAdapter",
    "RetrievalAdapter",
    "RetrievalHit",
    "RetrievalRequest",
    "RetrievalBenchmarkResult",
    "RetrievalQueryResult",
    "run_retrieval_benchmark",
]
