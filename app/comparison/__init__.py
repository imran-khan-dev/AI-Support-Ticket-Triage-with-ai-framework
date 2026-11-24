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
from .answer_adapters import (
    AnswerAdapter,
    AnswerResult,
    RetrievedContext,
    CustomAnswerAdapter,
    FrameworkAnswerAdapter,
)
from .answer_metrics import (
    answer_correctness,
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
    unsupported_claims,
)
from .answer_comparison import (
    AnswerQualityComparisonResult,
    AnswerQualitySampleResult,
    DETERMINISTIC_METRIC_NAMES,
    RAGAS_METRIC_NAMES,
    ALL_METRIC_NAMES,
    load_answer_quality_dataset,
    run_answer_quality_comparison,
    save_json_report,
    save_markdown_report,
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
    "AnswerAdapter",
    "AnswerResult",
    "RetrievedContext",
    "CustomAnswerAdapter",
    "FrameworkAnswerAdapter",
    "answer_correctness",
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
    "unsupported_claims",
    "AnswerQualityComparisonResult",
    "AnswerQualitySampleResult",
    "DETERMINISTIC_METRIC_NAMES",
    "RAGAS_METRIC_NAMES",
    "ALL_METRIC_NAMES",
    "load_answer_quality_dataset",
    "run_answer_quality_comparison",
    "save_json_report",
    "save_markdown_report",
]
