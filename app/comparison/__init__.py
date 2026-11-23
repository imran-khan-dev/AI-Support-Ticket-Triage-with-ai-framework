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
]