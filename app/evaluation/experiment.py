from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any

from .ragas_adapter import records_to_samples
from .ragas_evaluator import run_ragas_evaluation


@dataclass
class ExperimentResult:
    run_id: str
    dataset_name: str
    timestamp: str
    metric_names: list[str]
    sample_count: int
    per_sample: list[dict[str, Any]]
    aggregate: dict[str, Any]
    status: str
    error: str | None = None
    config: dict[str, Any] = field(default_factory=dict)


def run_experiment(
    records: list[dict[str, Any]],
    dataset_name: str = "default",
    metric_names: list[str] | None = None,
    config: dict[str, Any] | None = None,
    llm: Any = None,
    embeddings: Any = None,
    run_id: str | None = None,
) -> ExperimentResult:
    config = config or {}
    metric_names = metric_names or ["exact_match"]
    # validate without mutating
    samples = records_to_samples(records)
    if not samples:
        return ExperimentResult(
            run_id=run_id or __import__("uuid").uuid4().hex,
            dataset_name=dataset_name,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            metric_names=metric_names,
            sample_count=0,
            per_sample=[],
            aggregate={},
            status="failed",
            error="empty dataset",
            config=config,
        )
    try:
        eval_result = run_ragas_evaluation(
            records, metric_names=metric_names, llm=llm, embeddings=embeddings, run_id=run_id
        )
    except Exception as exc:
        return ExperimentResult(
            run_id=run_id or __import__("uuid").uuid4().hex,
            dataset_name=dataset_name,
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            metric_names=metric_names,
            sample_count=len(samples),
            per_sample=[],
            aggregate={},
            status="failed",
            error=str(exc),
            config=config,
        )
    return ExperimentResult(
        run_id=eval_result["run_id"],
        dataset_name=dataset_name,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        metric_names=eval_result["metric_names"],
        sample_count=eval_result["sample_count"],
        per_sample=eval_result["per_sample"],
        aggregate=eval_result["aggregate"],
        status=eval_result["status"],
        error=eval_result["error"],
        config=config,
    )