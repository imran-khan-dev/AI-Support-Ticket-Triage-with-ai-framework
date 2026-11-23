from __future__ import annotations

import uuid
import warnings
from typing import Any

from .ragas_adapter import RagasSample, records_to_samples, sample_to_ragas_dict


class RagasEvaluationError(Exception):
    pass


METRIC_FACTORIES = {}


def _load_metric(metric_name: str):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if metric_name == "exact_match":
                from ragas.metrics import ExactMatch

                return ExactMatch()
            if metric_name == "bleu_score":
                from ragas.metrics import BleuScore

                return BleuScore()
            if metric_name == "rouge_score":
                from ragas.metrics import RougeScore

                return RougeScore()
            if metric_name == "string_presence":
                from ragas.metrics import StringPresence

                return StringPresence()
            if metric_name == "topic_adherence":
                from ragas.metrics import TopicAdherence

                return TopicAdherence()
            raise RagasEvaluationError(f"unsupported metric: {metric_name}")
    except RagasEvaluationError:
        raise
    except Exception as exc:  # pragma: no cover
        raise RagasEvaluationError(str(exc)) from exc


def run_ragas_evaluation(
    records: list[dict[str, Any]],
    metric_names: list[str] | None = None,
    llm: Any = None,
    embeddings: Any = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    if metric_names is None:
        metric_names = ["exact_match"]
    if not metric_names:
        raise RagasEvaluationError("at least one metric is required")
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics.base import Metric
    except Exception as exc:  # pragma: no cover
        raise RagasEvaluationError(str(exc)) from exc

    samples = records_to_samples(records)
    if not samples:
        raise RagasEvaluationError("evaluation dataset is empty")
    ragas_samples = [sample_to_ragas_dict(s) for s in samples]
    dataset = EvaluationDataset.from_list(ragas_samples)
    metrics = [_load_metric(m) for m in metric_names]
    for metric in metrics:
        if not isinstance(metric, Metric):
            raise RagasEvaluationError(f"metric {metric!r} is not a valid Ragas metric")
    kwargs: dict[str, Any] = {}
    if llm is not None:
        kwargs["llm"] = llm
    if embeddings is not None:
        kwargs["embeddings"] = embeddings
    try:
        result = evaluate(dataset, metrics=metrics, **kwargs)
    except Exception as exc:
        raise RagasEvaluationError(str(exc)) from exc

    scores = getattr(result, "scores", None)
    if scores is None:
        scores = []
    per_sample = [dict(s) if not isinstance(s, dict) else s for s in scores] if isinstance(scores, list) else []
    aggregate: dict[str, Any] = {}
    for name in metric_names:
        vals = []
        for row in per_sample:
            if isinstance(row, dict) and name in row:
                v = row[name]
                if isinstance(v, (int, float)):
                    vals.append(v)
        aggregate[name] = (sum(vals) / len(vals)) if vals else None

    return {
        "run_id": run_id or str(uuid.uuid4()),
        "metric_names": metric_names,
        "sample_count": len(samples),
        "per_sample": per_sample,
        "aggregate": aggregate,
        "status": "completed",
        "error": None,
    }