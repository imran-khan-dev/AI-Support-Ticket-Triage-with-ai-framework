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
            if metric_name == "faithfulness":
                from ragas.metrics import Faithfulness

                return Faithfulness()
            if metric_name == "answer_relevancy":
                from ragas.metrics import AnswerRelevancy

                return AnswerRelevancy()
            if metric_name == "answer_correctness":
                from ragas.metrics import AnswerCorrectness

                return AnswerCorrectness()
            if metric_name == "context_precision":
                from ragas.metrics import ContextPrecision

                return ContextPrecision()
            if metric_name == "context_recall":
                from ragas.metrics import ContextRecall

                return ContextRecall()
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

    # Detect when all scores are NaN/None (e.g., Ragas fell back to an
    # unsupported model like gpt-4o-mini on Groq). Treat as failed rather
    # than completed so callers don't mistake NaN for valid scores.
    def _has_usable_scores() -> bool:
        for row in per_sample:
            if isinstance(row, dict):
                for mname in metric_names:
                    v = row.get(mname)
                    if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and v != v
                    ):
                        return True
        return False

    if not _has_usable_scores():
        status = "failed"
        error = "all Ragas metric scores are NaN or None (LLM evaluator may be unavailable)"
    else:
        status = "completed"
        error = None

    return {
        "run_id": run_id or str(uuid.uuid4()),
        "metric_names": metric_names,
        "sample_count": len(samples),
        "per_sample": per_sample,
        "aggregate": aggregate,
        "status": status,
        "error": error,
    }