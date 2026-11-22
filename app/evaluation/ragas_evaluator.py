from __future__ import annotations

from typing import Any

from .ragas_adapter import RagasSample, records_to_samples, sample_to_ragas_dict


class RagasEvaluationError(Exception):
    pass


def run_ragas_evaluation(records: list[dict[str, Any]], metric_name: str = "exact_match") -> dict[str, Any]:
    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics.collections import ExactMatch
    except Exception as exc:  # pragma: no cover
        raise RagasEvaluationError(str(exc)) from exc

    samples = records_to_samples(records)
    ragas_samples = [sample_to_ragas_dict(s) for s in samples]
    dataset = EvaluationDataset.from_list(ragas_samples)
    metric = ExactMatch() if metric_name == "exact_match" else None
    if metric is None:
        raise RagasEvaluationError(f"unsupported metric: {metric_name}")
    try:
        result = evaluate(dataset, metrics=[metric])
    except Exception as exc:
        raise RagasEvaluationError(str(exc)) from exc
    return {"metric": metric_name, "scores": list(result.scores) if hasattr(result, "scores") else [], "raw": result}