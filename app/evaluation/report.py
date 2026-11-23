from __future__ import annotations

import json
from typing import Any

from .experiment import ExperimentResult


def experiment_to_json(result: ExperimentResult) -> str:
    payload = {
        "run_id": result.run_id,
        "dataset_name": result.dataset_name,
        "timestamp": result.timestamp,
        "metric_names": result.metric_names,
        "sample_count": result.sample_count,
        "per_sample": result.per_sample,
        "aggregate": result.aggregate,
        "status": result.status,
        "error": result.error,
        "config": result.config,
    }
    return json.dumps(payload, indent=2, default=str)


def experiment_to_markdown(result: ExperimentResult) -> str:
    lines = [
        f"# Evaluation Report: {result.dataset_name}",
        "",
        f"- Run ID: {result.run_id}",
        f"- Timestamp: {result.timestamp}",
        f"- Status: {result.status}",
        f"- Samples: {result.sample_count}",
        f"- Metrics: {', '.join(result.metric_names)}",
        "",
        "## Aggregate Scores",
        "",
    ]
    for k, v in result.aggregate.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Per-Sample Scores")
    lines.append("")
    for row in result.per_sample:
        lines.append(f"- {row}")
    if result.error:
        lines.append("")
        lines.append(f"## Error\n\n{result.error}")
    return "\n".join(lines)