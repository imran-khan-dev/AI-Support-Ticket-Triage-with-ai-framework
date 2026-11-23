from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagasSample:
    user_input: str
    response: str | None = None
    reference: str | None = None
    retrieved_contexts: list[str] | None = None


def record_to_sample(record: dict[str, Any]) -> RagasSample:
    if not isinstance(record, dict):
        raise ValueError("evaluation record must be a mapping")
    user_input = record.get("user_input") or record.get("question") or record.get("message")
    if not user_input:
        raise ValueError("evaluation record is missing user_input")
    contexts = record.get("retrieved_contexts") or record.get("contexts")
    if contexts is not None and not isinstance(contexts, list):
        raise ValueError("retrieved_contexts must be a list")
    return RagasSample(
        user_input=str(user_input),
        response=record.get("response") if record.get("response") is not None else record.get("answer"),
        reference=record.get("reference") if record.get("reference") is not None else record.get("ground_truth"),
        retrieved_contexts=[str(c) for c in contexts] if contexts is not None else None,
    )


def records_to_samples(records: list[dict[str, Any]]) -> list[RagasSample]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    return [record_to_sample(r) for r in records]


def sample_to_ragas_dict(sample: RagasSample) -> dict[str, Any]:
    payload: dict[str, Any] = {"user_input": sample.user_input}
    if sample.response is not None:
        payload["response"] = sample.response
    if sample.reference is not None:
        payload["reference"] = sample.reference
    if sample.retrieved_contexts is not None:
        payload["retrieved_contexts"] = sample.retrieved_contexts
    return payload