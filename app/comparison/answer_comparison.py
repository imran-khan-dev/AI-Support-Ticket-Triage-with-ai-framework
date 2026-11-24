"""Answer-quality and faithfulness comparison runner.

Runs equivalent answer-generation scenarios against the Custom and Framework
editions, evaluates both outputs using identical framework-independent
metrics, and produces per-sample and aggregate results.

All comparison-specific logic lives within app/comparison.

Metric layers:
- Deterministic metrics (answer_correctness, answer_relevancy, faithfulness,
  context_precision, context_recall) are always computed. They require no
  external LLM calls and are suitable for unit testing.
- Ragas metrics (ragas_faithfulness, ragas_answer_relevancy,
  ragas_answer_correctness, ragas_context_precision, ragas_context_recall)
  are computed only when the ``ragas`` package is installed and a real LLM
  evaluator is available. They are the primary common evaluation methodology
  and are kept separate from deterministic metrics to avoid conflating
  keyword-overlap heuristics with LLM-based evaluation.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.comparison.answer_adapters import (
    AnswerAdapter,
    AnswerResult,
    CustomAnswerAdapter,
    FrameworkAnswerAdapter,
)
from app.comparison.answer_metrics import (
    answer_correctness,
    answer_relevancy,
    average,
    context_precision,
    context_recall,
    faithfulness,
    unsupported_claims,
)
from app.comparison.contracts import (
    AnswerSample,
    ComparisonDataset,
    ComparisonResult,
    ComparisonRunConfig,
    Component,
    Implementation,
    MetricScore,
)
from app.core.config import Settings

# Ragas evaluation is optional. Import lazily so the module loads without it.
_RAGAS_AVAILABLE: bool = False


def _try_import_ragas() -> bool:
    global _RAGAS_AVAILABLE
    if _RAGAS_AVAILABLE:
        return True
    try:
        from app.evaluation.ragas_evaluator import run_ragas_evaluation  # noqa: F401
        _RAGAS_AVAILABLE = True
    except Exception:
        _RAGAS_AVAILABLE = False
    return _RAGAS_AVAILABLE


def _run_ragas_evaluation(
    records: list[dict[str, Any]],
    metric_names: list[str] | None = None,
    llm: Any = None,
    embeddings: Any = None,
) -> dict[str, Any]:
    """Run Ragas evaluation on a list of records.

    Returns a dict with ``status``, ``error``, ``per_sample``, and
    ``aggregate`` keys. When Ragas is unavailable, returns
    ``{"status": "unavailable", "error": "..."}``.
    """
    if not _try_import_ragas():
        return {
            "status": "unavailable",
            "error": "ragas package not installed",
            "per_sample": [],
            "aggregate": {},
        }
    from app.evaluation.ragas_evaluator import run_ragas_evaluation as _run

    try:
        return _run(records, metric_names=metric_names, llm=llm, embeddings=embeddings)
    except Exception as exc:  # noqa: BLE001 - capture boundary errors
        return {
            "status": "failed",
            "error": str(exc),
            "per_sample": [],
            "aggregate": {},
        }


# Ragas metric names used for the common comparison.
RAGAS_METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "answer_correctness",
    "context_precision",
    "context_recall",
]

DETERMINISTIC_METRIC_NAMES = [
    "answer_correctness",
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
]

ALL_METRIC_NAMES = DETERMINISTIC_METRIC_NAMES + [
    f"ragas_{m}" for m in RAGAS_METRIC_NAMES
]


@dataclass
class AnswerQualitySampleResult:
    """Per-sample result for one implementation."""

    sample_id: str
    question: str
    reference: str | None
    implementation: str
    answer: str | None
    contexts: list[str]
    context_chunk_ids: list[str]
    metrics: dict[str, float | None]
    status: str
    error: str | None = None
    unsupported_claims_list: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerQualityComparisonResult:
    """Full comparison result across both implementations."""

    experiment_id: str
    dataset_id: str
    dataset_name: str
    run_config: ComparisonRunConfig
    samples: list[dict[str, Any]] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None
    ragas_available: bool = False
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_contract_results(self) -> list[ComparisonResult]:
        """Normalize into F4.1 ComparisonResult contract objects."""
        results: list[ComparisonResult] = []
        for sample in self.samples:
            for impl in ("custom", "framework"):
                impl_data = sample.get(impl, {})
                scores = []
                for metric_name, value in impl_data.get("metrics", {}).items():
                    scores.append(
                        MetricScore(
                            metric_name=metric_name,
                            provenance=impl,
                            score=value,
                            sample_id=sample["sample_id"],
                        )
                    )
                results.append(
                    ComparisonResult(
                        experiment_id=self.experiment_id,
                        implementation=Implementation(impl),
                        component=Component.agent,
                        dataset_id=self.dataset_id,
                        sample_id=sample["sample_id"],
                        status=impl_data.get("status", "unknown"),
                        scores=scores,
                        error=impl_data.get("error"),
                        metadata={
                            "question": sample.get("question", ""),
                            "answer": impl_data.get("answer"),
                            "context_chunk_ids": impl_data.get(
                                "context_chunk_ids", []
                            ),
                        },
                    )
                )
        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "run_config": self.run_config.model_dump(),
            "samples": self.samples,
            "aggregate": self.aggregate,
            "status": self.status,
            "error": self.error,
            "ragas_available": self.ragas_available,
            "completed_at": self.completed_at,
        }


def load_answer_quality_dataset(
    path: str | Path,
) -> dict[str, Any]:
    """Load the versioned answer-quality dataset from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_document_chunks(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Build chunk dicts for the Custom Retrieval Stack from the dataset."""
    chunks = []
    for doc in dataset.get("documents", []):
        for chunk in doc.get("chunks", []):
            chunks.append(
                {
                    "logical_id": chunk["logical_id"],
                    "document_id": doc["id"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "metadata": dict(chunk.get("metadata", {})),
                }
            )
    return chunks


def _build_adapter_documents(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    """Build document dicts for the Framework adapter from the dataset."""
    documents = []
    for doc in dataset.get("documents", []):
        documents.append(
            {
                "filename": doc["filename"],
                "content_type": doc["content_type"],
                "content": doc["content"],
                "chunks": [
                    {
                        "logical_id": c["logical_id"],
                        "chunk_index": c["chunk_index"],
                        "content": c["content"],
                        "metadata": dict(c.get("metadata", {})),
                    }
                    for c in doc.get("chunks", [])
                ],
            }
        )
    return documents


def _evaluate_sample_deterministic(
    sample: dict[str, Any],
    custom_result: AnswerResult,
    framework_result: AnswerResult,
    chunk_id_to_text: dict[str, str],
) -> dict[str, Any]:
    """Evaluate one sample for both implementations using deterministic metrics."""
    question = sample["question"]
    reference = sample.get("reference_answer")
    expected_facts = sample.get("expected_facts", [])
    expected_chunk_ids = sample.get("expected_context_chunk_ids", [])

    def eval_impl(result: AnswerResult) -> dict[str, Any]:
        answer = result.answer
        contexts = result.context_texts
        chunk_ids = result.context_chunk_ids

        # Build chunk_id -> text mapping for this result
        c2t = {c.chunk_id: c.content for c in result.contexts}

        metrics = {
            "answer_correctness": answer_correctness(answer, reference, expected_facts),
            "answer_relevancy": answer_relevancy(answer, question, reference),
            "faithfulness": faithfulness(answer, contexts),
            "context_precision": context_precision(
                chunk_ids, expected_chunk_ids, chunk_id_to_text=c2t
            ),
            "context_recall": context_recall(chunk_ids, expected_chunk_ids),
        }

        unsupported = unsupported_claims(answer, contexts)

        return {
            "answer": answer,
            "contexts": contexts,
            "context_chunk_ids": chunk_ids,
            "metrics": metrics,
            "status": result.status,
            "error": result.error,
            "unsupported_claims": unsupported,
            "metadata": result.metadata,
        }

    return {
        "sample_id": sample["sample_id"],
        "question": question,
        "reference": reference,
        "expected_facts": expected_facts,
        "expected_context_chunk_ids": expected_chunk_ids,
        "category": sample.get("category", "unknown"),
        "sufficient_context": sample.get("sufficient_context", True),
        "custom": eval_impl(custom_result),
        "framework": eval_impl(framework_result),
    }


def _apply_ragas_evaluation(
    samples: list[dict[str, Any]],
    client: Any = None,
    settings: Any = None,
) -> dict[str, Any]:
    """Run Ragas evaluation on all samples for both implementations.

    Returns a dict with ``status``, ``error``, ``available`` keys.
    When Ragas is unavailable or fails, samples are unchanged and
    ``available`` is False.
    """
    if not _try_import_ragas():
        return {
            "status": "unavailable",
            "error": "ragas package not installed",
            "available": False,
        }

    import os

    # Ragas 0.4.x internally creates an OpenAI-compatible client. Provide the
    # Groq-configured key/base URL so it can initialize with our provider.
    _s = settings
    if _s is None:
        from app.core.config import Settings as _Settings

        _s = Settings()
    if _s.groq_api_key:
        os.environ.setdefault("OPENAI_API_KEY", _s.groq_api_key)
    if _s.groq_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", _s.groq_base_url)

    # Build a LangChain LLM backed by the configured Groq client for Ragas.
    ragas_llm = None
    if client is not None:
        try:
            from langchain_openai import ChatOpenAI

            ragas_llm = ChatOpenAI(
                model=_s.groq_model,
                openai_api_key=_s.groq_api_key,
                base_url=_s.groq_base_url,
                temperature=0.0,
            )
        except Exception:
            ragas_llm = None

    # Build Ragas records for each implementation.
    ragas_records: dict[str, list[dict[str, Any]]] = {
        "custom": [],
        "framework": [],
    }
    for sample in samples:
        for impl in ("custom", "framework"):
            impl_data = sample[impl]
            ragas_records[impl].append(
                {
                    "user_input": sample["question"],
                    "response": impl_data.get("answer"),
                    "reference": sample.get("reference"),
                    "retrieved_contexts": impl_data.get("contexts", []),
                }
            )

    # Run Ragas evaluation for each implementation.
    ragas_results: dict[str, dict[str, Any]] = {}
    for impl in ("custom", "framework"):
        ragas_results[impl] = _run_ragas_evaluation(
            ragas_records[impl],
            metric_names=RAGAS_METRIC_NAMES,
            llm=ragas_llm,
        )

    # Check if either implementation succeeded with usable scores.
    # Ragas may report status "completed" while returning all None scores
    # (e.g., when the underlying LLM model is unavailable). Treat that as
    # unavailable rather than successful.
    def _has_usable_scores(ragas_result: dict[str, Any]) -> bool:
        for row in ragas_result.get("per_sample", []):
            if isinstance(row, dict):
                for mname in RAGAS_METRIC_NAMES:
                    v = row.get(mname)
                    if isinstance(v, (int, float)) and not (
                        isinstance(v, float) and v != v
                    ):
                        return True
        return False

    any_success = any(
        r.get("status") == "completed" and _has_usable_scores(r)
        for r in ragas_results.values()
    )

    if not any_success:
        return {
            "status": ragas_results["custom"].get("status", "failed"),
            "error": "; ".join(
                r.get("error", "") for r in ragas_results.values() if r.get("error")
            ),
            "available": False,
        }

    # Merge Ragas per-sample scores into samples.
    sample_idx = 0
    for sample in samples:
        for impl in ("custom", "framework"):
            impl_ragas = ragas_results[impl]
            per_sample = impl_ragas.get("per_sample", [])
            if sample_idx < len(per_sample):
                row = per_sample[sample_idx]
                if isinstance(row, dict):
                    ragas_metrics = {}
                    for mname in RAGAS_METRIC_NAMES:
                        v = row.get(mname)
                        ragas_metrics[f"ragas_{mname}"] = (
                            float(v) if isinstance(v, (int, float)) else None
                        )
                    sample[impl]["metrics"].update(ragas_metrics)
        sample_idx += 1

    # Build aggregate Ragas metrics.
    ragas_aggregate: dict[str, Any] = {}
    for impl in ("custom", "framework"):
        impl_ragas = ragas_results[impl]
        aggregate = impl_ragas.get("aggregate", {})
        ragas_aggregate[impl] = {}
        for mname in RAGAS_METRIC_NAMES:
            key = f"ragas_{mname}"
            mdata = aggregate.get(mname, {})
            ragas_aggregate[impl][key] = {
                "mean": mdata if isinstance(mdata, (int, float)) else None,
                "count": None,
            }

    return {
        "status": "completed",
        "error": None,
        "available": True,
        "results": ragas_results,
        "aggregate": ragas_aggregate,
    }


def _compute_aggregate(
    samples: list[dict[str, Any]],
    ragas_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute aggregate metrics across all samples for both implementations."""
    aggregate: dict[str, Any] = {
        "num_samples": len(samples),
        "num_completed_custom": 0,
        "num_completed_framework": 0,
        "num_failed_custom": 0,
        "num_failed_framework": 0,
        "custom": {},
        "framework": {},
    }

    metric_names = list(DETERMINISTIC_METRIC_NAMES)
    if ragas_info and ragas_info.get("available"):
        metric_names += [f"ragas_{m}" for m in RAGAS_METRIC_NAMES]

    for impl in ("custom", "framework"):
        completed = 0
        failed = 0
        metric_values: dict[str, list[float]] = {m: [] for m in metric_names}

        for sample in samples:
            impl_data = sample[impl]
            if impl_data["status"] in ("completed", "mock", "success"):
                completed += 1
            else:
                failed += 1

            for m in metric_names:
                v = impl_data["metrics"].get(m)
                if v is not None:
                    metric_values[m].append(v)

        aggregate[f"num_completed_{impl}"] = completed
        aggregate[f"num_failed_{impl}"] = failed
        aggregate[impl] = {
            "num_completed": completed,
            "num_failed": failed,
            "metrics": {},
        }
        for m in metric_names:
            vals = metric_values[m]
            aggregate[impl]["metrics"][m] = {
                "mean": average(vals),
                "min": min(vals) if vals else None,
                "max": max(vals) if vals else None,
                "count": len(vals),
            }

    # Merge Ragas aggregate if available.
    if ragas_info and ragas_info.get("available"):
        for impl in ("custom", "framework"):
            if impl in ragas_info.get("aggregate", {}):
                for key, val in ragas_info["aggregate"][impl].items():
                    aggregate[impl]["metrics"][key] = val

    return aggregate


def run_answer_quality_comparison(
    dataset_path: str | Path,
    db: Any | None = None,
    tenant_id: str | None = None,
    top_k: int = 5,
    client: Any | None = None,
    settings: Any | None = None,
    experiment_id: str | None = None,
    run_ragas: bool = False,
) -> AnswerQualityComparisonResult:
    """Run the answer-quality comparison for both implementations.

    Args:
        dataset_path: Path to the answer-quality dataset JSON file.
        db: Database handle required by the Framework adapter.
        tenant_id: Tenant ID for retrieval isolation.
        top_k: Number of context chunks to retrieve per question.
        client: Optional OpenAI-compatible client for LLM calls. When None,
            both adapters run in mock mode (deterministic, no API calls).
        settings: Optional Settings object. When None, a default Settings()
            is used.
        experiment_id: Optional experiment ID. When None, a UUID is generated.
        run_ragas: When True, attempt Ragas evaluation on both implementations'
            outputs. Requires the ``ragas`` package and a configured LLM
            evaluator. When False (default), only deterministic metrics are
            computed.

    Returns:
        AnswerQualityComparisonResult with per-sample and aggregate results.
    """
    if settings is None:
        settings = Settings()

    if experiment_id is None:
        experiment_id = str(uuid.uuid4())

    dataset = load_answer_quality_dataset(dataset_path)
    dataset_id = dataset.get("dataset_id", str(uuid.uuid4()))
    dataset_name = dataset.get("name", "answer-quality-eval")
    effective_tenant_id = tenant_id or dataset.get("tenant_id")

    chunks = _build_document_chunks(dataset)
    documents = _build_adapter_documents(dataset)

    custom_adapter = CustomAnswerAdapter(
        chunks=chunks,
        settings=settings,
        client=client,
    )

    if db is None:
        framework_result_error = "Framework adapter requires a database handle"
        framework_adapter = None
    else:
        framework_adapter = FrameworkAnswerAdapter(
            db=db,
            tenant_id=effective_tenant_id or "default",
            documents=documents,
            settings=settings,
            client=client,
        )

    # Build chunk_id -> text mapping from dataset for context precision.
    chunk_id_to_text: dict[str, str] = {}
    for doc in dataset.get("documents", []):
        for chunk in doc.get("chunks", []):
            chunk_id_to_text[chunk["logical_id"]] = chunk["content"]

    samples: list[dict[str, Any]] = []

    for sample in dataset.get("samples", []):
        question = sample["question"]

        # Custom edition.
        custom_result = custom_adapter.generate(
            question=question,
            tenant_id=effective_tenant_id or "default",
            top_k=top_k,
        )

        # Framework edition.
        if framework_adapter is None:
            framework_result = AnswerResult(
                implementation="framework",
                question=question,
                answer=None,
                status="failed",
                error=framework_result_error,
            )
        else:
            framework_result = framework_adapter.generate(
                question=question,
                tenant_id=effective_tenant_id or "default",
                top_k=top_k,
            )

        sample_result = _evaluate_sample_deterministic(
            sample, custom_result, framework_result, chunk_id_to_text
        )
        samples.append(sample_result)

    # Run Ragas evaluation if requested and available.
    ragas_info = None
    ragas_available = False
    if run_ragas:
        ragas_info = _apply_ragas_evaluation(samples, client=client, settings=settings)
        ragas_available = ragas_info.get("available", False)

    # Determine overall status. "mock" and "success" are successful statuses.
    successful_statuses = {"completed", "mock", "success"}
    all_failed = all(
        s["custom"]["status"] not in successful_statuses
        and s["framework"]["status"] not in successful_statuses
        for s in samples
    )
    overall_status = "failed" if all_failed else "completed"

    aggregate = _compute_aggregate(samples, ragas_info)

    run_config = ComparisonRunConfig(
        implementation=Implementation.custom,  # comparison spans both
        component=Component.agent,
        dataset_id=dataset_id,
        experiment_id=experiment_id,
        top_k=top_k,
        tenant_id=effective_tenant_id,
        extra={
            "framework_included": db is not None,
            "ragas_available": ragas_available,
            "ragas_run": run_ragas,
        },
    )

    return AnswerQualityComparisonResult(
        experiment_id=experiment_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        run_config=run_config,
        samples=samples,
        aggregate=aggregate,
        status=overall_status,
        ragas_available=ragas_available,
    )


def save_json_report(
    result: AnswerQualityComparisonResult, path: str | Path
) -> None:
    """Save the comparison result as a JSON report."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def save_markdown_report(
    result: AnswerQualityComparisonResult, path: str | Path
) -> None:
    """Save the comparison result as a human-readable Markdown report."""
    lines: list[str] = []

    lines.append(f"# Answer Quality Comparison Report")
    lines.append("")
    lines.append(f"- **Experiment ID:** {result.experiment_id}")
    lines.append(f"- **Dataset:** {result.dataset_name} ({result.dataset_id})")
    lines.append(f"- **Status:** {result.status}")
    lines.append(f"- **Completed at:** {result.completed_at}")
    lines.append(f"- **Top-k:** {result.run_config.top_k}")
    lines.append(f"- **Tenant:** {result.run_config.tenant_id}")
    lines.append(f"- **Ragas available:** {result.ragas_available}")
    lines.append("")

    # Aggregate section.
    lines.append("## Aggregate Results")
    lines.append("")
    lines.append(
        "| Metric | Custom Mean | Custom Count | Framework Mean | Framework Count |"
    )
    lines.append(
        "|--------|-------------|--------------|----------------|-----------------|"
    )
    metric_names = list(DETERMINISTIC_METRIC_NAMES)
    if result.ragas_available:
        metric_names += [f"ragas_{m}" for m in RAGAS_METRIC_NAMES]
    for mname in metric_names:
        c = result.aggregate.get("custom", {}).get("metrics", {}).get(mname, {})
        f = result.aggregate.get("framework", {}).get("metrics", {}).get(mname, {})
        c_mean = f"{c.get('mean', 0.0):.4f}" if c.get("mean") is not None else "N/A"
        f_mean = f"{f.get('mean', 0.0):.4f}" if f.get("mean") is not None else "N/A"
        lines.append(
            f"| {mname} | {c_mean} | {c.get('count', 0)} | {f_mean} | {f.get('count', 0)} |"
        )
    lines.append("")

    lines.append(f"- **Custom completed:** {result.aggregate.get('num_completed_custom', 0)}")
    lines.append(f"- **Custom failed:** {result.aggregate.get('num_failed_custom', 0)}")
    lines.append(f"- **Framework completed:** {result.aggregate.get('num_completed_framework', 0)}")
    lines.append(f"- **Framework failed:** {result.aggregate.get('num_failed_framework', 0)}")
    lines.append("")

    # Per-sample section.
    lines.append("## Per-Sample Results")
    lines.append("")
    for sample in result.samples:
        lines.append(f"### {sample['sample_id']}: {sample['question']}")
        lines.append(f"- **Category:** {sample.get('category', 'unknown')}")
        lines.append(f"- **Sufficient context:** {sample.get('sufficient_context', True)}")
        if sample.get("reference"):
            lines.append(f"- **Reference:** {sample['reference']}")
        lines.append("")

        for impl in ("custom", "framework"):
            impl_data = sample[impl]
            lines.append(f"#### {impl.title()}")
            lines.append(f"- **Status:** {impl_data['status']}")
            if impl_data.get("error"):
                lines.append(f"- **Error:** {impl_data['error']}")
            if impl_data.get("answer"):
                lines.append(f"- **Answer:** {impl_data['answer']}")
            lines.append(f"- **Context chunks:** {impl_data['context_chunk_ids']}")
            lines.append("- **Metrics:**")
            for mname, mvalue in impl_data.get("metrics", {}).items():
                mv = f"{mvalue:.4f}" if mvalue is not None else "N/A"
                lines.append(f"  - {mname}: {mv}")
            if impl_data.get("unsupported_claims"):
                lines.append(f"- **Unsupported claims:** {impl_data['unsupported_claims']}")
            lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- This is a small, deterministic evaluation dataset.")
    lines.append("- No statistical significance claims are made from these results.")
    lines.append(
        "- Deterministic faithfulness uses keyword overlap, not LLM judgment. "
        "Ragas faithfulness (when available) uses LLM-based evaluation."
    )
    lines.append("- Mock mode (no LLM client) returns deterministic mock answers.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
