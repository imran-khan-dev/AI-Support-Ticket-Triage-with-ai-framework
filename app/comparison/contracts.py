from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class Implementation(str, Enum):
    custom = "custom"
    framework = "framework"


class Component(str, Enum):
    retrieval = "retrieval"
    agent = "agent"
    evaluation = "evaluation"


class RetrievalSample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)
    expected_chunk_ids: list[str] | None = None
    expected_doc_ids: list[str] | None = None
    reference_contexts: list[str] | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerSample(BaseModel):
    sample_id: str = Field(..., min_length=1)
    user_input: str = Field(..., min_length=1)
    reference: str | None = None
    contexts: list[str] | None = None
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentScenario(BaseModel):
    scenario_id: str = Field(..., min_length=1)
    user_request: str = Field(..., min_length=1)
    expected_tool_names: list[str] | None = None
    expected_outcome: str | None = None
    requires_approval: bool = False
    tenant_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ComparisonScenario(AgentScenario):
    """Alias for an agent scenario in the shared contract."""


class ComparisonSample(BaseModel):
    retrieval: list[RetrievalSample] = Field(default_factory=list)
    answer: list[AnswerSample] = Field(default_factory=list)
    agent: list[AgentScenario] = Field(default_factory=list)

    @model_validator(mode="after")
    def at_least_one_sample(self) -> "ComparisonSample":
        if not (self.retrieval or self.answer or self.agent):
            raise ValueError("comparison sample must contain at least one dimension")
        return self


class ComparisonDataset(BaseModel):
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    version: str = "1.0.0"
    description: str | None = None
    samples: ComparisonSample = Field(default_factory=ComparisonSample)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_ids(self) -> "ComparisonDataset":
        seen: set[str] = set()
        for sample in self.samples.retrieval:
            if sample.sample_id in seen:
                raise ValueError(f"duplicate retrieval sample id: {sample.sample_id}")
            seen.add(sample.sample_id)
        for sample in self.samples.answer:
            if sample.sample_id in seen:
                raise ValueError(f"duplicate answer sample id: {sample.sample_id}")
            seen.add(sample.sample_id)
        for scenario in self.samples.agent:
            if scenario.scenario_id in seen:
                raise ValueError(f"duplicate agent scenario id: {scenario.scenario_id}")
            seen.add(scenario.scenario_id)
        return self


class MetricScore(BaseModel):
    metric_name: str
    provenance: str
    score: float | None = None
    sample_id: str | None = None


class ComparisonRunConfig(BaseModel):
    implementation: Implementation
    component: Component
    dataset_id: str
    experiment_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str | None = None
    embedding_name: str | None = None
    top_k: int | None = None
    max_iterations: int | None = None
    max_tool_calls: int | None = None
    tenant_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ComparisonResult(BaseModel):
    experiment_id: str
    implementation: Implementation
    component: Component
    dataset_id: str
    sample_id: str | None = None
    status: str
    scores: list[MetricScore] = Field(default_factory=list)
    latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())