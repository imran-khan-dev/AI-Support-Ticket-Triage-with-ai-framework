from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import ComparisonDataset, ComparisonResult, ComparisonRunConfig


class ComparisonAdapter(ABC):
    """Independent adapter between the shared contract and an implementation."""

    @property
    def implementation(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self, source: Any) -> ComparisonDataset:
        """Load an implementation-specific dataset into the shared contract."""

    @abstractmethod
    def run(self, config: ComparisonRunConfig, dataset: ComparisonDataset) -> ComparisonResult:
        """Execute a single comparison run and return a normalized result."""

    def normalize_result(self, raw: Any, config: ComparisonRunConfig, sample_id: str | None = None) -> ComparisonResult:
        return ComparisonResult(
            experiment_id=config.experiment_id,
            implementation=config.implementation,
            component=config.component,
            dataset_id=config.dataset_id,
            sample_id=sample_id,
            status="completed",
            scores=[],
            metadata={"raw": str(raw)},
        )


class NoOpAdapter(ComparisonAdapter):
    """Minimal adapter used for contract-level tests."""

    def __init__(self, implementation: str = "custom") -> None:
        self._implementation = implementation

    @property
    def implementation(self) -> str:
        return self._implementation

    def load_dataset(self, source: Any) -> ComparisonDataset:
        if isinstance(source, ComparisonDataset):
            return source
        raise ValueError("source must be a ComparisonDataset")

    def run(self, config: ComparisonRunConfig, dataset: ComparisonDataset) -> ComparisonResult:
        return ComparisonResult(
            experiment_id=config.experiment_id,
            implementation=config.implementation,
            component=config.component,
            dataset_id=config.dataset_id,
            status="completed",
            scores=[],
        )