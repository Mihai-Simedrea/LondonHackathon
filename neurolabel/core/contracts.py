from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CollectSummary:
    ok: bool
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessSummary:
    oc_windows: int
    dataset_full: Path
    dataset_dirty: Path
    dataset_clean: Path


@dataclass(frozen=True)
class TrainSummary:
    model_dirty: Path
    model_clean: Path


@dataclass(frozen=True)
class SimSummary:
    dirty_results: dict[str, Any]
    clean_results: dict[str, Any]


@dataclass(frozen=True)
class DemoSummary:
    process: ProcessSummary
    train: TrainSummary
    sim: SimSummary
