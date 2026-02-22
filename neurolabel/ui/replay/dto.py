from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ReplayRun:
    alive_time: int
    seed: int | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayBundle:
    backend: str
    runs: list[ReplayRun]
    metrics: dict[str, Any] = field(default_factory=dict)
