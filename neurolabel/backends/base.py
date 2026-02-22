from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from neurolabel.config.schema import Settings
from neurolabel.core.contracts import CollectSummary, TrainSummary


class BackendAdapter(Protocol):
    name: str
    supports_visualizer: bool

    def collect_live(self, settings: Settings) -> CollectSummary: ...
    def generate_synthetic(self, settings: Settings, *, duration_seconds: int) -> CollectSummary: ...
    def train_models(self, settings: Settings) -> TrainSummary: ...
    def simulate_comparison(self, settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]: ...
    def visualize(self, settings: Settings, *, dirty_results: dict[str, Any] | None = None, clean_results: dict[str, Any] | None = None) -> None: ...
