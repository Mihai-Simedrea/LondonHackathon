from __future__ import annotations

from typing import Any

from neurolabel.backends.base import BackendAdapter
from neurolabel.config.schema import Settings
from neurolabel.core.contracts import CollectSummary, TrainSummary


class MetaDriveBackend:
    name = "metadrive"
    supports_visualizer = False

    def collect_live(self, settings: Settings) -> CollectSummary:
        from neurolabel.backends.metadrive.recording import record_session

        summary = record_session()
        return CollectSummary(ok=True, message="metadrive live collection", metadata=summary or {})

    def generate_synthetic(self, settings: Settings, *, duration_seconds: int) -> CollectSummary:
        from neurolabel.backends.metadrive.synthetic import generate_synthetic_metadrive

        generate_synthetic_metadrive(duration_seconds=duration_seconds)
        return CollectSummary(ok=True, message="metadrive synthetic generated")

    def train_models(self, settings: Settings) -> TrainSummary:
        from neurolabel.backends.metadrive.model import train_model, save_model

        dirty_model = train_model(str(settings.paths.dataset_dirty))
        save_model(dirty_model, str(settings.paths.model_dirty))

        clean_model = train_model(str(settings.paths.dataset_clean))
        save_model(clean_model, str(settings.paths.model_clean))

        return TrainSummary(model_dirty=settings.paths.model_dirty, model_clean=settings.paths.model_clean)

    def simulate_comparison(self, settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
        from neurolabel.simulation.runner import run_comparison

        return run_comparison(settings)

    def visualize(self, settings: Settings, *, dirty_results=None, clean_results=None) -> None:
        raise RuntimeError("Replay visualization is currently supported only for the Velocity backend")
