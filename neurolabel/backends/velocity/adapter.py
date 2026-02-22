from __future__ import annotations

import subprocess
import sys
from typing import Any

from neurolabel.backends.base import BackendAdapter
from neurolabel.config.schema import Settings
from neurolabel.core.contracts import CollectSummary, TrainSummary


class VelocityBackend:
    name = "velocity"
    supports_visualizer = True

    def collect_live(self, settings: Settings) -> CollectSummary:
        # Reuse legacy server CLI path to preserve behavior during migration.
        result = subprocess.run([sys.executable, "server.py", "--record"])
        ok = result.returncode == 0
        return CollectSummary(ok=ok, message="velocity live collection", metadata={"returncode": result.returncode})

    def generate_synthetic(self, settings: Settings, *, duration_seconds: int) -> CollectSummary:
        from neurolabel.backends.velocity.synthetic import generate_synthetic

        generate_synthetic(duration_seconds=duration_seconds)
        return CollectSummary(ok=True, message="velocity synthetic generated")

    def train_models(self, settings: Settings) -> TrainSummary:
        from neurolabel.backends.velocity.model import train_model, save_model

        dirty_model = train_model(str(settings.paths.dataset_dirty))
        save_model(dirty_model, str(settings.paths.model_dirty))

        clean_model = train_model(str(settings.paths.dataset_clean))
        save_model(clean_model, str(settings.paths.model_clean))

        return TrainSummary(model_dirty=settings.paths.model_dirty, model_clean=settings.paths.model_clean)

    def simulate_comparison(self, settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
        from neurolabel.simulation.runner import run_comparison

        return run_comparison(settings)

    def visualize(self, settings: Settings, *, dirty_results=None, clean_results=None) -> None:
        from neurolabel.ui.replay.velocity_viewer import run_from_results

        run_from_results(settings, dirty_results=dirty_results, clean_results=clean_results)
