from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

DeviceMode = Literal["eeg", "fnirs"]
BackendName = Literal["velocity", "metadrive"]


@dataclass(frozen=True)
class Paths:
    project_dir: Path
    data_dir: Path
    models_dir: Path
    results_dir: Path

    eeg_csv: Path
    fnirs_csv: Path
    game_jsonl: Path
    oc_scores_csv: Path
    dataset_dirty: Path
    dataset_clean: Path
    model_dirty: Path
    model_clean: Path
    results_dirty: Path
    results_clean: Path

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.models_dir, self.results_dir):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    backend: str
    device_mode: str
    paths: Paths

    oc_cutoff: float
    sims_per_model: int
    sim_workers: int
    batch_size: int

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    @property
    def brain_csv(self) -> Path:
        return self.paths.fnirs_csv if self.device_mode == "fnirs" else self.paths.eeg_csv
