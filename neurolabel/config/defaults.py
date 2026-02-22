from __future__ import annotations

from pathlib import Path

import config as legacy_config

from .schema import Paths, Settings


def from_legacy_config() -> Settings:
    project_dir = Path(getattr(legacy_config, "PROJECT_DIR", Path(__file__).resolve().parents[2]))
    paths = Paths(
        project_dir=project_dir,
        data_dir=Path(legacy_config.DATA_DIR),
        models_dir=Path(legacy_config.MODELS_DIR),
        results_dir=Path(legacy_config.RESULTS_DIR),
        eeg_csv=Path(legacy_config.EEG_CSV),
        fnirs_csv=Path(legacy_config.FNIRS_CSV),
        game_jsonl=Path(legacy_config.GAME_JSONL),
        oc_scores_csv=Path(legacy_config.OC_SCORES_CSV),
        dataset_dirty=Path(legacy_config.DATASET_DIRTY),
        dataset_clean=Path(legacy_config.DATASET_CLEAN),
        model_dirty=Path(legacy_config.MODEL_DIRTY),
        model_clean=Path(legacy_config.MODEL_CLEAN),
        results_dirty=Path(legacy_config.RESULTS_DIRTY),
        results_clean=Path(legacy_config.RESULTS_CLEAN),
    )
    return Settings(
        backend=str(getattr(legacy_config, "GAME_BACKEND", "velocity")),
        device_mode=str(getattr(legacy_config, "DEVICE_MODE", "eeg")),
        paths=paths,
        oc_cutoff=float(getattr(legacy_config, "OC_CUTOFF", 0.6)),
        sims_per_model=int(getattr(legacy_config, "SIMS_PER_MODEL", 20)),
        sim_workers=int(getattr(legacy_config, "SIM_WORKERS", 2)),
        batch_size=int(getattr(legacy_config, "BATCH_SIZE", 10)),
    )
