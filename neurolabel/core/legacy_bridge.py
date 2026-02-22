from __future__ import annotations

"""Bridge layer to keep legacy flat modules working during migration."""

from neurolabel.config.schema import Settings


def apply_legacy_settings(settings: Settings) -> None:
    """Synchronize legacy `config.py` globals with the new Settings object.

    Many existing modules still import and branch on module-level config values.
    This keeps the new orchestrator authoritative while legacy modules continue
    to function unchanged.
    """
    import config as legacy_config

    legacy_config.GAME_BACKEND = settings.backend
    legacy_config.DEVICE_MODE = settings.device_mode
    legacy_config.OC_CUTOFF = settings.oc_cutoff
    legacy_config.SIMS_PER_MODEL = settings.sims_per_model
    legacy_config.SIM_WORKERS = settings.sim_workers
    legacy_config.BATCH_SIZE = settings.batch_size

    # Sync paths (useful if settings loading evolves beyond legacy defaults).
    legacy_config.DATA_DIR = settings.paths.data_dir
    legacy_config.MODELS_DIR = settings.paths.models_dir
    legacy_config.RESULTS_DIR = settings.paths.results_dir
    legacy_config.EEG_CSV = settings.paths.eeg_csv
    legacy_config.FNIRS_CSV = settings.paths.fnirs_csv
    legacy_config.BRAIN_CSV = settings.brain_csv
    legacy_config.GAME_JSONL = settings.paths.game_jsonl
    legacy_config.OC_SCORES_CSV = settings.paths.oc_scores_csv
    legacy_config.DATASET_DIRTY = settings.paths.dataset_dirty
    legacy_config.DATASET_CLEAN = settings.paths.dataset_clean
    legacy_config.MODEL_DIRTY = settings.paths.model_dirty
    legacy_config.MODEL_CLEAN = settings.paths.model_clean
    legacy_config.RESULTS_DIRTY = settings.paths.results_dirty
    legacy_config.RESULTS_CLEAN = settings.paths.results_clean

