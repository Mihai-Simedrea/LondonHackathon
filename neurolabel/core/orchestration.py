from __future__ import annotations

import csv
import time
from typing import Any

from neurolabel.backends.registry import load_backend
from neurolabel.brain.registry import load_scorer
from neurolabel.config.schema import Settings
from neurolabel.core.contracts import DemoSummary, ProcessSummary
from neurolabel.core.legacy_bridge import apply_legacy_settings
from neurolabel.core.paths import read_game_start_timestamp
from neurolabel.data.dataset_builder import build_dataset
from neurolabel.data.dataset_filter import filter_dataset


def collect(settings: Settings) -> dict[str, Any]:
    apply_legacy_settings(settings)
    backend = load_backend(settings.backend)
    summary = backend.collect_live(settings)
    return {"ok": summary.ok, "message": summary.message, **summary.metadata}


def generate_synthetic(settings: Settings, *, duration_seconds: int) -> dict[str, Any]:
    apply_legacy_settings(settings)
    backend = load_backend(settings.backend)
    summary = backend.generate_synthetic(settings, duration_seconds=duration_seconds)
    return {"ok": summary.ok, "message": summary.message, **summary.metadata}


def process(settings: Settings) -> ProcessSummary:
    apply_legacy_settings(settings)
    trim_before = read_game_start_timestamp(settings.paths.game_jsonl)
    scorer = load_scorer(settings.device_mode)
    oc_results = scorer.compute_scores(settings, trim_before=trim_before, include_timestamp_in_csv=True)

    full_path = build_dataset(settings)
    dirty_path, clean_path = filter_dataset(settings, full_path, cutoff=settings.oc_cutoff)

    return ProcessSummary(
        oc_windows=len(oc_results),
        dataset_full=full_path,
        dataset_dirty=dirty_path,
        dataset_clean=clean_path,
    )


def train(settings: Settings):
    apply_legacy_settings(settings)
    backend = load_backend(settings.backend)
    return backend.train_models(settings)


def simulate(settings: Settings):
    apply_legacy_settings(settings)
    backend = load_backend(settings.backend)
    dirty_results, clean_results = backend.simulate_comparison(settings)
    return dirty_results, clean_results


def visualize(settings: Settings, *, dirty_results=None, clean_results=None) -> None:
    apply_legacy_settings(settings)
    backend = load_backend(settings.backend)
    if not backend.supports_visualizer:
        raise RuntimeError(f"Visualization is not available for backend '{settings.backend}' yet")
    backend.visualize(settings, dirty_results=dirty_results, clean_results=clean_results)


def demo(settings: Settings, *, synthetic: bool = False, dev: bool = False, duration_seconds: int = 600) -> DemoSummary:
    if synthetic:
        generate_synthetic(settings, duration_seconds=duration_seconds)
    else:
        collect(settings)

    process_summary = process(settings)

    # Keep the existing warning behavior.
    try:
        with open(settings.paths.dataset_clean, "r") as f:
            clean_count = sum(1 for _ in csv.reader(f)) - 1
        if clean_count < 20:
            print(f"\n  Warning: only {clean_count} clean rows; model may be unreliable.")
    except Exception:
        pass

    train_summary = train(settings)
    dirty_results, clean_results = simulate(settings)

    if not dev:
        try:
            visualize(settings, dirty_results=dirty_results, clean_results=clean_results)
        except RuntimeError as exc:
            print(f"\n[visualize] {exc}")

    from neurolabel.core.contracts import SimSummary
    return DemoSummary(
        process=process_summary,
        train=train_summary,
        sim=SimSummary(dirty_results=dirty_results, clean_results=clean_results),
    )
