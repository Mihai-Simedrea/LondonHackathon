#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.simulation.runner`."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from neurolabel.config.loader import load_settings
from neurolabel.simulation.runner import run_comparison as _run_comparison_impl
from neurolabel.simulation.runner import run_simulations as _run_simulations_impl


def run_simulations(model_path, n_sims=None, batch_size=None):
    settings = load_settings()
    return _run_simulations_impl(settings, model_path, n_sims=n_sims, batch_size=batch_size)


def run_comparison(dirty_model_path=None, clean_model_path=None):
    settings = load_settings()
    if dirty_model_path is not None or clean_model_path is not None:
        paths = settings.paths
        paths = replace(
            paths,
            model_dirty=Path(dirty_model_path) if dirty_model_path is not None else paths.model_dirty,
            model_clean=Path(clean_model_path) if clean_model_path is not None else paths.model_clean,
        )
        settings = settings.with_overrides(paths=paths)
    return _run_comparison_impl(settings)


if __name__ == "__main__":
    run_comparison()
