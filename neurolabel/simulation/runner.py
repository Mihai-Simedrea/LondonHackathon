from __future__ import annotations

"""Parallel simulation runner (package-native)."""

import multiprocessing
import random
import time
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from neurolabel.config.schema import Settings
from neurolabel.core.results import save_result_json


def _load_backend_functions(backend: str):
    """Load backend-specific model/simulator functions lazily for worker safety."""
    if backend == "metadrive":
        from neurolabel.backends.metadrive.model import load_model
        from neurolabel.backends.metadrive.simulation import simulate
    else:
        from neurolabel.backends.velocity.model import load_model
        from neurolabel.backends.velocity.simulation import simulate
    return load_model, simulate


def _run_seed_batch(args):
    """Worker function: load model once, run one batch of seeds."""
    backend, model_path, seeds = args
    load_model, simulate = _load_backend_functions(backend)
    model = load_model(model_path)
    return [simulate(model, seed=seed) for seed in seeds]


def _chunked(items, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    it = iter(items)
    while True:
        chunk = list(islice(it, chunk_size))
        if not chunk:
            return
        yield chunk


def run_simulations(
    settings: Settings,
    model_path: str | Path,
    *,
    n_sims: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run repeated simulations for one model and aggregate metrics."""
    n_sims = n_sims or settings.sims_per_model
    batch_size = batch_size or settings.batch_size
    model_path = str(model_path)

    all_runs: list[dict[str, Any]] = []
    seeds = random.sample(range(100_000), n_sims)
    n_batches = (n_sims + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_sims)
        batch_seeds = seeds[batch_start:batch_end]

        worker_count = min(len(batch_seeds), settings.sim_workers)
        seeds_per_worker = max(1, (len(batch_seeds) + worker_count - 1) // worker_count)
        args_list = [
            (settings.backend, model_path, seed_chunk)
            for seed_chunk in _chunked(batch_seeds, seeds_per_worker)
        ]

        with multiprocessing.Pool(processes=worker_count) as pool:
            batch_results = pool.map(_run_seed_batch, args_list)

        for worker_runs in batch_results:
            all_runs.extend(worker_runs)

        alive_times = [r["alive_time"] for r in all_runs]
        avg_so_far = sum(alive_times) / len(alive_times)
        print(
            f"  Batch {batch_idx + 1}/{n_batches} complete "
            f"({len(all_runs)}/{n_sims} sims, running avg: {avg_so_far:.0f} frames)"
        )

    alive_times = [r["alive_time"] for r in all_runs]
    rewards = [r.get("total_reward", 0) for r in all_runs]
    completions = [r.get("route_completion", 0) for r in all_runs]

    return {
        "avg_alive": float(np.mean(alive_times)),
        "std_alive": float(np.std(alive_times)),
        "min_alive": int(np.min(alive_times)),
        "max_alive": int(np.max(alive_times)),
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_route_completion": float(np.mean(completions)),
        "runs": all_runs,
    }


def _save_summary(path: Path, settings: Settings, label: str, results: dict[str, Any]) -> None:
    summary = {k: v for k, v in results.items() if k != "runs"}
    summary["alive_times"] = [r["alive_time"] for r in results.get("runs", [])]
    summary["rewards"] = [r.get("total_reward", 0) for r in results.get("runs", [])]
    summary["route_completions"] = [r.get("route_completion", 0) for r in results.get("runs", [])]
    save_result_json(path, {"backend": settings.backend, "label": label, **summary})


def run_comparison(settings: Settings) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run dirty vs clean model simulations and save versioned summaries."""
    print("\n" + "=" * 50)
    print("SIMULATION: Dirty Model (all training data)")
    print("=" * 50)
    start = time.time()
    dirty_results = run_simulations(settings, settings.paths.model_dirty)
    dirty_time = time.time() - start
    print(f"  Time: {dirty_time:.1f}s")

    print("\n" + "=" * 50)
    print("SIMULATION: Clean Model (OC-filtered training data)")
    print("=" * 50)
    start = time.time()
    clean_results = run_simulations(settings, settings.paths.model_clean)
    clean_time = time.time() - start
    print(f"  Time: {clean_time:.1f}s")

    _save_summary(settings.paths.results_dirty, settings, "dirty", dirty_results)
    _save_summary(settings.paths.results_clean, settings, "clean", clean_results)

    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    print(
        f"  DIRTY MODEL (fat):  avg = {dirty_results['avg_alive']:.0f} frames "
        f"({dirty_results['avg_alive']/60:.1f}s), route = {dirty_results.get('avg_route_completion', 0):.2%}"
    )
    print(
        f"  CLEAN MODEL (slim): avg = {clean_results['avg_alive']:.0f} frames "
        f"({clean_results['avg_alive']/60:.1f}s), route = {clean_results.get('avg_route_completion', 0):.2%}"
    )

    if dirty_results["avg_alive"] > 0:
        improvement = ((clean_results["avg_alive"] - dirty_results["avg_alive"]) / dirty_results["avg_alive"]) * 100
        print(f"  Survival improvement: {improvement:+.1f}%")

    print(f"\n  DIRTY reward:  avg = {dirty_results['avg_reward']:.2f} (+/- {dirty_results['std_reward']:.2f})")
    print(f"  CLEAN reward:  avg = {clean_results['avg_reward']:.2f} (+/- {clean_results['std_reward']:.2f})")
    if dirty_results["avg_reward"] != 0:
        reward_imp = ((clean_results["avg_reward"] - dirty_results["avg_reward"]) / abs(dirty_results["avg_reward"])) * 100
        print(f"  Reward improvement: {reward_imp:+.1f}%")

    print(f"\n  DIRTY route:   avg = {dirty_results['avg_route_completion']:.1%}")
    print(f"  CLEAN route:   avg = {clean_results['avg_route_completion']:.1%}")
    return dirty_results, clean_results
