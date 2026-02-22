#!/usr/bin/env python3
"""Parallel simulation runner — runs 50 game simulations per model in batches of 13."""

import json
import multiprocessing
import random
import time
from pathlib import Path

import numpy as np

import config
from model import load_model
from simulator import simulate


def _run_single_sim(args):
    """Worker function for multiprocessing. Takes (model_path, seed) tuple."""
    model_path, seed = args
    model = load_model(model_path)
    return simulate(model, seed=seed)


def run_simulations(model_path, n_sims=None, batch_size=None):
    """
    Run n_sims simulations using the model at model_path.

    Uses multiprocessing.Pool with batch_size workers.
    Processes simulations in batches for progress reporting.

    Args:
        model_path: path to joblib model file
        n_sims: number of simulations (default: config.SIMS_PER_MODEL = 50)
        batch_size: workers per batch (default: config.BATCH_SIZE = 13)

    Returns:
        dict: {
            'avg_alive': float,
            'std_alive': float,
            'min_alive': int,
            'max_alive': int,
            'runs': list of run dicts from simulate()
        }
    """
    n_sims = n_sims or config.SIMS_PER_MODEL
    batch_size = batch_size or config.BATCH_SIZE

    all_runs = []
    seeds = random.sample(range(100_000), n_sims)

    # Process in batches
    n_batches = (n_sims + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_sims)
        batch_seeds = seeds[batch_start:batch_end]

        args_list = [(str(model_path), seed) for seed in batch_seeds]

        with multiprocessing.Pool(processes=min(len(batch_seeds), config.SIM_WORKERS)) as pool:
            results = pool.map(_run_single_sim, args_list)

        all_runs.extend(results)

        # Progress
        alive_times = [r["alive_time"] for r in all_runs]
        avg_so_far = sum(alive_times) / len(alive_times)
        print(
            f"  Batch {batch_idx + 1}/{n_batches} complete "
            f"({len(all_runs)}/{n_sims} sims, running avg: {avg_so_far:.0f} frames)"
        )

    # Aggregate
    alive_times = [r["alive_time"] for r in all_runs]

    results = {
        "avg_alive": float(np.mean(alive_times)),
        "std_alive": float(np.std(alive_times)),
        "min_alive": int(np.min(alive_times)),
        "max_alive": int(np.max(alive_times)),
        "runs": all_runs,
    }

    return results


def run_comparison(dirty_model_path=None, clean_model_path=None):
    """
    Run simulations for both dirty and clean models, save results.

    Returns:
        tuple: (dirty_results, clean_results)
    """
    dirty_model_path = dirty_model_path or config.MODEL_DIRTY
    clean_model_path = clean_model_path or config.MODEL_CLEAN

    print("\n" + "=" * 50)
    print("SIMULATION: Dirty Model (all training data)")
    print("=" * 50)
    start = time.time()
    dirty_results = run_simulations(dirty_model_path)
    dirty_time = time.time() - start
    print(f"  Time: {dirty_time:.1f}s")

    print("\n" + "=" * 50)
    print("SIMULATION: Clean Model (OC-filtered training data)")
    print("=" * 50)
    start = time.time()
    clean_results = run_simulations(clean_model_path)
    clean_time = time.time() - start
    print(f"  Time: {clean_time:.1f}s")

    # Save results (without frame data for the JSON — too large)
    def save_summary(results, path):
        summary = {k: v for k, v in results.items() if k != "runs"}
        summary["alive_times"] = [r["alive_time"] for r in results["runs"]]
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

    config.RESULTS_DIR.mkdir(exist_ok=True)
    save_summary(dirty_results, config.RESULTS_DIRTY)
    save_summary(clean_results, config.RESULTS_CLEAN)

    # Print comparison
    print("\n" + "=" * 50)
    print("RESULTS COMPARISON")
    print("=" * 50)
    print(
        f"  DIRTY MODEL (fat):  avg = {dirty_results['avg_alive']:.0f} frames "
        f"({dirty_results['avg_alive']/60:.1f}s), std = {dirty_results['std_alive']:.0f}"
    )
    print(
        f"  CLEAN MODEL (slim): avg = {clean_results['avg_alive']:.0f} frames "
        f"({clean_results['avg_alive']/60:.1f}s), std = {clean_results['std_alive']:.0f}"
    )

    if dirty_results["avg_alive"] > 0:
        improvement = (
            (clean_results["avg_alive"] - dirty_results["avg_alive"])
            / dirty_results["avg_alive"]
            * 100
        )
        print(f"  Improvement: {improvement:+.1f}%")

    return dirty_results, clean_results


if __name__ == "__main__":
    run_comparison()
