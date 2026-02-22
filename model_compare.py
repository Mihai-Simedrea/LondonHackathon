#!/usr/bin/env python3
"""
Model comparison — finds the best architecture for raw game survival performance.
Evaluates on CLEAN data (the actual use case), then shows dirty vs clean improvement.
"""

import warnings
warnings.filterwarnings('ignore')

import json
import time
import os
import tempfile
import multiprocessing
import numpy as np
import csv
import joblib
from pathlib import Path

import config

# Use fork context to avoid macOS spawn re-importing catboost/xgboost in workers
_mp_ctx = multiprocessing.get_context('fork')
N_WORKERS = config.SIM_WORKERS

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from simulator import simulate
from model import predict as model_predict, _engineer_features

N_SIMS = 30


def load_dataset(csv_path):
    """Load dataset CSV, return X, y with 14 engineered features (matching model.py)."""
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    X, y = [], []
    for row in rows:
        lane = int(row["lane"])
        lane_onehot = [0, 0, 0]
        lane_onehot[lane] = 1
        obs_d0 = float(row["obs_d0"])
        obs_d1 = float(row["obs_d1"])
        obs_d2 = float(row["obs_d2"])
        X.append(_engineer_features(lane_onehot, obs_d0, obs_d1, obs_d2))
        y.append(int(row["decision"]) + 1)
    return np.array(X), np.array(y), rows


def mirror_augment_from_rows(rows, X, y):
    """Double dataset by mirroring left/right symmetry using raw row data."""
    mirror_X, mirror_y = [], []
    for row in rows:
        lane = int(row["lane"])
        mirrored_lane = 2 - lane
        mirror_onehot = [0, 0, 0]
        mirror_onehot[mirrored_lane] = 1
        obs_d0 = float(row["obs_d0"])
        obs_d1 = float(row["obs_d1"])
        obs_d2 = float(row["obs_d2"])
        mirror_X.append(_engineer_features(mirror_onehot, obs_d2, obs_d1, obs_d0))

        decision = int(row["decision"])
        action = decision + 1
        if action == 0:
            mirrored_action = 2
        elif action == 2:
            mirrored_action = 0
        else:
            mirrored_action = 1
        mirror_y.append(mirrored_action)

    return np.vstack([X, np.array(mirror_X)]), np.concatenate([y, np.array(mirror_y)])


def _sim_worker(args):
    model_path, seed = args
    model = joblib.load(model_path)
    return simulate(model, seed=seed)["alive_time"]


def run_sims(model, n=N_SIMS, label=""):
    """Run n simulations in parallel across 10 cores using fork context."""
    tmp = tempfile.NamedTemporaryFile(suffix='.joblib', delete=False)
    joblib.dump(model, tmp.name)
    tmp_path = tmp.name
    tmp.close()

    try:
        with _mp_ctx.Pool(processes=N_WORKERS) as pool:
            alive_times = pool.map(_sim_worker, [(tmp_path, s) for s in range(n)])
    finally:
        os.unlink(tmp_path)

    avg = np.mean(alive_times)
    print(f"    {label}: {n}/{n} done, avg={avg:.0f}f", flush=True)
    return avg, np.std(alive_times)


def make_models():
    """Return dict of model name -> (factory, description)."""
    models = {
        "RF-500-balanced": (
            lambda: RandomForestClassifier(
                n_estimators=500, max_depth=10, class_weight='balanced',
                random_state=42, n_jobs=1
            ),
            "RandomForest with class balancing"
        ),
        "RF-1000": (
            lambda: RandomForestClassifier(
                n_estimators=1000, max_depth=15, min_samples_leaf=2,
                class_weight='balanced', random_state=42, n_jobs=1
            ),
            "Large RandomForest"
        ),
        "CatBoost-ordered": (
            lambda: CatBoostClassifier(
                iterations=500, depth=4, learning_rate=0.03,
                l2_leaf_reg=5, bagging_temperature=1.0,
                boosting_type='Ordered', verbose=0,
                random_seed=42
            ),
            "CatBoost with Ordered boosting"
        ),
        "XGBoost-reg": (
            lambda: XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                random_state=42, eval_metric='mlogloss',
                use_label_encoder=False
            ),
            "XGBoost with regularization"
        ),
        "GB-500": (
            lambda: GradientBoostingClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.03,
                subsample=0.8, random_state=42
            ),
            "GradientBoosting slow learning"
        ),
        "Ensemble-soft": (
            lambda: VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(
                    n_estimators=500, max_depth=10,
                    class_weight='balanced', random_state=42, n_jobs=1)),
                ('gb', GradientBoostingClassifier(
                    n_estimators=300, max_depth=5,
                    learning_rate=0.05, subsample=0.8, random_state=42)),
                ('lr', LogisticRegression(C=0.1, max_iter=1000, random_state=42)),
            ], voting='soft', n_jobs=1),
            "Soft voting: RF + GB + LR"
        ),
    }
    return models


def main():
    print("=" * 70)
    print("  MODEL COMPARISON — Optimizing for Raw Survival Performance")
    print("=" * 70)

    if not config.DATASET_CLEAN.exists():
        print("\nNo data. Run: python pipeline.py demo --synthetic --dev")
        return

    X_dirty, y_dirty, rows_dirty = load_dataset(config.DATASET_DIRTY)
    X_clean, y_clean, rows_clean = load_dataset(config.DATASET_CLEAN)

    # Mirror augmentation (recomputes engineered features from mirrored raw inputs)
    X_dirty_aug, y_dirty_aug = mirror_augment_from_rows(rows_dirty, X_dirty, y_dirty)
    X_clean_aug, y_clean_aug = mirror_augment_from_rows(rows_clean, X_clean, y_clean)

    print(f"\nDirty: {len(X_dirty)} rows (augmented: {len(X_dirty_aug)})")
    print(f"Clean: {len(X_clean)} rows (augmented: {len(X_clean_aug)})")
    print(f"Features per sample: {X_dirty.shape[1]}")
    print(f"Sims per model: {N_SIMS}")

    models = make_models()
    results = []

    for name, (model_fn, desc) in models.items():
        print(f"\n--- {name}: {desc} ---")
        t0 = time.time()

        # Train on augmented clean data (the real use case)
        model_clean = model_fn()
        model_clean.fit(X_clean_aug, y_clean_aug)
        clean_acc = model_clean.score(X_clean, y_clean)

        # Simulate clean model
        clean_avg, clean_std = run_sims(model_clean, label=f"{name} clean")

        # Also train dirty for comparison
        model_dirty = model_fn()
        model_dirty.fit(X_dirty_aug, y_dirty_aug)
        dirty_avg, dirty_std = run_sims(model_dirty, label=f"{name} dirty")

        improvement = ((clean_avg - dirty_avg) / dirty_avg * 100) if dirty_avg > 0 else 0
        elapsed = time.time() - t0

        print(f"  Clean acc: {clean_acc:.3f}")
        print(f"  Clean sim: {clean_avg:.0f}+/-{clean_std:.0f} frames ({clean_avg/60:.1f}s)")
        print(f"  Dirty sim: {dirty_avg:.0f}+/-{dirty_std:.0f} frames ({dirty_avg/60:.1f}s)")
        print(f"  Improvement: {improvement:+.1f}% | Time: {elapsed:.1f}s")

        results.append({
            "name": name, "desc": desc,
            "clean_acc": round(float(clean_acc), 4),
            "clean_avg": round(float(clean_avg), 1),
            "clean_std": round(float(clean_std), 1),
            "dirty_avg": round(float(dirty_avg), 1),
            "dirty_std": round(float(dirty_std), 1),
            "improvement": round(float(improvement), 2),
            "time": round(float(elapsed), 2),
        })

    # Sort by raw clean performance (the metric that matters)
    results.sort(key=lambda x: -x["clean_avg"])

    print("\n" + "=" * 70)
    print(f"{'Model':<20} {'Clean Avg':>10} {'Dirty Avg':>10} {'Delta':>8} {'Acc':>6} {'Time':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['clean_avg']:>8.0f}f  {r['dirty_avg']:>8.0f}f  "
              f"{r['improvement']:>+6.1f}%  {r['clean_acc']:>.3f}  {r['time']:>5.1f}s")
    print("=" * 70)

    best = results[0]
    print(f"\nBest raw performance: {best['name']} — {best['clean_avg']:.0f} frames ({best['clean_avg']/60:.1f}s)")
    print(f"  Dirty->Clean improvement: {best['improvement']:+.1f}%")

    # Save results to JSON
    output_path = config.RESULTS_DIR / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump({
            "n_sims": N_SIMS,
            "dirty_rows": len(X_dirty),
            "clean_rows": len(X_clean),
            "dirty_augmented": len(X_dirty_aug),
            "clean_augmented": len(X_clean_aug),
            "sorted_by": "clean_avg_descending",
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
