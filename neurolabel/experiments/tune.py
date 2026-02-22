#!/usr/bin/env python3
"""
model_tuning.py — Comprehensive model comparison and hyperparameter tuning.

Generates synthetic data, tests multiple ML architectures head-to-head,
runs GridSearchCV for each, evaluates with game simulation, and picks the winner.
"""

import csv
import time
import warnings
import numpy as np
from pathlib import Path

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import config
from neurolabel.models.velocity_data import load_velocity_raw_xy

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# Step 1: Generate data if needed
# ─────────────────────────────────────────

def ensure_data():
    """Generate synthetic data and build dataset if not already present."""
    dataset_path = config.DATASET_CLEAN
    if not dataset_path.exists():
        print("=" * 60)
        print("STEP 1: Generating synthetic data and building dataset")
        print("=" * 60)

        import json

        # Generate synthetic EEG data
        from synthetic_data import generate_synthetic_eeg
        eeg_data = generate_synthetic_eeg(600)

        eeg_path = config.DATA_DIR / "eeg_recording.csv"
        with open(eeg_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + config.CHANNEL_NAMES)
            for row in eeg_data:
                writer.writerow(row.tolist())
        print(f"  Saved EEG: {eeg_path} ({len(eeg_data)} samples)")

        # Generate game data with json-safe types
        from synthetic_data import heuristic_decision
        from game_engine import GameState, FPS
        import time as _time

        game = GameState(seed=42)
        records = []
        transition_sec = int(0.6 * 600)
        base_time = _time.time()
        rng = np.random.RandomState(42)

        for sec in range(600):
            if not game.alive:
                game = GameState(seed=42 + sec)

            decision = heuristic_decision(game)
            if sec >= transition_sec:
                if rng.random() < 0.4:
                    decision = int(rng.choice([-1, 0, 1]))

            state = game.encode()
            record = {
                "t": base_time + sec,
                "sec": sec,
                "lane": state["lane"],
                "obs": state["obs"],
                "decision": int(decision),
                "score": state["score"],
                "alive": state["alive"]
            }
            records.append(record)

            for frame in range(FPS):
                if frame == 0:
                    game.step(decision)
                else:
                    game.step(0)
                if not game.alive:
                    break

        game_path = config.DATA_DIR / "game_recording.jsonl"
        with open(game_path, 'w') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')
        print(f"  Saved game: {game_path} ({len(records)} records)")

        # Compute OC scores
        from oc_scorer import compute_oc_scores
        oc_results = compute_oc_scores(
            eeg_path,
            output_path=config.OC_SCORES_CSV,
            include_timestamp_in_csv=True,
        )
        print(f"  OC scores computed: {len(oc_results)} windows")

        # Build and filter dataset
        from dataset import build_dataset, filter_dataset
        full_path = build_dataset(game_path, config.OC_SCORES_CSV)
        dirty_path, clean_path = filter_dataset(full_path)
        print()
    else:
        print("Dataset already exists, skipping generation.")
    return dataset_path


# ─────────────────────────────────────────
# Step 2: Load dataset
# ─────────────────────────────────────────

def load_dataset(csv_path):
    """Load dataset CSV into X, y arrays."""
    X, y, _rows = load_velocity_raw_xy(csv_path)
    return X, y


# ─────────────────────────────────────────
# Step 3: Define model candidates
# ─────────────────────────────────────────

CANDIDATES = {
    "MLP_small": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=42, max_iter=2000)),
        ]),
        "params": {
            "clf__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "clf__activation": ["relu", "tanh"],
            "clf__solver": ["adam", "lbfgs"],
            "clf__alpha": [0.0001, 0.001, 0.01],
        },
    },
    "MLP_large": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(random_state=42, max_iter=2000)),
        ]),
        "params": {
            "clf__hidden_layer_sizes": [
                (128, 64, 32),
                (256, 128, 64),
                (256, 128, 64, 32),
            ],
            "clf__activation": ["relu"],
            "clf__solver": ["adam"],
            "clf__alpha": [0.0001, 0.001],
            "clf__learning_rate": ["constant", "adaptive"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight="balanced"),
        "params": {
            "n_estimators": [50, 100, 200, 500],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
    },
    "SVM": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=42, class_weight="balanced", probability=True)),
        ]),
        "params": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "poly"],
            "clf__gamma": ["scale", "auto", 0.1],
        },
    },
    "KNN": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier()),
        ]),
        "params": {
            "clf__n_neighbors": [3, 5, 7, 11, 15],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    },
    "LogisticRegression": {
        "model": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                random_state=42, max_iter=1000, class_weight="balanced"
            )),
        ]),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__solver": ["lbfgs", "saga"],
        },
    },
}


# ─────────────────────────────────────────
# Step 4: Simulation evaluator
# ─────────────────────────────────────────

def evaluate_simulation(model, n_sims=10):
    """Run game simulations and return average alive time."""
    from game_engine import GameState, FPS

    alive_times = []
    max_frames = 100_000

    for seed in range(n_sims):
        game = GameState(seed=seed)
        current_decision = 0

        while game.alive and game.frame < max_frames:
            if game.frame % FPS == 0:
                nearest = game.get_nearest_obstacles()
                lane = game.player.lane
                lane_onehot = [0, 0, 0]
                lane_onehot[lane] = 1
                features = np.array([lane_onehot + list(nearest)])

                try:
                    pred_class = model.predict(features)[0]
                    current_decision = pred_class - 1
                except Exception:
                    current_decision = 0

            if game.frame % FPS == 0:
                game.step(current_decision)
            else:
                game.step(0)

        alive_times.append(game.frame)

    return np.mean(alive_times)


# ─────────────────────────────────────────
# Step 5: Run the comparison
# ─────────────────────────────────────────

def run_comparison(X, y):
    """Run GridSearchCV for each candidate and return results."""
    results = []

    print("=" * 60)
    print("STEP 2: Running model comparison with GridSearchCV")
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: {np.unique(y)} (counts: {np.bincount(y)})")
    print("=" * 60)
    print()

    for name, spec in CANDIDATES.items():
        print(f"--- {name} ---")
        t0 = time.time()

        grid = GridSearchCV(
            spec["model"],
            spec["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            error_score="raise",
        )

        try:
            grid.fit(X, y)
        except Exception as e:
            print(f"  FAILED: {e}")
            print()
            continue

        train_time = time.time() - t0
        best_score = grid.best_score_
        best_params = grid.best_params_

        # Summarize best params (short form)
        short_params = ", ".join(
            f"{k.split('__')[-1]}={v}" for k, v in best_params.items()
        )

        print(f"  CV Accuracy: {best_score:.4f}")
        print(f"  Best params: {short_params}")
        print(f"  Train time:  {train_time:.1f}s")

        # Simulation evaluation
        print(f"  Running 10 simulations...")
        sim_t0 = time.time()
        avg_alive = evaluate_simulation(grid.best_estimator_, n_sims=10)
        sim_time = time.time() - sim_t0
        print(f"  Avg alive:   {avg_alive:.0f} frames ({avg_alive / 60:.1f}s)")
        print(f"  Sim time:    {sim_time:.1f}s")
        print()

        results.append({
            "name": name,
            "cv_acc": best_score,
            "best_params": best_params,
            "short_params": short_params,
            "train_time": train_time,
            "avg_alive": avg_alive,
            "best_estimator": grid.best_estimator_,
        })

    return results


# ─────────────────────────────────────────
# Step 6: Ensemble methods
# ─────────────────────────────────────────

def run_ensembles(X, y, top_results):
    """Try VotingClassifier and StackingClassifier with the top 3 models."""
    print("=" * 60)
    print("STEP 3: Testing ensemble methods with top 3 models")
    print("=" * 60)
    print()

    top3 = top_results[:3]
    estimators = [(r["name"], r["best_estimator"]) for r in top3]
    print(f"  Top 3 models: {[r['name'] for r in top3]}")
    print()

    ensemble_results = []

    # Voting Classifier
    print("--- VotingClassifier (soft) ---")
    t0 = time.time()
    voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    try:
        scores = cross_val_score(voting, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        cv_acc = scores.mean()
        voting.fit(X, y)
        train_time = time.time() - t0

        print(f"  CV Accuracy: {cv_acc:.4f}")
        print(f"  Train time:  {train_time:.1f}s")

        print(f"  Running 10 simulations...")
        avg_alive = evaluate_simulation(voting, n_sims=10)
        print(f"  Avg alive:   {avg_alive:.0f} frames ({avg_alive / 60:.1f}s)")
        print()

        ensemble_results.append({
            "name": "Voting(top3)",
            "cv_acc": cv_acc,
            "best_params": {},
            "short_params": f"soft, {'+'.join(r['name'] for r in top3)}",
            "train_time": train_time,
            "avg_alive": avg_alive,
            "best_estimator": voting,
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        print()

    # Stacking Classifier
    print("--- StackingClassifier (meta=LogReg) ---")
    t0 = time.time()
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42, max_iter=1000),
        cv=5,
        n_jobs=-1,
    )
    try:
        scores = cross_val_score(stacking, X, y, cv=5, scoring="accuracy", n_jobs=-1)
        cv_acc = scores.mean()
        stacking.fit(X, y)
        train_time = time.time() - t0

        print(f"  CV Accuracy: {cv_acc:.4f}")
        print(f"  Train time:  {train_time:.1f}s")

        print(f"  Running 10 simulations...")
        avg_alive = evaluate_simulation(stacking, n_sims=10)
        print(f"  Avg alive:   {avg_alive:.0f} frames ({avg_alive / 60:.1f}s)")
        print()

        ensemble_results.append({
            "name": "Stacking(top3)",
            "cv_acc": cv_acc,
            "best_params": {},
            "short_params": f"meta=LogReg, {'+'.join(r['name'] for r in top3)}",
            "train_time": train_time,
            "avg_alive": avg_alive,
            "best_estimator": stacking,
        })
    except Exception as e:
        print(f"  FAILED: {e}")
        print()

    return ensemble_results


# ─────────────────────────────────────────
# Step 7: Print results table and pick winner
# ─────────────────────────────────────────

def print_results_table(all_results):
    """Print comprehensive results table."""
    # Sort by combined rank (cv_acc rank + sim rank)
    for r in all_results:
        r["_cv_rank"] = 0
        r["_sim_rank"] = 0

    sorted_by_cv = sorted(all_results, key=lambda x: x["cv_acc"], reverse=True)
    for i, r in enumerate(sorted_by_cv):
        r["_cv_rank"] = i + 1

    sorted_by_sim = sorted(all_results, key=lambda x: x["avg_alive"], reverse=True)
    for i, r in enumerate(sorted_by_sim):
        r["_sim_rank"] = i + 1

    for r in all_results:
        r["_combined_rank"] = r["_cv_rank"] + r["_sim_rank"]

    all_results.sort(key=lambda x: x["_combined_rank"])

    print()
    print("=" * 95)
    print("FINAL RESULTS TABLE (ranked by combined CV accuracy + simulation performance)")
    print("=" * 95)

    header = f"{'Model':<22} {'CV Acc':>8} {'Best Params':<30} {'Time(s)':>8} {'SimAlive':>10}"
    print(header)
    print("-" * 95)

    for r in all_results:
        params_str = r["short_params"]
        if len(params_str) > 28:
            params_str = params_str[:25] + "..."
        line = f"{r['name']:<22} {r['cv_acc']:>8.4f} {params_str:<30} {r['train_time']:>8.1f} {r['avg_alive']:>10.0f}"
        print(line)

    print("-" * 95)

    winner = all_results[0]
    print()
    print(f"WINNER: {winner['name']}")
    print(f"  CV Accuracy:     {winner['cv_acc']:.4f}")
    print(f"  Avg Alive Time:  {winner['avg_alive']:.0f} frames ({winner['avg_alive'] / 60:.1f}s)")
    print(f"  Best Params:     {winner['short_params']}")
    print()

    return winner


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────

def main():
    overall_start = time.time()

    # Step 1: Ensure data exists
    dataset_path = ensure_data()

    # Step 2: Load dataset
    X, y = load_dataset(dataset_path)
    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print()

    # Step 3: Run individual model comparison
    individual_results = run_comparison(X, y)

    if not individual_results:
        print("ERROR: No models completed successfully!")
        return

    # Sort by CV accuracy for ensemble selection
    individual_results.sort(key=lambda x: x["cv_acc"], reverse=True)

    # Step 4: Run ensemble methods with top 3
    ensemble_results = []
    if len(individual_results) >= 3:
        ensemble_results = run_ensembles(X, y, individual_results)

    # Step 5: Print final results table
    all_results = individual_results + ensemble_results
    winner = print_results_table(all_results)

    total_time = time.time() - overall_start
    print(f"Total tuning time: {total_time:.1f}s")

    # Save best model info to a file for reference
    import json
    best_info = {
        "name": winner["name"],
        "cv_acc": winner["cv_acc"],
        "avg_alive": winner["avg_alive"],
        "params": {k: str(v) for k, v in winner["best_params"].items()},
    }
    info_path = config.RESULTS_DIR / "best_model_info.json"
    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"Best model info saved to: {info_path}")

    # Save the best model
    import joblib
    best_model_path = config.MODELS_DIR / "model_best_tuned.joblib"
    joblib.dump(winner["best_estimator"], best_model_path)
    print(f"Best model saved to: {best_model_path}")

    return winner


if __name__ == "__main__":
    main()
