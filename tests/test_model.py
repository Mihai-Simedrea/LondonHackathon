"""Tests for model.py — training, prediction, and persistence."""

import sys
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import train_model, predict, save_model, load_model


def _create_dataset_csv(path, n_rows=30):
    """Create a small inline CSV dataset for testing.

    Generates rows with 3 lanes and random obstacle distances,
    with decisions biased toward 'stay' (0) to mimic real data.
    """
    rng = np.random.RandomState(42)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lane", "obs_d0", "obs_d1", "obs_d2", "decision", "oc_score"])
        writer.writeheader()
        for _ in range(n_rows):
            lane = rng.randint(0, 3)
            obs_d0 = round(rng.uniform(0.1, 1.0), 3)
            obs_d1 = round(rng.uniform(0.1, 1.0), 3)
            obs_d2 = round(rng.uniform(0.1, 1.0), 3)
            # Weighted toward stay
            decision = rng.choice([-1, 0, 0, 0, 1])
            oc_score = round(rng.uniform(0.3, 0.9), 3)
            writer.writerow({
                "lane": lane,
                "obs_d0": obs_d0,
                "obs_d1": obs_d1,
                "obs_d2": obs_d2,
                "decision": decision,
                "oc_score": oc_score,
            })


def _create_balanced_dataset_csv(path):
    """Create a dataset with all three decision classes represented."""
    rows = [
        {"lane": 0, "obs_d0": 0.2, "obs_d1": 0.8, "obs_d2": 0.9, "decision": 1, "oc_score": 0.8},
        {"lane": 1, "obs_d0": 0.8, "obs_d1": 0.2, "obs_d2": 0.9, "decision": -1, "oc_score": 0.7},
        {"lane": 2, "obs_d0": 0.9, "obs_d1": 0.8, "obs_d2": 0.2, "decision": -1, "oc_score": 0.6},
        {"lane": 1, "obs_d0": 0.8, "obs_d1": 0.8, "obs_d2": 0.8, "decision": 0, "oc_score": 0.9},
        {"lane": 0, "obs_d0": 0.3, "obs_d1": 0.7, "obs_d2": 0.5, "decision": 1, "oc_score": 0.5},
        {"lane": 2, "obs_d0": 0.6, "obs_d1": 0.3, "obs_d2": 0.9, "decision": -1, "oc_score": 0.4},
        {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.5, "obs_d2": 0.5, "decision": 0, "oc_score": 0.8},
        {"lane": 0, "obs_d0": 0.1, "obs_d1": 0.9, "obs_d2": 0.4, "decision": 1, "oc_score": 0.7},
        {"lane": 1, "obs_d0": 0.7, "obs_d1": 0.9, "obs_d2": 0.3, "decision": 0, "oc_score": 0.6},
        {"lane": 2, "obs_d0": 0.4, "obs_d1": 0.6, "obs_d2": 0.1, "decision": -1, "oc_score": 0.5},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lane", "obs_d0", "obs_d1", "obs_d2", "decision", "oc_score"])
        writer.writeheader()
        writer.writerows(rows)


class TestTrainModel:
    def test_train_basic(self, tmp_path):
        """Training on a small CSV returns a model that can predict."""
        csv_path = tmp_path / "dataset.csv"
        _create_balanced_dataset_csv(csv_path)

        model = train_model(csv_path)
        assert model is not None

        # Model should be able to predict on a sample input
        state = {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.5, "obs_d2": 0.5}
        decision = predict(model, state)
        assert decision in (-1, 0, 1)


class TestPredict:
    def test_predict_valid_classes(self, tmp_path):
        """All predictions should be -1, 0, or 1."""
        csv_path = tmp_path / "dataset.csv"
        _create_balanced_dataset_csv(csv_path)
        model = train_model(csv_path)

        # Test various game states
        test_states = [
            {"lane": 0, "obs_d0": 0.2, "obs_d1": 0.8, "obs_d2": 0.9},
            {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.5, "obs_d2": 0.5},
            {"lane": 2, "obs_d0": 0.9, "obs_d1": 0.8, "obs_d2": 0.1},
            {"lane": 1, "nearest_obstacles": [0.3, 0.7, 0.5]},
        ]
        for state in test_states:
            decision = predict(model, state)
            assert decision in (-1, 0, 1), f"Unexpected decision {decision} for state {state}"

    def test_predict_with_nearest_obstacles_key(self, tmp_path):
        """predict() should accept game state with 'nearest_obstacles' key."""
        csv_path = tmp_path / "dataset.csv"
        _create_balanced_dataset_csv(csv_path)
        model = train_model(csv_path)

        state = {"lane": 1, "nearest_obstacles": [0.4, 0.8, 0.6]}
        decision = predict(model, state)
        assert decision in (-1, 0, 1)


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        """Save model, load it, verify predictions match."""
        csv_path = tmp_path / "dataset.csv"
        _create_balanced_dataset_csv(csv_path)
        model = train_model(csv_path)

        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)
        assert model_path.exists()

        loaded_model = load_model(model_path)

        # Predictions from original and loaded model should be identical
        test_states = [
            {"lane": 0, "obs_d0": 0.2, "obs_d1": 0.8, "obs_d2": 0.9},
            {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.5, "obs_d2": 0.5},
            {"lane": 2, "obs_d0": 0.9, "obs_d1": 0.8, "obs_d2": 0.1},
        ]
        for state in test_states:
            orig = predict(model, state)
            loaded = predict(loaded_model, state)
            assert orig == loaded, f"Predictions differ for state {state}: {orig} vs {loaded}"


class TestClassBalance:
    def test_class_balance(self, tmp_path):
        """Model handles imbalanced data (mostly 'stay')."""
        csv_path = tmp_path / "dataset.csv"
        _create_dataset_csv(csv_path, n_rows=30)  # biased toward stay
        model = train_model(csv_path)
        assert model is not None

        # Should still produce valid predictions
        state = {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.5, "obs_d2": 0.5}
        decision = predict(model, state)
        assert decision in (-1, 0, 1)
