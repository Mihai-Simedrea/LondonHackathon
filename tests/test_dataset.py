"""Tests for dataset.py — dataset building and filtering.

dataset.py may not exist yet; these tests define the expected interface.
Expected functions:
  - build_dataset(game_jsonl, oc_scores_csv, output_csv) -> writes CSV
    with columns: lane, obs_d0, obs_d1, obs_d2, decision, oc_score
  - filter_dataset(input_csv, output_csv, oc_cutoff) -> writes filtered CSV
    keeping only rows where oc_score >= oc_cutoff
"""

import sys
import csv
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

EXPECTED_COLUMNS = ["lane", "obs_d0", "obs_d1", "obs_d2", "decision", "oc_score"]


def _create_mock_dataset_csv(path, rows=None):
    """Create a mock dataset CSV with known data."""
    if rows is None:
        rows = [
            {"lane": 1, "obs_d0": 0.5, "obs_d1": 0.8, "obs_d2": 0.3, "decision": 0, "oc_score": 0.8},
            {"lane": 0, "obs_d0": 0.2, "obs_d1": 0.9, "obs_d2": 0.7, "decision": 1, "oc_score": 0.7},
            {"lane": 2, "obs_d0": 0.6, "obs_d1": 0.4, "obs_d2": 0.1, "decision": -1, "oc_score": 0.3},
            {"lane": 1, "obs_d0": 0.9, "obs_d1": 0.5, "obs_d2": 0.6, "decision": 0, "oc_score": 0.5},
            {"lane": 0, "obs_d0": 0.1, "obs_d1": 0.7, "obs_d2": 0.8, "decision": 1, "oc_score": 0.9},
            {"lane": 2, "obs_d0": 0.4, "obs_d1": 0.3, "obs_d2": 0.2, "decision": -1, "oc_score": 0.2},
        ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPECTED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _try_import_dataset():
    """Try to import dataset module, skip test if not available."""
    try:
        import dataset
        return dataset
    except ImportError:
        pytest.skip("dataset.py not yet implemented")


class TestBuildDataset:
    def test_build_dataset(self, tmp_path):
        """Verify output CSV has correct columns."""
        dataset = _try_import_dataset()
        if not hasattr(dataset, "build_dataset"):
            pytest.skip("build_dataset not yet implemented")

        # Create minimal mock inputs
        game_jsonl = tmp_path / "game.jsonl"
        oc_csv = tmp_path / "oc_scores.csv"
        output_csv = tmp_path / "output.csv"

        # Write mock game JSONL
        import json
        with open(game_jsonl, "w") as f:
            for sec in range(10):
                record = {
                    "sec": sec,
                    "lane": 1,
                    "obs": [[1, 0.5]],
                    "decision": 0,
                    "score": sec * 60,
                    "alive": True,
                }
                f.write(json.dumps(record) + "\n")

        # Write mock OC scores CSV
        with open(oc_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["sec", "fatigue_idx", "engagement_idx", "oc_score"])
            writer.writeheader()
            for sec in range(10):
                writer.writerow({"sec": sec, "fatigue_idx": 0.5, "engagement_idx": 0.6, "oc_score": 0.7})

        dataset.build_dataset(game_jsonl, oc_csv, output_csv)

        assert output_csv.exists(), "build_dataset should create output CSV"
        with open(output_csv) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "Output CSV should have rows"
        for col in EXPECTED_COLUMNS:
            assert col in rows[0], f"Missing column: {col}"


class TestFilterDataset:
    def test_filter_dataset(self, tmp_path):
        """Verify clean dataset has fewer rows than dirty dataset."""
        dataset = _try_import_dataset()
        if not hasattr(dataset, "filter_dataset"):
            pytest.skip("filter_dataset not yet implemented")

        dirty_csv = tmp_path / "dirty.csv"
        clean_csv = tmp_path / "clean.csv"
        rows = _create_mock_dataset_csv(dirty_csv)

        # Cutoff at 0.6: should keep rows with oc_score >= 0.6
        dataset.filter_dataset(dirty_csv, clean_csv, oc_cutoff=0.6)

        assert clean_csv.exists()
        with open(clean_csv) as f:
            clean_rows = list(csv.DictReader(f))

        expected_count = sum(1 for r in rows if float(r["oc_score"]) >= 0.6)
        assert len(clean_rows) == expected_count
        assert len(clean_rows) < len(rows), "Filtered dataset should be smaller"


class TestObstacleDistances:
    def test_obstacle_distances(self, tmp_path):
        """Verify distance values are in valid range [0, 1]."""
        dirty_csv = tmp_path / "dirty.csv"
        _create_mock_dataset_csv(dirty_csv)

        with open(dirty_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ["obs_d0", "obs_d1", "obs_d2"]:
                    val = float(row[col])
                    assert 0.0 <= val <= 1.0, f"{col} = {val} out of [0, 1]"

    def test_dataset_columns_types(self, tmp_path):
        """Verify dataset column values parse correctly."""
        dirty_csv = tmp_path / "dirty.csv"
        _create_mock_dataset_csv(dirty_csv)

        with open(dirty_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                lane = int(row["lane"])
                assert lane in (0, 1, 2)
                decision = int(row["decision"])
                assert decision in (-1, 0, 1)
                oc = float(row["oc_score"])
                assert 0.0 <= oc <= 1.0
