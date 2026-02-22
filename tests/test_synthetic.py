"""Tests for synthetic_data.py — synthetic EEG and game data generation."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synthetic_data import generate_synthetic_eeg, generate_synthetic_game


class TestGenerateEEG:
    def test_generate_eeg_shape(self):
        """EEG data has correct shape: (samples x 21) = timestamp + 20 channels."""
        duration = 5  # 5 seconds
        sample_rate = 250
        data = generate_synthetic_eeg(duration, sample_rate=sample_rate)

        expected_samples = duration * sample_rate
        assert data.shape == (expected_samples, 21), (
            f"Expected shape ({expected_samples}, 21), got {data.shape}"
        )

    def test_generate_eeg_timestamp_column(self):
        """First column should be timestamps (monotonically increasing)."""
        data = generate_synthetic_eeg(3, sample_rate=250)
        timestamps = data[:, 0]
        # Timestamps should be increasing
        assert np.all(np.diff(timestamps) > 0), "Timestamps should be monotonically increasing"

    def test_generate_eeg_channel_range(self):
        """Channel values should be in a reasonable EEG microvolt range."""
        data = generate_synthetic_eeg(5, sample_rate=250)
        channels = data[:, 1:]  # skip timestamp
        # Synthetic EEG should be roughly in [-100, 100] uV range
        assert np.abs(channels).max() < 500, (
            f"Channel values seem too large: max={np.abs(channels).max()}"
        )


class TestGenerateGame:
    def test_generate_game_length(self):
        """Game recording has expected number of seconds."""
        duration = 10
        records = generate_synthetic_game(duration)
        # Should have approximately duration records (may be fewer if game ends early
        # and restarts, but always <= duration)
        assert len(records) == duration, (
            f"Expected {duration} records, got {len(records)}"
        )

    def test_generate_game_record_keys(self):
        """Each game record has expected keys."""
        records = generate_synthetic_game(5)
        required_keys = {"t", "sec", "lane", "obs", "decision", "score", "alive"}
        for record in records:
            assert required_keys.issubset(record.keys()), (
                f"Missing keys: {required_keys - record.keys()}"
            )

    def test_generate_game_valid_decisions(self):
        """All decisions should be -1, 0, or 1."""
        records = generate_synthetic_game(10)
        for record in records:
            assert record["decision"] in (-1, 0, 1), (
                f"Invalid decision: {record['decision']}"
            )

    def test_generate_game_valid_lanes(self):
        """All lanes should be 0, 1, or 2."""
        records = generate_synthetic_game(10)
        for record in records:
            assert record["lane"] in (0, 1, 2), (
                f"Invalid lane: {record['lane']}"
            )
