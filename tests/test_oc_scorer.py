"""Tests for oc_scorer.py — OC score computation from EEG data."""

import sys
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oc_scorer import compute_oc_scores, _band_power, SAMPLE_RATE, WINDOW_SAMPLES


def _write_eeg_csv(path, data):
    """Helper: write EEG data array to CSV with header."""
    channel_names = [
        "Fp1", "Fp2", "Fpz", "Cp1", "-", "-", "T7", "-", "O1", "Fz",
        "O2", "Cp2", "T8", "-", "Oz", "P3", "P4", "P7", "Cz", "P8"
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + channel_names)
        for row in data:
            writer.writerow(row.tolist())


def _generate_sine_eeg(freq, duration_sec, amplitude=20.0):
    """Generate EEG data where all channels contain a pure sine wave at given freq."""
    n_samples = duration_sec * SAMPLE_RATE
    t = np.arange(n_samples) / SAMPLE_RATE
    sine = amplitude * np.sin(2 * np.pi * freq * t)
    data = np.zeros((n_samples, 21))
    data[:, 0] = np.arange(n_samples) / SAMPLE_RATE  # timestamp
    for ch in range(1, 21):
        data[:, ch] = sine + np.random.randn(n_samples) * 0.1  # tiny noise
    return data


class TestBandPower:
    def test_band_power_known_sine(self):
        """A pure 10Hz sine wave should have dominant alpha-band power."""
        from scipy.signal import welch as scipy_welch

        n_samples = SAMPLE_RATE * 4  # 4 seconds
        t = np.arange(n_samples) / SAMPLE_RATE
        signal = 20.0 * np.sin(2 * np.pi * 10 * t)  # 10 Hz = alpha band

        freqs, psd = scipy_welch(signal, fs=SAMPLE_RATE, nperseg=512)

        alpha_power = _band_power(freqs, psd, (8, 13))
        theta_power = _band_power(freqs, psd, (4, 8))
        beta_power = _band_power(freqs, psd, (13, 30))

        # Alpha should dominate
        assert alpha_power > theta_power * 5, "Alpha should be much larger than theta for 10Hz signal"
        assert alpha_power > beta_power * 5, "Alpha should be much larger than beta for 10Hz signal"


class TestComputeOCScores:
    def test_compute_oc_scores_basic(self, tmp_path):
        """Generate simple EEG data; verify OC scores are returned."""
        duration = 10  # 10 seconds
        eeg_data = _generate_sine_eeg(freq=10, duration_sec=duration)
        eeg_path = tmp_path / "test_eeg.csv"
        _write_eeg_csv(eeg_path, eeg_data)

        scores = compute_oc_scores(eeg_path)
        assert isinstance(scores, list)
        assert len(scores) > 0
        # Each score entry has required keys
        for entry in scores:
            assert "sec" in entry
            assert "fatigue_idx" in entry
            assert "engagement_idx" in entry
            assert "oc_score" in entry

    def test_oc_score_range(self, tmp_path):
        """All OC scores should be between 0 and 1."""
        duration = 10
        eeg_data = _generate_sine_eeg(freq=15, duration_sec=duration)
        eeg_path = tmp_path / "test_eeg.csv"
        _write_eeg_csv(eeg_path, eeg_data)

        scores = compute_oc_scores(eeg_path)
        for entry in scores:
            assert 0.0 <= entry["oc_score"] <= 1.0, f"OC score {entry['oc_score']} out of [0,1]"

    def test_empty_data(self, tmp_path):
        """Very short EEG data (less than one window) should return empty list."""
        # Only 1 second of data — less than WINDOW_SAMPLES (4 sec)
        short_data = _generate_sine_eeg(freq=10, duration_sec=1)
        eeg_path = tmp_path / "short_eeg.csv"
        _write_eeg_csv(eeg_path, short_data)

        scores = compute_oc_scores(eeg_path)
        assert scores == []

    def test_output_saved_to_file(self, tmp_path):
        """When output_path is given, OC scores CSV is written."""
        duration = 10
        eeg_data = _generate_sine_eeg(freq=10, duration_sec=duration)
        eeg_path = tmp_path / "test_eeg.csv"
        out_path = tmp_path / "oc_output.csv"
        _write_eeg_csv(eeg_path, eeg_data)

        scores = compute_oc_scores(eeg_path, output_path=out_path)
        assert out_path.exists()
        # Verify CSV has correct header
        with open(out_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == len(scores)
        assert set(rows[0].keys()) == {"sec", "fatigue_idx", "engagement_idx", "oc_score"}
