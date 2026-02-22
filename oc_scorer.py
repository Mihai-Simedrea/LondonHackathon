"""
oc_scorer.py — Operator Confidence scorer for NeuroLabel.

Computes OC scores from raw EEG data using frontal-channel PSD analysis.
Algorithm: sliding-window Welch PSD -> band powers -> fatigue/engagement indices
-> z-score against baseline -> sigmoid fusion -> OC score in [0, 1].
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import welch

SAMPLE_RATE = 250
WINDOW_SEC = 4
STRIDE_SEC = 1
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC  # 1000
STRIDE_SAMPLES = SAMPLE_RATE * STRIDE_SEC  # 250

# Indices into the 20-channel array (0-based, before timestamp offset)
# Fp1=0, Fp2=1, Fpz=2, Fz=9
FRONTAL_CHANNELS = [0, 1, 2, 9]

THETA_RANGE = (4, 8)
ALPHA_RANGE = (8, 13)
BETA_RANGE = (13, 30)

BASELINE_SEC = 120
MIN_BASELINE_SEC = 10

EPSILON = 1e-10


def compute_oc_scores(eeg_csv_path, output_path=None):
    """
    Compute OC scores from raw EEG CSV.

    Args:
        eeg_csv_path: Path to EEG CSV (timestamp + 20 channels)
        output_path: Optional path to save OC scores CSV

    Returns:
        List of dicts: [{"sec": int, "fatigue_idx": float,
                         "engagement_idx": float, "oc_score": float}, ...]
    """
    eeg_csv_path = Path(eeg_csv_path)

    # --- Read CSV, extract timestamps + frontal channels ---
    timestamps = []
    rows = []
    with open(eeg_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # skip header
        for row in reader:
            # columns: timestamp, ch0, ch1, ..., ch19
            timestamps.append(float(row[0]))
            frontal = [float(row[ch + 1]) for ch in FRONTAL_CHANNELS]
            rows.append(frontal)

    timestamps = np.array(timestamps)
    data = np.array(rows, dtype=np.float64)  # shape (n_samples, 4)
    n_samples = data.shape[0]

    # Need at least one full window (4 seconds)
    if n_samples < WINDOW_SAMPLES:
        return []

    # --- Sliding window: compute per-window band powers ---
    fatigue_indices = []
    engagement_indices = []
    window_secs = []
    window_timestamps = []

    sec = 0
    start = 0
    while start + WINDOW_SAMPLES <= n_samples:
        window = data[start : start + WINDOW_SAMPLES, :]  # (1000, 4)

        # Wall-clock timestamp at midpoint of this window
        mid = start + WINDOW_SAMPLES // 2
        window_ts = float(timestamps[min(mid, n_samples - 1)])

        # PSD per frontal channel via Welch
        psds = []
        for ch in range(window.shape[1]):
            freqs, psd = welch(window[:, ch], fs=SAMPLE_RATE, nperseg=512)
            psds.append(psd)

        # Average PSD across 4 frontal channels
        avg_psd = np.mean(psds, axis=0)

        # Band powers
        theta = _band_power(freqs, avg_psd, THETA_RANGE)
        alpha = _band_power(freqs, avg_psd, ALPHA_RANGE)
        beta = _band_power(freqs, avg_psd, BETA_RANGE)

        # Indices (epsilon guards division by zero)
        fatigue_idx = theta / (alpha + EPSILON)
        engagement_idx = beta / (alpha + theta + EPSILON)

        fatigue_indices.append(fatigue_idx)
        engagement_indices.append(engagement_idx)
        window_secs.append(sec)
        window_timestamps.append(window_ts)

        start += STRIDE_SAMPLES
        sec += STRIDE_SEC

    fatigue_indices = np.array(fatigue_indices)
    engagement_indices = np.array(engagement_indices)

    # --- Baseline: first 120 seconds of indices ---
    # Each index entry corresponds to 1-second stride, so baseline covers
    # up to BASELINE_SEC entries.  If fewer than MIN_BASELINE_SEC entries
    # exist, use all available data as baseline.
    n_windows = len(fatigue_indices)
    baseline_count = min(BASELINE_SEC, n_windows)
    if baseline_count < MIN_BASELINE_SEC:
        baseline_count = n_windows

    fat_base = fatigue_indices[:baseline_count]
    eng_base = engagement_indices[:baseline_count]

    fat_mean, fat_std = fat_base.mean(), fat_base.std() + EPSILON
    eng_mean, eng_std = eng_base.mean(), eng_base.std() + EPSILON

    # --- Z-score both indices ---
    z_fatigue = (fatigue_indices - fat_mean) / fat_std
    z_engagement = (engagement_indices - eng_mean) / eng_std

    # --- Sigmoid fusion ---
    # Bias of +0.8 shifts baseline-focused state from sigmoid(0)=0.5 to
    # sigmoid(0.8)≈0.69, so a normally-focused operator scores above the
    # OC_CUTOFF (0.6).  Fatigued periods still score near 0.
    raw_oc = 1.0 / (1.0 + np.exp(-(0.6 * z_engagement - 0.4 * z_fatigue + 0.8)))

    # Clamp to [0, 1] (sigmoid already in (0,1), but be safe)
    oc_scores = np.clip(raw_oc, 0.0, 1.0)

    # --- Build result list ---
    results = []
    for i in range(n_windows):
        results.append(
            {
                "sec": window_secs[i],
                "timestamp": round(window_timestamps[i], 2),
                "fatigue_idx": round(float(fatigue_indices[i]), 4),
                "engagement_idx": round(float(engagement_indices[i]), 4),
                "oc_score": round(float(oc_scores[i]), 4),
            }
        )

    # --- Summary ---
    if results:
        scores = [r['oc_score'] for r in results]
        above_cutoff = sum(1 for s in scores if s >= 0.6)
        print(f"  OC distribution: min={min(scores):.3f}, max={max(scores):.3f}, "
              f"mean={sum(scores)/len(scores):.3f}, above 0.6: {above_cutoff}/{len(scores)} "
              f"({above_cutoff/len(scores)*100:.1f}%)")

    # --- Optionally save to CSV ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["sec", "timestamp", "fatigue_idx", "engagement_idx", "oc_score"]
            )
            writer.writeheader()
            writer.writerows(results)

    return results


def _band_power(freqs, psd, freq_range):
    """Extract power in a frequency band using trapezoidal integration."""
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return _trapz(psd[mask], freqs[mask])
