"""
oc_scorer.py — Operator Confidence scorer for NeuroLabel.

Computes OC scores from raw EEG or fNIRS data.

EEG mode:  sliding-window Welch PSD -> band powers -> fatigue/engagement indices
           -> z-score against baseline -> sigmoid fusion -> OC score in [0, 1].
fNIRS mode: sliding-window MBLL -> dHbO/dHbR -> engagement/fatigue indices
            -> z-score against baseline -> sigmoid fusion -> OC score in [0, 1].
"""

import csv
from pathlib import Path

import numpy as np
from scipy.signal import welch

from neurolabel.brain.fnirs.signal_math import intensity_to_od, od_to_concentrations

DEFAULT_SAMPLE_RATE = 250
DEFAULT_FNIRS_SAMPLE_RATE = 11
WINDOW_SEC = 4
STRIDE_SEC = 1

# Backward-compatible constants used by tests / older scripts.
SAMPLE_RATE = DEFAULT_SAMPLE_RATE
WINDOW_SAMPLES = SAMPLE_RATE * WINDOW_SEC

# Indices into the 20-channel array (0-based, before timestamp offset)
# Fp1=0, Fp2=1, Fpz=2, Fz=9
FRONTAL_CHANNELS = [0, 1, 2, 9]

THETA_RANGE = (4, 8)
ALPHA_RANGE = (8, 13)
BETA_RANGE = (13, 30)

BASELINE_SEC = 120
MIN_BASELINE_SEC = 10

FNIRS_BASELINE_SEC = 30

EPSILON = 1e-10


def compute_oc_scores(csv_path, output_path=None, trim_before=None, include_timestamp_in_csv=False):
    """
    Auto-detect EEG vs fNIRS CSV and compute OC scores.

    Inspects the CSV header: if it contains 'ir_l' it is treated as fNIRS
    data, otherwise as EEG data.

    Args:
        csv_path: Path to EEG or fNIRS CSV
        output_path: Optional path to save OC scores CSV
        trim_before: Optional wall-clock timestamp — discard samples before
                     this time (e.g. game start)
        include_timestamp_in_csv: If True, include wall-clock timestamp column in
                                  saved CSV output for precise matching.

    Returns:
        List of dicts with sec, timestamp, fatigue_idx, engagement_idx,
        oc_score.
    """
    csv_path = Path(csv_path)
    with open(csv_path, "r", newline="") as f:
        first_line = f.readline().lower()

    if "ir_l" in first_line:
        return compute_oc_scores_fnirs(csv_path, output_path=output_path,
                                       trim_before=trim_before,
                                       include_timestamp_in_csv=include_timestamp_in_csv)
    return compute_oc_scores_eeg(csv_path, output_path=output_path,
                                 trim_before=trim_before,
                                 include_timestamp_in_csv=include_timestamp_in_csv)


def compute_oc_scores_eeg(eeg_csv_path, output_path=None, trim_before=None, include_timestamp_in_csv=False):
    """
    Compute OC scores from raw EEG CSV.

    Args:
        eeg_csv_path: Path to EEG CSV (timestamp + 20 channels)
        output_path: Optional path to save OC scores CSV
        trim_before: Optional wall-clock timestamp — discard all EEG samples
                     before this time (e.g. game start time, to skip idle periods)

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
            ts = float(row[0])
            if trim_before is not None and ts < trim_before:
                continue
            timestamps.append(ts)
            frontal = [float(row[ch + 1]) for ch in FRONTAL_CHANNELS]
            rows.append(frontal)

    timestamps = np.array(timestamps)
    data = np.array(rows, dtype=np.float64)  # shape (n_samples, 4)
    n_samples = data.shape[0]

    if trim_before is not None:
        trimmed = n_samples
        print(f"  Trimmed EEG to game start: {n_samples} samples kept")

    # --- Auto-detect sample rate from timestamps ---
    if n_samples > 1:
        total_duration = timestamps[-1] - timestamps[0]
        sample_rate = (n_samples - 1) / total_duration if total_duration > 0 else DEFAULT_SAMPLE_RATE
        # Round to nearest plausible rate
        sample_rate = round(sample_rate)
    else:
        sample_rate = DEFAULT_SAMPLE_RATE

    if sample_rate != DEFAULT_SAMPLE_RATE:
        print(f"  Auto-detected sample rate: {sample_rate} Hz (config default: {DEFAULT_SAMPLE_RATE})")

    window_samples = sample_rate * WINDOW_SEC
    stride_samples = sample_rate * STRIDE_SEC

    # Need at least one full window
    if n_samples < window_samples:
        print(f"  Not enough samples for one window ({n_samples} < {window_samples})")
        return []

    # --- Sliding window: compute per-window band powers ---
    fatigue_indices = []
    engagement_indices = []
    window_secs = []
    window_timestamps = []

    nperseg = min(512, window_samples // 2)  # adapt Welch segment size to window

    sec = 0
    start = 0
    while start + window_samples <= n_samples:
        window = data[start : start + window_samples, :]

        # Wall-clock timestamp at midpoint of this window
        mid = start + window_samples // 2
        window_ts = float(timestamps[min(mid, n_samples - 1)])

        # PSD per frontal channel via Welch (use actual sample rate)
        psds = []
        for ch in range(window.shape[1]):
            freqs, psd = welch(window[:, ch], fs=sample_rate, nperseg=nperseg)
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

        start += stride_samples
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
        fieldnames = ["sec", "fatigue_idx", "engagement_idx", "oc_score"]
        if include_timestamp_in_csv:
            fieldnames.insert(1, "timestamp")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                if include_timestamp_in_csv:
                    writer.writerow({name: row[name] for name in fieldnames})
                else:
                    writer.writerow(
                        {
                            "sec": row["sec"],
                            "fatigue_idx": row["fatigue_idx"],
                            "engagement_idx": row["engagement_idx"],
                            "oc_score": row["oc_score"],
                        }
                    )

    return results


# ---------------------------------------------------------------------------
# fNIRS OC scoring
# ---------------------------------------------------------------------------

def compute_oc_scores_fnirs(fnirs_csv_path, output_path=None, trim_before=None, include_timestamp_in_csv=False):
    """
    Compute OC scores from raw fNIRS CSV (private local headset integration).

    Uses the Modified Beer-Lambert Law (MBLL) to convert raw optical
    intensities into HbO/HbR concentration changes per sliding window.
    Prefrontal HbO increase maps to engagement; HbR increase maps to fatigue.

    Args:
        fnirs_csv_path: Path to fNIRS CSV with columns:
                        timestamp, ir_l, red_l, amb_l, ir_r, red_r, amb_r,
                        ir_p, red_p, amb_p, temp
        output_path: Optional path to save OC scores CSV
        trim_before: Optional wall-clock timestamp — discard samples before
                     this time

    Returns:
        List of dicts: [{"sec": int, "timestamp": float, "fatigue_idx": float,
                         "engagement_idx": float, "oc_score": float}, ...]
    """
    fnirs_csv_path = Path(fnirs_csv_path)

    # --- Read CSV ---
    timestamps = []
    ir_l_raw, red_l_raw, amb_l_raw = [], [], []
    ir_r_raw, red_r_raw, amb_r_raw = [], [], []

    with open(fnirs_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["timestamp"])
            if trim_before is not None and ts < trim_before:
                continue
            timestamps.append(ts)
            ir_l_raw.append(float(row["ir_l"]))
            red_l_raw.append(float(row["red_l"]))
            amb_l_raw.append(float(row["amb_l"]))
            ir_r_raw.append(float(row["ir_r"]))
            red_r_raw.append(float(row["red_r"]))
            amb_r_raw.append(float(row["amb_r"]))

    timestamps = np.array(timestamps)
    n_samples = len(timestamps)

    if trim_before is not None:
        print(f"  Trimmed fNIRS to game start: {n_samples} samples kept")

    if n_samples < 2:
        print("  Not enough fNIRS samples")
        return []

    # --- Ambient-corrected intensities per channel ---
    # Each array shape: (n_samples, 2) for [Red, IR]
    intensity_l = np.column_stack([
        np.array(red_l_raw) - np.array(amb_l_raw),
        np.array(ir_l_raw) - np.array(amb_l_raw),
    ])
    intensity_r = np.column_stack([
        np.array(red_r_raw) - np.array(amb_r_raw),
        np.array(ir_r_raw) - np.array(amb_r_raw),
    ])

    # --- Auto-detect sample rate from timestamps ---
    total_duration = timestamps[-1] - timestamps[0]
    if total_duration > 0:
        sample_rate = (n_samples - 1) / total_duration
        sample_rate = round(sample_rate)
    else:
        sample_rate = DEFAULT_FNIRS_SAMPLE_RATE

    if sample_rate != DEFAULT_FNIRS_SAMPLE_RATE:
        print(f"  Auto-detected fNIRS sample rate: {sample_rate} Hz "
              f"(config default: {DEFAULT_FNIRS_SAMPLE_RATE})")

    window_samples = sample_rate * WINDOW_SEC
    stride_samples = sample_rate * STRIDE_SEC

    if n_samples < window_samples:
        print(f"  Not enough fNIRS samples for one window "
              f"({n_samples} < {window_samples})")
        return []

    # --- Compute per-channel baseline (first FNIRS_BASELINE_SEC seconds) ---
    baseline_samples = min(sample_rate * FNIRS_BASELINE_SEC, n_samples)
    baseline_l = intensity_l[:baseline_samples].mean(axis=0)  # shape (2,)
    baseline_r = intensity_r[:baseline_samples].mean(axis=0)

    # --- Sliding window: MBLL per window -> dHbO / dHbR ---
    engagement_indices = []
    fatigue_indices = []
    window_secs = []
    window_timestamps = []

    sec = 0
    start = 0
    while start + window_samples <= n_samples:
        win_l = intensity_l[start : start + window_samples]
        win_r = intensity_r[start : start + window_samples]

        # Delta optical density for this window (generic MBLL helpers)
        dod_l = intensity_to_od(win_l, baseline_l)   # shape (win, 2)
        dod_r = intensity_to_od(win_r, baseline_r)

        # MBLL inversion -> [dHbO, dHbR] in uM (generic MBLL helper)
        conc_l = od_to_concentrations(dod_l)  # shape (win, 2)
        conc_r = od_to_concentrations(dod_r)

        # Average HbO and HbR across left + right channels
        hbo = (conc_l[:, 0] + conc_r[:, 0]) / 2.0
        hbr = (conc_l[:, 1] + conc_r[:, 1]) / 2.0

        # Window-level indices: mean concentration change
        engagement_idx = float(np.mean(hbo))   # prefrontal oxygenation
        fatigue_idx = float(np.mean(hbr))      # deoxygenation

        engagement_indices.append(engagement_idx)
        fatigue_indices.append(fatigue_idx)

        # Wall-clock timestamp at midpoint
        mid = start + window_samples // 2
        window_timestamps.append(float(timestamps[min(mid, n_samples - 1)]))
        window_secs.append(sec)

        start += stride_samples
        sec += STRIDE_SEC

    engagement_indices = np.array(engagement_indices)
    fatigue_indices = np.array(fatigue_indices)
    n_windows = len(engagement_indices)

    if n_windows == 0:
        return []

    # --- Z-score against baseline windows (first FNIRS_BASELINE_SEC) ---
    baseline_win_count = min(FNIRS_BASELINE_SEC, n_windows)
    if baseline_win_count < MIN_BASELINE_SEC:
        baseline_win_count = n_windows

    eng_base = engagement_indices[:baseline_win_count]
    fat_base = fatigue_indices[:baseline_win_count]

    eng_mean, eng_std = eng_base.mean(), eng_base.std() + EPSILON
    fat_mean, fat_std = fat_base.mean(), fat_base.std() + EPSILON

    z_engagement = (engagement_indices - eng_mean) / eng_std
    z_fatigue = (fatigue_indices - fat_mean) / fat_std

    # --- Sigmoid fusion (same formula as EEG) ---
    raw_oc = 1.0 / (1.0 + np.exp(-(0.6 * z_engagement - 0.4 * z_fatigue + 0.8)))
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
        scores = [r["oc_score"] for r in results]
        above_cutoff = sum(1 for s in scores if s >= 0.6)
        print(f"  OC distribution (fNIRS): min={min(scores):.3f}, "
              f"max={max(scores):.3f}, mean={sum(scores)/len(scores):.3f}, "
              f"above 0.6: {above_cutoff}/{len(scores)} "
              f"({above_cutoff/len(scores)*100:.1f}%)")

    # --- Optionally save to CSV ---
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sec", "fatigue_idx", "engagement_idx", "oc_score"]
        if include_timestamp_in_csv:
            fieldnames.insert(1, "timestamp")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                if include_timestamp_in_csv:
                    writer.writerow({name: row[name] for name in fieldnames})
                else:
                    writer.writerow(
                        {
                            "sec": row["sec"],
                            "fatigue_idx": row["fatigue_idx"],
                            "engagement_idx": row["engagement_idx"],
                            "oc_score": row["oc_score"],
                        }
                    )

    return results


def _band_power(freqs, psd, freq_range):
    """Extract power in a frequency band using trapezoidal integration."""
    _trapz = getattr(np, 'trapezoid', None) or np.trapz
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    return _trapz(psd[mask], freqs[mask])
