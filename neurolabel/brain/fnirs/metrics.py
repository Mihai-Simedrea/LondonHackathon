from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .windows import FnirsRawSeries, FnirsWindow, iter_windows

DEFAULT_BASELINE_SEC = 30
EPS = 1e-6


@dataclass(frozen=True)
class FnirsBaseline:
    red_l_mean: float
    red_l_std: float
    ir_l_mean: float
    ir_l_std: float
    amb_l_mean: float
    amb_l_std: float
    red_r_mean: float
    red_r_std: float
    ir_r_mean: float
    ir_r_std: float
    amb_r_mean: float
    amb_r_std: float


@dataclass(frozen=True)
class FnirsRawWindowMetrics:
    sec: float
    timestamp: float
    left_raw_score: float
    right_raw_score: float
    n_samples: int
    sample_rate_est_hz: float
    pulse_quality: float
    left_red_z: float
    left_ir_z: float
    left_amb_z: float
    right_red_z: float
    right_ir_z: float
    right_amb_z: float


@dataclass(frozen=True)
class FnirsMetricsSequence:
    baseline: FnirsBaseline
    windows: list[FnirsRawWindowMetrics]
    sample_rate_est_hz: float


def _safe_std(arr: np.ndarray) -> float:
    std = float(np.std(arr))
    return std if std > EPS else 1.0


def compute_baseline(series: FnirsRawSeries, *, baseline_sec: int = DEFAULT_BASELINE_SEC) -> FnirsBaseline:
    sample_rate = max(1, int(round(series.sample_rate_est_hz or 11)))
    n = min(series.n_samples, sample_rate * int(baseline_sec))
    if n < 2:
        n = series.n_samples
    sl = slice(0, max(1, n))
    return FnirsBaseline(
        red_l_mean=float(np.mean(series.red_l_corr[sl])),
        red_l_std=_safe_std(series.red_l_corr[sl]),
        ir_l_mean=float(np.mean(series.ir_l_corr[sl])),
        ir_l_std=_safe_std(series.ir_l_corr[sl]),
        amb_l_mean=float(np.mean(series.amb_l[sl])),
        amb_l_std=_safe_std(series.amb_l[sl]),
        red_r_mean=float(np.mean(series.red_r_corr[sl])),
        red_r_std=_safe_std(series.red_r_corr[sl]),
        ir_r_mean=float(np.mean(series.ir_r_corr[sl])),
        ir_r_std=_safe_std(series.ir_r_corr[sl]),
        amb_r_mean=float(np.mean(series.amb_r[sl])),
        amb_r_std=_safe_std(series.amb_r[sl]),
    )


def _z_mean(values: np.ndarray, mean: float, std: float) -> float:
    return float(np.mean((values - mean) / max(std, EPS)))


def compute_window_metrics(window: FnirsWindow, baseline: FnirsBaseline, *, sample_rate_est_hz: float) -> FnirsRawWindowMetrics:
    left_red_z = _z_mean(window.red_l_corr, baseline.red_l_mean, baseline.red_l_std)
    left_ir_z = _z_mean(window.ir_l_corr, baseline.ir_l_mean, baseline.ir_l_std)
    left_amb_z = _z_mean(window.amb_l, baseline.amb_l_mean, baseline.amb_l_std)
    right_red_z = _z_mean(window.red_r_corr, baseline.red_r_mean, baseline.red_r_std)
    right_ir_z = _z_mean(window.ir_r_corr, baseline.ir_r_mean, baseline.ir_r_std)
    right_amb_z = _z_mean(window.amb_r, baseline.amb_r_mean, baseline.amb_r_std)

    # Use all 3 raw optical sources per side:
    # red + IR carry signal amplitude (ambient-corrected), ambient contributes a noise/context term.
    left_score = (0.44 * left_red_z) + (0.44 * left_ir_z) - (0.12 * left_amb_z)
    right_score = (0.44 * right_red_z) + (0.44 * right_ir_z) - (0.12 * right_amb_z)

    pulse_quality = float(np.std(window.ir_p) + np.std(window.red_p) + 0.15 * np.std(window.amb_p))
    return FnirsRawWindowMetrics(
        sec=float(window.sec),
        timestamp=float(window.mid_timestamp),
        left_raw_score=float(np.clip(left_score, -4.0, 4.0)),
        right_raw_score=float(np.clip(right_score, -4.0, 4.0)),
        n_samples=int(window.n_samples),
        sample_rate_est_hz=float(sample_rate_est_hz),
        pulse_quality=pulse_quality,
        left_red_z=float(np.clip(left_red_z, -6.0, 6.0)),
        left_ir_z=float(np.clip(left_ir_z, -6.0, 6.0)),
        left_amb_z=float(np.clip(left_amb_z, -6.0, 6.0)),
        right_red_z=float(np.clip(right_red_z, -6.0, 6.0)),
        right_ir_z=float(np.clip(right_ir_z, -6.0, 6.0)),
        right_amb_z=float(np.clip(right_amb_z, -6.0, 6.0)),
    )


def compute_window_metrics_sequence(
    series: FnirsRawSeries,
    *,
    baseline_sec: int = DEFAULT_BASELINE_SEC,
    window_sec: float = 4,
    stride_sec: float = 1,
) -> FnirsMetricsSequence:
    baseline = compute_baseline(series, baseline_sec=baseline_sec)
    sr = float(series.sample_rate_est_hz or 11.0)
    windows = [
        compute_window_metrics(w, baseline, sample_rate_est_hz=sr)
        for w in iter_windows(series, window_sec=window_sec, stride_sec=stride_sec)
    ]
    return FnirsMetricsSequence(baseline=baseline, windows=windows, sample_rate_est_hz=sr)
