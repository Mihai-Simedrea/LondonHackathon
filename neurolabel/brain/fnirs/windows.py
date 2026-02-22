from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

DEFAULT_WINDOW_SEC = 4.0
DEFAULT_STRIDE_SEC = 1.0
EXPECTED_FNIRS_COLUMNS = [
    'timestamp',
    'ir_l', 'red_l', 'amb_l',
    'ir_r', 'red_r', 'amb_r',
    'ir_p', 'red_p', 'amb_p',
    'temp',
]


@dataclass(frozen=True)
class FnirsRawSeries:
    timestamps: np.ndarray
    red_l_corr: np.ndarray
    ir_l_corr: np.ndarray
    amb_l: np.ndarray
    red_r_corr: np.ndarray
    ir_r_corr: np.ndarray
    amb_r: np.ndarray
    red_p: np.ndarray
    ir_p: np.ndarray
    amb_p: np.ndarray
    temp: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.timestamps.size)

    @property
    def duration_sec(self) -> float:
        if self.n_samples < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def sample_rate_est_hz(self) -> float:
        if self.n_samples < 2:
            return 0.0
        duration = self.duration_sec
        if duration <= 0:
            return 0.0
        return float((self.n_samples - 1) / duration)


@dataclass(frozen=True)
class FnirsWindow:
    sec: float
    start_idx: int
    end_idx: int
    timestamps: np.ndarray
    red_l_corr: np.ndarray
    ir_l_corr: np.ndarray
    amb_l: np.ndarray
    red_r_corr: np.ndarray
    ir_r_corr: np.ndarray
    amb_r: np.ndarray
    red_p: np.ndarray
    ir_p: np.ndarray
    amb_p: np.ndarray
    temp: np.ndarray

    @property
    def n_samples(self) -> int:
        return int(self.timestamps.size)

    @property
    def mid_timestamp(self) -> float:
        if self.timestamps.size == 0:
            return 0.0
        mid = self.timestamps.size // 2
        return float(self.timestamps[mid])


class FnirsCsvError(ValueError):
    pass


def _coerce_rows(rows: Iterable[dict], *, trim_before: float | None = None) -> list[dict[str, float]]:
    coerced: list[dict[str, float]] = []
    for row in rows:
        try:
            ts = float(row['timestamp'])
            if trim_before is not None and ts < trim_before:
                continue
            ir_l = float(row['ir_l'])
            red_l = float(row['red_l'])
            amb_l = float(row['amb_l'])
            ir_r = float(row['ir_r'])
            red_r = float(row['red_r'])
            amb_r = float(row['amb_r'])
            ir_p = float(row.get('ir_p', 0.0))
            red_p = float(row.get('red_p', 0.0))
            amb_p = float(row.get('amb_p', 0.0))
            temp = float(row.get('temp', 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise FnirsCsvError(f'Invalid fNIRS CSV row: {row}') from exc

        coerced.append({
            'timestamp': ts,
            'red_l_corr': red_l - amb_l,
            'ir_l_corr': ir_l - amb_l,
            'amb_l': amb_l,
            'red_r_corr': red_r - amb_r,
            'ir_r_corr': ir_r - amb_r,
            'amb_r': amb_r,
            'red_p_corr': red_p - amb_p,
            'ir_p_corr': ir_p - amb_p,
            'amb_p': amb_p,
            'temp': temp,
        })
    if not coerced:
        raise FnirsCsvError('No valid fNIRS samples found in CSV')
    return coerced


def series_from_rows(rows: Iterable[dict], *, trim_before: float | None = None) -> FnirsRawSeries:
    coerced = _coerce_rows(rows, trim_before=trim_before)
    timestamps = np.array([r['timestamp'] for r in coerced], dtype=np.float64)
    order = np.argsort(timestamps)

    def arr(key: str) -> np.ndarray:
        return np.array([coerced[int(i)][key] for i in order], dtype=np.float64)

    return FnirsRawSeries(
        timestamps=timestamps[order],
        red_l_corr=arr('red_l_corr'),
        ir_l_corr=arr('ir_l_corr'),
        amb_l=arr('amb_l'),
        red_r_corr=arr('red_r_corr'),
        ir_r_corr=arr('ir_r_corr'),
        amb_r=arr('amb_r'),
        red_p=arr('red_p_corr'),
        ir_p=arr('ir_p_corr'),
        amb_p=arr('amb_p'),
        temp=arr('temp'),
    )


def _dict_reader_from_text(csv_text: str) -> csv.DictReader:
    return csv.DictReader(io.StringIO(csv_text))


def load_fnirs_csv_text(csv_text: str, *, trim_before: float | None = None) -> FnirsRawSeries:
    reader = _dict_reader_from_text(csv_text)
    header = list(reader.fieldnames or [])
    if 'ir_l' not in header or 'red_l' not in header:
        raise FnirsCsvError('CSV does not look like Mendi fNIRS data (missing ir_l/red_l columns)')
    return series_from_rows(reader, trim_before=trim_before)


def load_fnirs_csv(path: str | Path, *, trim_before: float | None = None) -> FnirsRawSeries:
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        header = list(reader.fieldnames or [])
        if 'ir_l' not in header or 'red_l' not in header:
            raise FnirsCsvError(f'CSV at {path} does not look like Mendi fNIRS data')
        return series_from_rows(reader, trim_before=trim_before)


def iter_windows(
    series: FnirsRawSeries,
    *,
    window_sec: float = DEFAULT_WINDOW_SEC,
    stride_sec: float = DEFAULT_STRIDE_SEC,
    sample_rate_hz: float | int | None = None,
) -> Iterator[FnirsWindow]:
    if series.n_samples < 2:
        return
    sample_rate = int(round(float(sample_rate_hz or series.sample_rate_est_hz or 11)))
    sample_rate = max(1, sample_rate)
    window_samples = max(1, int(round(sample_rate * float(window_sec))))
    stride_samples = max(1, int(round(sample_rate * float(stride_sec))))
    if series.n_samples < window_samples:
        return

    sec = 0.0
    start = 0
    while start + window_samples <= series.n_samples:
        end = start + window_samples
        yield FnirsWindow(
            sec=sec,
            start_idx=start,
            end_idx=end,
            timestamps=series.timestamps[start:end],
            red_l_corr=series.red_l_corr[start:end],
            ir_l_corr=series.ir_l_corr[start:end],
            amb_l=series.amb_l[start:end],
            red_r_corr=series.red_r_corr[start:end],
            ir_r_corr=series.ir_r_corr[start:end],
            amb_r=series.amb_r[start:end],
            red_p=series.red_p[start:end],
            ir_p=series.ir_p[start:end],
            amb_p=series.amb_p[start:end],
            temp=series.temp[start:end],
        )
        start += stride_samples
        sec += float(stride_sec)
