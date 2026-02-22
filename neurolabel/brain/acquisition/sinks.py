from __future__ import annotations

"""Shared recording buffers and CSV sinks for brain-acquisition flows."""

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import config


EEG_CHANNEL_NAMES = list(config.CHANNEL_NAMES)
FNIRS_CHANNEL_NAMES = list(config.FNIRS_CHANNEL_NAMES)


@dataclass
class RecordBuffer:
    """In-memory sample buffer with basic recording stats."""

    samples: list[list[float]] = field(default_factory=list)
    packet_count: int = 0
    start_time: datetime | None = None

    def append(self, sample: list[float]) -> None:
        if self.start_time is None:
            self.start_time = datetime.now()
        self.samples.append(sample)
        self.packet_count += 1

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time is None:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    def is_empty(self) -> bool:
        return not self.samples


@dataclass(frozen=True)
class SaveConfig:
    save_folder: Path
    fixed_filename: str | None = None
    device_mode: str = "eeg"

    def ensure_dir(self) -> None:
        self.save_folder.mkdir(parents=True, exist_ok=True)

    def resolve_filepath(self) -> Path:
        self.ensure_dir()
        if self.fixed_filename:
            return self.save_folder / self.fixed_filename

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if self.device_mode == "fnirs":
            return self.save_folder / f"fnirs-recording-{timestamp}.csv"
        return self.save_folder / f"eeg-recording-{timestamp}.csv"

    def header(self) -> list[str]:
        return FNIRS_CHANNEL_NAMES if self.device_mode == "fnirs" else EEG_CHANNEL_NAMES


def save_buffer_to_csv(buffer: RecordBuffer, save: SaveConfig) -> Path | None:
    """Persist buffered samples to CSV. Returns output path or None if empty."""
    if buffer.is_empty():
        print("❌ No data to save")
        return None

    path = save.resolve_filepath()
    header = save.header()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + header)
        writer.writerows(buffer.samples)

    print(f"\n✅ Saved: {path}")
    print(f"📈 Total samples: {len(buffer.samples):,}")
    print(f"⏱️  Duration: {buffer.elapsed_seconds:.1f}s")
    print(f"📊 Channels: {len(header)}")
    return path
