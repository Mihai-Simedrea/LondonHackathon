from __future__ import annotations

from neurolabel.brain.base import BrainScorer
from neurolabel.brain.scoring.oc import OcScorer


def load_scorer(_device_mode: str) -> BrainScorer:
    # Current OC scorer auto-detects EEG vs fNIRS from CSV header.
    return OcScorer()
