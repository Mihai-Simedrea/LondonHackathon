from __future__ import annotations

from pathlib import Path
from typing import Any

from neurolabel.core.results import load_result_json, save_result_json


def load_summary(path: Path) -> dict[str, Any]:
    return load_result_json(path)


def save_summary(path: Path, payload: dict[str, Any]) -> Path:
    return save_result_json(path, payload)
