from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def save_result_json(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"schema_version": SCHEMA_VERSION, **payload}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def load_result_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "schema_version" not in data:
        data = {"schema_version": 0, **data}
    return data
