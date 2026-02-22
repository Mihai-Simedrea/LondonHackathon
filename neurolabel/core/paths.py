from __future__ import annotations

import json
from pathlib import Path


def read_game_start_timestamp(game_jsonl: Path) -> float | None:
    """Return the first record's wall-clock timestamp (`t`) if present."""
    if not game_jsonl.exists():
        return None
    try:
        with open(game_jsonl, "r") as f:
            first_line = f.readline().strip()
        if not first_line:
            return None
        return json.loads(first_line).get("t")
    except Exception:
        return None
