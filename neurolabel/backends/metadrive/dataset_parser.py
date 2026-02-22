from __future__ import annotations

"""MetaDrive recording -> dataset row conversion helpers."""

from neurolabel.backends.metadrive.env import FEATURE_NAMES


def fieldnames() -> list[str]:
    return list(FEATURE_NAMES) + ["action", "oc_score"]


def record_to_dataset_row(record: dict, oc_score: float) -> dict:
    row = dict(record["features"])
    row["action"] = record["action"]
    row["oc_score"] = round(float(oc_score), 4)
    return row
