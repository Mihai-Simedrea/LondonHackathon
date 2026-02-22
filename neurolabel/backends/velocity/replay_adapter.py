from __future__ import annotations

from typing import Any


def supports_replay() -> bool:
    return True


def to_velocity_viewer_payload(results: dict[str, Any]) -> dict[str, Any]:
    """Current velocity viewer already consumes the legacy run format."""
    return results
