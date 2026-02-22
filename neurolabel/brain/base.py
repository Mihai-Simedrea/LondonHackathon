from __future__ import annotations

from typing import Protocol, Any

from neurolabel.config.schema import Settings


class BrainScorer(Protocol):
    def compute_scores(
        self,
        settings: Settings,
        *,
        trim_before: float | None = None,
        include_timestamp_in_csv: bool = True,
    ) -> list[dict[str, Any]]: ...
