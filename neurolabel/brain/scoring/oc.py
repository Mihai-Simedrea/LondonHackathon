from __future__ import annotations

from typing import Any

from neurolabel.config.schema import Settings


class OcScorer:
    """Wrapper around the legacy `oc_scorer` module."""

    def compute_scores(
        self,
        settings: Settings,
        *,
        trim_before: float | None = None,
        include_timestamp_in_csv: bool = True,
    ) -> list[dict[str, Any]]:
        from oc_scorer import compute_oc_scores

        return compute_oc_scores(
            str(settings.brain_csv),
            output_path=str(settings.paths.oc_scores_csv),
            trim_before=trim_before,
            include_timestamp_in_csv=include_timestamp_in_csv,
        )
