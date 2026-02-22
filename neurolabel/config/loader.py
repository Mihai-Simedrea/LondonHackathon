from __future__ import annotations

from .defaults import from_legacy_config
from .schema import Settings


def load_settings(*, backend: str | None = None, device_mode: str | None = None) -> Settings:
    """Load runtime settings, defaulting to values from the legacy config module."""
    settings = from_legacy_config()
    if backend is not None:
        settings = settings.with_overrides(backend=backend)
    if device_mode is not None:
        settings = settings.with_overrides(device_mode=device_mode)
    settings.paths.ensure_dirs()
    return settings
