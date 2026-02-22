from __future__ import annotations

from neurolabel.backends.base import BackendAdapter
from neurolabel.backends.velocity.adapter import VelocityBackend
from neurolabel.backends.metadrive.adapter import MetaDriveBackend

_BACKENDS = {
    "velocity": VelocityBackend,
    "metadrive": MetaDriveBackend,
}


def load_backend(name: str) -> BackendAdapter:
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {name!r}. Expected one of {sorted(_BACKENDS)}") from exc


def available_backends() -> list[str]:
    return sorted(_BACKENDS)
