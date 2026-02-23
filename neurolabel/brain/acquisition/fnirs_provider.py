"""Optional local fNIRS provider loader for private device integrations.

Public repo code can depend on this module without shipping any vendor SDKs.
A local private provider module can be installed on a
developer machine and exposed via ``NEUROLABEL_FNIRS_PROVIDER_MODULE``.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Protocol, TypeVar


class FnirsProviderUnavailableError(RuntimeError):
    """Raised when no local private fNIRS provider module is configured."""


class FnirsClientProtocol(Protocol):
    """Minimal async client contract used by the app."""

    async def __aenter__(self) -> "FnirsClientProtocol": ...
    async def __aexit__(self, *exc: Any) -> None: ...
    def on(self, event: str, callback: Any) -> None: ...
    async def start_streaming(self) -> None: ...
    async def stop_streaming(self) -> None: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_streaming(self) -> bool: ...


T = TypeVar("T")


_DEFAULT_MODULE_CANDIDATES = (
    "private_fnirs_provider",
    "private_fnirs.provider",
    "local_fnirs_provider",
)
_DEFAULT_CLASS_CANDIDATES = (
    "FnirsClient",
    "PrivateFnirsClient",
    "Client",
)


def _iter_module_candidates() -> list[str]:
    env_module = (os.getenv("NEUROLABEL_FNIRS_PROVIDER_MODULE") or "").strip()
    if env_module:
        return [env_module]
    return list(_DEFAULT_MODULE_CANDIDATES)


def _iter_class_candidates() -> list[str]:
    env_class = (os.getenv("NEUROLABEL_FNIRS_PROVIDER_CLASS") or "").strip()
    if env_class:
        return [env_class]
    return list(_DEFAULT_CLASS_CANDIDATES)


def _load_attr(module: Any, names: list[str]) -> Any | None:
    for name in names:
        value = getattr(module, name, None)
        if value is not None:
            return value
    return None


def get_private_fnirs_client_class() -> type[FnirsClientProtocol]:
    """Return the configured local private fNIRS client class.

    The public repository intentionally does not ship a headset SDK. To enable
    a real headset locally, create a private module exposing a client class
    matching ``FnirsClientProtocol`` and set ``NEUROLABEL_FNIRS_PROVIDER_MODULE``.
    """

    errors: list[str] = []
    for module_name in _iter_module_candidates():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
            continue

        attr = _load_attr(module, list(_iter_class_candidates()))
        if attr is None:
            errors.append(f"{module_name}: no client class ({', '.join(_iter_class_candidates())})")
            continue
        return attr

    detail = "; ".join(errors) if errors else "no provider modules configured"
    raise FnirsProviderUnavailableError(
        "Real fNIRS device integration is not included in the public repo. "
        "Install a private local provider and set NEUROLABEL_FNIRS_PROVIDER_MODULE. "
        f"Lookup details: {detail}"
    )
