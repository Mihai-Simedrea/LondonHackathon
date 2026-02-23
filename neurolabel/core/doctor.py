from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from neurolabel.config.schema import Settings
from neurolabel.backends.registry import available_backends


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_doctor(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    checks.append(Check("backend", settings.backend in available_backends(), f"backend={settings.backend}"))
    checks.append(Check("device_mode", settings.device_mode in {"eeg", "fnirs"}, f"device={settings.device_mode}"))
    checks.append(Check("pygame", _has_module("pygame"), "required for Velocity UI / visualizer"))
    checks.append(Check("numpy", _has_module("numpy"), "required"))
    checks.append(Check("scipy", _has_module("scipy"), "required for OC scoring"))
    checks.append(Check("sklearn", _has_module("sklearn"), "required for sklearn models"))
    checks.append(Check("joblib", _has_module("joblib"), "required for model persistence"))
    checks.append(Check("bleak", _has_module("bleak"), "required for EEG BLE / local fNIRS integration"))
    checks.append(Check("metadrive", _has_module("metadrive"), "required for MetaDrive backend"))
    checks.append(Check("torch", _has_module("torch"), "optional for torch experiments"))
    checks.append(Check("catboost", _has_module("catboost"), "optional for compare experiment"))
    checks.append(Check("xgboost", _has_module("xgboost"), "optional for compare experiment"))

    paths = settings.paths
    checks.append(Check("data_dir", paths.data_dir.exists(), str(paths.data_dir)))
    checks.append(Check("models_dir", paths.models_dir.exists(), str(paths.models_dir)))
    checks.append(Check("results_dir", paths.results_dir.exists(), str(paths.results_dir)))
    return checks
