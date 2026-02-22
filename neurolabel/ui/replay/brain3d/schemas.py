from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BrainMesh:
    source: str
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray     # (M, 3)
    units: str = 'arb'


@dataclass(frozen=True)
class PfcMapping:
    roi_mask: np.ndarray      # (N,) bool
    left_weights: np.ndarray  # (N,) float
    right_weights: np.ndarray # (N,) float
    anchors: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class HeatmapFrame:
    sec: float
    timestamp: float
    left_raw_score: float
    right_raw_score: float
    pulse_quality: float
    n_samples: int
    left_red_z: float = 0.0
    left_ir_z: float = 0.0
    left_amb_z: float = 0.0
    right_red_z: float = 0.0
    right_ir_z: float = 0.0
    right_amb_z: float = 0.0


@dataclass(frozen=True)
class HeatmapBundle:
    schema_version: int
    source: dict[str, Any]
    mesh: dict[str, Any]
    mapping: dict[str, Any]
    windows: list[dict[str, Any]]
    viewer_defaults: dict[str, Any] = field(default_factory=dict)


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_list(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_list(v) for v in value]
    return value


def mesh_to_json(mesh: BrainMesh) -> dict[str, Any]:
    return {
        'source': mesh.source,
        'units': mesh.units,
        'vertices': mesh.vertices.tolist(),
        'faces': mesh.faces.tolist(),
    }


def mapping_to_json(mapping: PfcMapping) -> dict[str, Any]:
    return {
        'roi_mask': mapping.roi_mask.astype(int).tolist(),
        'left_weights': mapping.left_weights.tolist(),
        'right_weights': mapping.right_weights.tolist(),
        'anchors': _to_list(mapping.anchors),
    }


def frame_to_json(frame: HeatmapFrame) -> dict[str, Any]:
    return {
        'sec': float(frame.sec),
        'timestamp': float(frame.timestamp),
        'left_raw_score': float(frame.left_raw_score),
        'right_raw_score': float(frame.right_raw_score),
        'pulse_quality': float(frame.pulse_quality),
        'n_samples': int(frame.n_samples),
        'left_red_z': float(frame.left_red_z),
        'left_ir_z': float(frame.left_ir_z),
        'left_amb_z': float(frame.left_amb_z),
        'right_red_z': float(frame.right_red_z),
        'right_ir_z': float(frame.right_ir_z),
        'right_amb_z': float(frame.right_amb_z),
    }
