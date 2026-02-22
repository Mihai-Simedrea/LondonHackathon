"""Compatibility wrapper for the canonical MetaDrive environment module."""

from neurolabel.backends.metadrive.env import (
    EGO_DIM,
    FEATURE_NAMES,
    LIDAR_RAYS,
    LIDAR_START,
    NAVI_DIM,
    NUM_ACTIONS,
    NUM_FEATURES,
    SECTORS,
    MetaDriveGame,
    _build_env_config,
    extract_features,
)

__all__ = [
    "EGO_DIM",
    "NAVI_DIM",
    "LIDAR_START",
    "LIDAR_RAYS",
    "SECTORS",
    "FEATURE_NAMES",
    "NUM_FEATURES",
    "NUM_ACTIONS",
    "_build_env_config",
    "extract_features",
    "MetaDriveGame",
]
