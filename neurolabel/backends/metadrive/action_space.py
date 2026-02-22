from __future__ import annotations

"""Shared MetaDrive discrete action-space helpers."""

import numpy as np

import config


def steering_values() -> np.ndarray:
    return np.linspace(-1, 1, config.METADRIVE_STEERING_DIM)


def throttle_values() -> np.ndarray:
    return np.linspace(-1, 1, config.METADRIVE_THROTTLE_DIM)


def noop_action() -> int:
    """Middle steering + middle throttle discrete action."""
    return (
        (config.METADRIVE_THROTTLE_DIM // 2) * config.METADRIVE_STEERING_DIM
        + (config.METADRIVE_STEERING_DIM // 2)
    )


def continuous_to_discrete(steer: float, throttle: float) -> int:
    """Quantize continuous [-1,1] steering/throttle to a discrete action index."""
    s_vals = steering_values()
    t_vals = throttle_values()
    s_idx = int(np.argmin(np.abs(s_vals - steer)))
    t_idx = int(np.argmin(np.abs(t_vals - throttle)))
    # MetaDrive decodes: steer = action % steering_dim, throttle = action // steering_dim
    return t_idx * config.METADRIVE_STEERING_DIM + s_idx

