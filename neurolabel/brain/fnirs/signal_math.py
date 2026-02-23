"""Public fNIRS optical-to-hemoglobin math helpers.

This module intentionally contains only generic MBLL conversion utilities and
constants. Device-specific BLE decoding/transport code is not included here.
"""

from __future__ import annotations

import numpy as np

# Extinction coefficients in mM^-1 cm^-1
#                          660 nm (Red)  940 nm (IR)
EPSILON_HBO = np.array([0.15, 0.30], dtype=np.float64)
EPSILON_HBR = np.array([0.35, 0.12], dtype=np.float64)

# Rows = wavelengths (Red, IR), cols = (HbO, HbR)
EPSILON_MATRIX = np.array(
    [
        [EPSILON_HBO[0], EPSILON_HBR[0]],
        [EPSILON_HBO[1], EPSILON_HBR[1]],
    ],
    dtype=np.float64,
)

# Differential pathlength factor * source-detector separation (forehead proxy)
DPF = 6.0
SD_DISTANCE_CM = 3.0
PATH_LENGTH = DPF * SD_DISTANCE_CM


def intensity_to_od(intensity: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Convert raw intensity to delta optical density (OD)."""
    safe_intensity = np.clip(intensity, 1.0, None)
    safe_baseline = np.clip(baseline, 1.0, None)
    return -np.log10(safe_intensity / safe_baseline)


def od_to_concentrations(delta_od: np.ndarray) -> np.ndarray:
    """Solve the 2x2 MBLL system for [delta_HbO, delta_HbR] in uM."""
    e = EPSILON_MATRIX * PATH_LENGTH
    e_inv = np.linalg.pinv(e)
    concentrations = (e_inv @ delta_od.T).T
    return concentrations * 1000.0  # mM -> uM

