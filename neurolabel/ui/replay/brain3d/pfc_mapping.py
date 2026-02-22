from __future__ import annotations

import numpy as np

from .schemas import BrainMesh, PfcMapping


def _gaussian_weights(vertices: np.ndarray, anchor: np.ndarray, sigma: float) -> np.ndarray:
    d2 = np.sum((vertices - anchor[None, :]) ** 2, axis=1)
    return np.exp(-d2 / max(1e-6, 2 * sigma * sigma))


def build_pfc_proxy_mapping(mesh: BrainMesh) -> PfcMapping:
    v = mesh.vertices.astype(np.float32)
    mins = v.min(axis=0)
    maxs = v.max(axis=0)
    center = (mins + maxs) / 2.0
    span = np.maximum(maxs - mins, 1e-6)

    x = (v[:, 0] - center[0]) / span[0]  # left/right
    y = (v[:, 1] - center[1]) / span[1]  # anterior/posterior (front positive assumed)
    z = (v[:, 2] - center[2]) / span[2]  # superior/inferior

    # Frontal / prefrontal proxy ROI: front shell, upper half.
    roi_mask = (y > 0.02) & (z > -0.15)

    # Anchor points near left/right frontal areas on the shell.
    left_anchor = np.array([mins[0] + 0.30 * span[0], mins[1] + 0.80 * span[1], mins[2] + 0.62 * span[2]], dtype=np.float32)
    right_anchor = np.array([mins[0] + 0.70 * span[0], mins[1] + 0.80 * span[1], mins[2] + 0.62 * span[2]], dtype=np.float32)

    sigma = float(np.linalg.norm(span) * 0.12)
    left_w = _gaussian_weights(v, left_anchor, sigma)
    right_w = _gaussian_weights(v, right_anchor, sigma)

    # Suppress the opposite hemisphere contribution and keep only frontal ROI.
    left_w *= (x < 0.10).astype(np.float32)
    right_w *= (x > -0.10).astype(np.float32)
    left_w *= roi_mask.astype(np.float32)
    right_w *= roi_mask.astype(np.float32)

    # Spatial taper so heat concentrates around anchors instead of painting the whole front uniformly.
    frontal_taper = np.clip((y - 0.02) / 0.42, 0.0, 1.0).astype(np.float32)
    superior_taper = np.clip((z + 0.10) / 0.55, 0.0, 1.0).astype(np.float32)
    raw_total = left_w + right_w
    max_total = float(np.percentile(raw_total[roi_mask], 95)) if np.any(roi_mask) else 1.0
    amp = np.clip(raw_total / max(max_total, 1e-6), 0.0, 1.0).astype(np.float32)
    amp *= frontal_taper * superior_taper
    amp = amp ** 0.85

    total = raw_total.copy()
    nz = total > 1e-8
    left_w[nz] = left_w[nz] / total[nz]
    right_w[nz] = right_w[nz] / total[nz]
    left_w *= amp
    right_w *= amp

    # Outside ROI, keep weights zero.
    left_w[~roi_mask] = 0.0
    right_w[~roi_mask] = 0.0

    anchors = {
        'left_pfc': {
            'label': 'Mendi Left',
            'xyz': [float(x) for x in left_anchor.tolist()],
        },
        'right_pfc': {
            'label': 'Mendi Right',
            'xyz': [float(x) for x in right_anchor.tolist()],
        },
    }

    return PfcMapping(
        roi_mask=roi_mask,
        left_weights=left_w.astype(np.float32),
        right_weights=right_w.astype(np.float32),
        anchors=anchors,
    )


def project_scores_to_vertices(mapping: PfcMapping, left_score: float, right_score: float) -> np.ndarray:
    values = mapping.left_weights * float(left_score) + mapping.right_weights * float(right_score)
    values = values.astype(np.float32)
    values[~mapping.roi_mask] = 0.0
    return values
