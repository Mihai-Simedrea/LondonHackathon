#!/usr/bin/env python3
"""Generate 4 psychedelic overlay PNGs for the game."""

import numpy as np
from PIL import Image
import colorsys
import os

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
SIZE = 512


def hsv_to_rgb_array(h, s, v):
    """Convert HSV arrays (0-1 range) to RGB (0-255 range)."""
    h = h % 1.0
    c = v * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = v - c

    r = np.zeros_like(h)
    g = np.zeros_like(h)
    b = np.zeros_like(h)

    idx = (h < 1/6)
    r[idx], g[idx], b[idx] = c[idx], x[idx], 0
    idx = (h >= 1/6) & (h < 2/6)
    r[idx], g[idx], b[idx] = x[idx], c[idx], 0
    idx = (h >= 2/6) & (h < 3/6)
    r[idx], g[idx], b[idx] = 0, c[idx], x[idx]
    idx = (h >= 3/6) & (h < 4/6)
    r[idx], g[idx], b[idx] = 0, x[idx], c[idx]
    idx = (h >= 4/6) & (h < 5/6)
    r[idx], g[idx], b[idx] = x[idx], 0, c[idx]
    idx = (h >= 5/6)
    r[idx], g[idx], b[idx] = c[idx], 0, x[idx]

    r = ((r + m) * 255).astype(np.uint8)
    g = ((g + m) * 255).astype(np.uint8)
    b = ((b + m) * 255).astype(np.uint8)
    return r, g, b


def radial_alpha(size):
    """Create alpha mask: opaque center, transparent edges."""
    y, x = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    dist = np.sqrt(x**2 + y**2)
    alpha = np.clip(1.0 - dist, 0, 1)
    # Smooth falloff
    alpha = (alpha ** 0.7 * 200).astype(np.uint8)
    return alpha


def generate_mandelbrot():
    """Colorful Mandelbrot set with psychedelic HSV palette."""
    print("Generating fractal_mandelbrot.png...")
    # Zoom into an interesting region
    cx, cy = -0.745, 0.186
    span = 0.01
    x = np.linspace(cx - span, cx + span, SIZE)
    y = np.linspace(cy - span, cy + span, SIZE)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    escape_time = np.zeros((SIZE, SIZE), dtype=np.float64)
    mask = np.ones((SIZE, SIZE), dtype=bool)
    max_iter = 256

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + C[mask]
        newly_escaped = mask & (np.abs(Z) > 2)
        # Smooth coloring
        escape_time[newly_escaped] = i + 1 - np.log2(np.log2(np.abs(Z[newly_escaped]) + 1e-10))
        mask[newly_escaped] = False

    # Normalize and color
    escape_time[escape_time == 0] = max_iter
    t = escape_time / max_iter

    h = (t * 5.0) % 1.0  # cycle hue multiple times
    s = np.ones_like(t) * 0.9
    v = np.where(escape_time < max_iter, 0.9, 0.0)

    r, g, b = hsv_to_rgb_array(h, s, v)
    alpha = radial_alpha(SIZE)

    img_array = np.stack([r, g, b, alpha], axis=-1)
    img = Image.fromarray(img_array, 'RGBA')
    img.save(os.path.join(ASSETS_DIR, "fractal_mandelbrot.png"))
    print("  Done.")


def generate_kaleidoscope():
    """6-fold symmetric kaleidoscope pattern with rainbow colors."""
    print("Generating kaleidoscope_flower.png...")
    y, x = np.mgrid[-1:1:complex(SIZE), -1:1:complex(SIZE)]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # 6-fold symmetry
    theta6 = np.abs(((theta + np.pi) % (np.pi / 3)) - np.pi / 6)

    # Create petal pattern
    pattern = np.sin(theta6 * 12) * np.cos(r * 8 * np.pi)
    pattern2 = np.cos(theta6 * 6 + r * 4 * np.pi)
    pattern3 = np.sin(r * 12 * np.pi) * np.cos(theta * 6)

    combined = (pattern + pattern2 + pattern3) / 3.0

    # Color mapping
    h = (combined * 0.5 + 0.5 + r * 0.3) % 1.0
    s = np.clip(0.8 + 0.2 * np.sin(r * 10), 0, 1)
    v = np.clip(0.7 + 0.3 * combined, 0, 1)

    rv, gv, bv = hsv_to_rgb_array(h, s, v)
    alpha = radial_alpha(SIZE)

    img_array = np.stack([rv, gv, bv, alpha], axis=-1)
    img = Image.fromarray(img_array, 'RGBA')
    img.save(os.path.join(ASSETS_DIR, "kaleidoscope_flower.png"))
    print("  Done.")


def generate_sacred_geometry():
    """Flower of Life — overlapping circles in hexagonal grid with rainbow gradients."""
    print("Generating sacred_geometry.png...")
    img_array = np.zeros((SIZE, SIZE, 4), dtype=np.uint8)

    y_coords, x_coords = np.mgrid[0:SIZE, 0:SIZE]
    x_norm = (x_coords - SIZE / 2) / (SIZE / 2)
    y_norm = (y_coords - SIZE / 2) / (SIZE / 2)

    # Hexagonal grid of circle centers
    circle_r = 0.18
    spacing = circle_r * 1.0
    centers = []
    for row in range(-5, 6):
        for col in range(-5, 6):
            cx = col * spacing * 2
            cy = row * spacing * np.sqrt(3)
            if col % 2 != 0:
                cy += spacing * np.sqrt(3) / 2
            if cx**2 + cy**2 < 1.5:
                centers.append((cx, cy))

    # Accumulate circle edges
    edge_acc = np.zeros((SIZE, SIZE), dtype=np.float64)
    fill_acc = np.zeros((SIZE, SIZE), dtype=np.float64)

    for i, (cx, cy) in enumerate(centers):
        dist = np.sqrt((x_norm - cx)**2 + (y_norm - cy)**2)
        # Circle edge glow
        edge = np.exp(-((dist - circle_r) ** 2) / 0.0008)
        edge_acc += edge
        # Inside fill (faint)
        fill = np.clip(1.0 - dist / circle_r, 0, 1) * 0.15
        fill_acc += fill

    edge_acc = np.clip(edge_acc, 0, 1)
    fill_acc = np.clip(fill_acc, 0, 0.5)

    combined = np.clip(edge_acc + fill_acc, 0, 1)

    # Rainbow gradient based on angle from center
    angle = np.arctan2(y_norm, x_norm)
    dist_center = np.sqrt(x_norm**2 + y_norm**2)
    h = ((angle / (2 * np.pi)) + 0.5 + dist_center * 0.3) % 1.0
    s = np.ones_like(h) * 0.85
    v = combined

    rv, gv, bv = hsv_to_rgb_array(h, s, v)

    alpha_base = radial_alpha(SIZE).astype(np.float64)
    alpha = (combined * alpha_base).astype(np.uint8)

    img_array = np.stack([rv, gv, bv, alpha], axis=-1)
    img = Image.fromarray(img_array, 'RGBA')
    img.save(os.path.join(ASSETS_DIR, "sacred_geometry.png"))
    print("  Done.")


def generate_spiral_vortex():
    """Logarithmic spiral with color bands radiating outward."""
    print("Generating spiral_vortex.png...")
    y, x = np.mgrid[-1:1:complex(SIZE), -1:1:complex(SIZE)]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)

    # Logarithmic spiral
    log_r = np.log(r + 0.001)
    spiral = np.sin(theta * 3 - log_r * 12)
    spiral2 = np.cos(theta * 5 + log_r * 8)
    spiral3 = np.sin(theta * 2 - log_r * 20) * 0.5

    combined = (spiral + spiral2 + spiral3) / 2.5

    # Color bands
    h = (theta / (2 * np.pi) + 0.5 + r * 2.0 + combined * 0.2) % 1.0
    s = np.clip(0.7 + 0.3 * np.abs(combined), 0, 1)
    v = np.clip(0.5 + 0.5 * combined, 0.1, 1.0)

    rv, gv, bv = hsv_to_rgb_array(h, s, v)
    alpha = radial_alpha(SIZE)

    img_array = np.stack([rv, gv, bv, alpha], axis=-1)
    img = Image.fromarray(img_array, 'RGBA')
    img.save(os.path.join(ASSETS_DIR, "spiral_vortex.png"))
    print("  Done.")


if __name__ == "__main__":
    os.makedirs(ASSETS_DIR, exist_ok=True)
    generate_mandelbrot()
    generate_kaleidoscope()
    generate_sacred_geometry()
    generate_spiral_vortex()
    print(f"\nAll 4 psychedelic assets saved to {ASSETS_DIR}/")
