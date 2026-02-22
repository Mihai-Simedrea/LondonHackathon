#!/usr/bin/env python3
"""Dataset builder — merges game recording with OC scores, splits into dirty/clean."""

import json
import csv
import numpy as np
from pathlib import Path
import config
from game_engine import LANE_COUNT, HEIGHT


def _compute_nearest_obstacle_distances(obstacles_list):
    """
    Convert obstacle list [[lane, norm_y], ...] to per-lane nearest distances.

    Args:
        obstacles_list: list of [lane, normalized_y] where norm_y is 0.0 (top) to 1.0 (bottom)

    Returns:
        list of 3 floats: distance to nearest obstacle in each lane (1.0 = no obstacle)
        Distance is measured from player position (near bottom) — smaller = closer/more dangerous
    """
    distances = [1.0] * LANE_COUNT
    for obs in obstacles_list:
        lane = int(obs[0])
        norm_y = float(obs[1])
        # Player is at ~bottom of screen (HEIGHT - 120).
        # norm_y = obstacle.y / HEIGHT: 0.0 = top (far), ~0.83 = near player.
        # distance = 1.0 - norm_y: top = far = 1.0, near player = close = ~0.17
        # High distance = safe, low distance = danger.
        if norm_y < 0.9:  # Only count obstacles that haven't passed the player
            dist = 1.0 - norm_y
            if dist < distances[lane]:
                distances[lane] = dist

    return distances


def build_dataset(game_jsonl_path=None, oc_scores_csv_path=None, output_path=None):
    """
    Merge game recording JSONL with OC scores CSV by matching 'sec' field.

    Output CSV columns: lane, obs_d0, obs_d1, obs_d2, decision, oc_score

    Returns: path to full dataset CSV
    """
    game_jsonl_path = game_jsonl_path or config.GAME_JSONL
    oc_scores_csv_path = oc_scores_csv_path or config.OC_SCORES_CSV
    output_path = output_path or (config.DATA_DIR / "dataset_full.csv")

    # Read OC scores
    oc_scores = {}
    with open(oc_scores_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = int(row['sec'])
            oc_scores[sec] = float(row['oc_score'])

    # Read game recording and merge
    rows = []
    with open(game_jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())
            sec = record['sec']

            # Get OC score (default 0.5 if missing)
            oc = oc_scores.get(sec, 0.5)

            # Compute per-lane obstacle distances
            distances = _compute_nearest_obstacle_distances(record.get('obs', []))

            rows.append({
                'lane': record['lane'],
                'obs_d0': round(distances[0], 4),
                'obs_d1': round(distances[1], 4),
                'obs_d2': round(distances[2], 4),
                'decision': record['decision'],
                'oc_score': round(oc, 4)
            })

    # Write dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lane', 'obs_d0', 'obs_d1', 'obs_d2', 'decision', 'oc_score'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Dataset built: {output_path} ({len(rows)} rows)")
    return output_path


def filter_dataset(dataset_path=None, clean_path=None, cutoff=None, oc_cutoff=None):
    """
    Split dataset into dirty (all data) and clean (OC >= cutoff).

    Returns: (dirty_path, clean_path)
    """
    dataset_path = dataset_path or (config.DATA_DIR / "dataset_full.csv")
    # Support both 'cutoff' and 'oc_cutoff' keyword arguments
    if oc_cutoff is not None:
        cutoff = oc_cutoff
    cutoff = cutoff if cutoff is not None else config.OC_CUTOFF

    # Read full dataset
    rows = []
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Split
    clean_rows = [r for r in rows if float(r['oc_score']) >= cutoff]

    # Write dirty (all data)
    dirty_path = config.DATASET_DIRTY
    with open(dirty_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lane', 'obs_d0', 'obs_d1', 'obs_d2', 'decision', 'oc_score'])
        writer.writeheader()
        writer.writerows(rows)

    # Write clean (filtered)
    clean_path = Path(clean_path) if clean_path is not None else config.DATASET_CLEAN
    with open(clean_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['lane', 'obs_d0', 'obs_d1', 'obs_d2', 'decision', 'oc_score'])
        writer.writeheader()
        writer.writerows(clean_rows)

    pct_kept = len(clean_rows) / len(rows) * 100 if rows else 0
    pct_filtered = 100 - pct_kept

    print(f"  Dataset split (cutoff={cutoff}):")
    print(f"    Dirty (all):     {dirty_path} ({len(rows)} rows)")
    print(f"    Clean (filtered): {clean_path} ({len(clean_rows)} rows)")
    print(f"    Filtered out: {len(rows) - len(clean_rows)} rows ({pct_filtered:.1f}%)")

    return dirty_path, clean_path


if __name__ == "__main__":
    full_path = build_dataset()
    filter_dataset(full_path)
