#!/usr/bin/env python3
"""Dataset builder — merges game recording with OC scores, splits into dirty/clean."""

import json
import csv
from bisect import bisect_left
import numpy as np
from pathlib import Path
import config
from game_engine import LANE_COUNT, HEIGHT


def _compute_nearest_obstacle_distances(obstacles_list):
    """
    Convert obstacle list [[lane, norm_y], ...] to per-lane nearest distances.

    Must match game_engine.GameState.get_nearest_obstacles() exactly so that
    features at training time equal features at simulation/inference time.

    Args:
        obstacles_list: list of [lane, normalized_y] where norm_y = obstacle.y / HEIGHT

    Returns:
        list of 3 floats: distance to nearest obstacle in each lane (1.0 = no obstacle)
    """
    PLAYER_NORM_Y = (HEIGHT - 120) / HEIGHT  # ~0.8333
    distances = [1.0] * LANE_COUNT
    for obs in obstacles_list:
        lane = int(obs[0])
        norm_y = float(obs[1])
        if norm_y < PLAYER_NORM_Y:  # only obstacles ahead of player
            dist = (PLAYER_NORM_Y - norm_y)
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

    # Read OC scores — index by wall-clock timestamp for accurate matching
    oc_ts_keys = []    # sorted timestamp list for bisect
    oc_ts_values = []  # corresponding oc_scores
    oc_by_sec = {}     # fallback: sec -> oc_score (for data without timestamps)
    with open(oc_scores_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            oc_by_sec[int(row['sec'])] = float(row['oc_score'])
            if 'timestamp' in row and row['timestamp']:
                oc_ts_keys.append(float(row['timestamp']))
                oc_ts_values.append(float(row['oc_score']))

    # Read game recording and merge
    rows = []
    unmatched = 0
    with open(game_jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())

            # Match by wall-clock timestamp if available, else fall back to sec
            oc = None
            if oc_ts_keys and 't' in record:
                game_ts = float(record['t'])
                # bisect for true nearest neighbor
                idx = bisect_left(oc_ts_keys, game_ts)
                best_dist = float('inf')
                # Check the two candidates adjacent to the insertion point
                for candidate in (idx - 1, idx):
                    if 0 <= candidate < len(oc_ts_keys):
                        d = abs(oc_ts_keys[candidate] - game_ts)
                        if d < best_dist:
                            best_dist = d
                            oc = oc_ts_values[candidate]
                if best_dist > 2.5:
                    oc = None
                    unmatched += 1
            else:
                # Fallback for data without wall-clock timestamps
                sec = record.get('sec', record.get('s', 0))
                oc = oc_by_sec.get(sec)
                if oc is None:
                    unmatched += 1

            if oc is None:
                oc = 0.5  # default for unmatched

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

    if unmatched > 0:
        print(f"  ⚠ {unmatched}/{len(rows)} game records had no matching OC score (assigned 0.5)")
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
