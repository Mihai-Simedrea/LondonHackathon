#!/usr/bin/env python3
"""Dataset builder — merges game recording with OC scores, splits into dirty/clean."""

import json
import csv
from bisect import bisect_left
from pathlib import Path
import config
from game_engine import LANE_COUNT, HEIGHT


VELOCITY_FIELDNAMES = ["lane", "obs_d0", "obs_d1", "obs_d2", "decision", "oc_score"]


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


def _match_oc_score(record, oc_ts_keys, oc_ts_values, oc_by_sec):
    """Match a game record to its OC score by timestamp or sec field."""
    oc = None
    matched = True
    if oc_ts_keys and 't' in record:
        game_ts = float(record['t'])
        idx = bisect_left(oc_ts_keys, game_ts)
        best_dist = float('inf')
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(oc_ts_keys):
                d = abs(oc_ts_keys[candidate] - game_ts)
                if d < best_dist:
                    best_dist = d
                    oc = oc_ts_values[candidate]
        if best_dist > 2.5:
            oc = None
            matched = False
    else:
        sec = record.get('sec', record.get('s', 0))
        oc = oc_by_sec.get(sec)
        if oc is None:
            matched = False

    if oc is None:
        oc = 0.5
    return oc, matched


def _load_oc_lookup(oc_scores_csv_path):
    """Load OC scores indexed by both integer second and timestamp."""
    oc_ts_pairs = []
    oc_by_sec = {}
    with open(oc_scores_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oc_score = float(row["oc_score"])
            oc_by_sec[int(row["sec"])] = oc_score
            if row.get("timestamp"):
                oc_ts_pairs.append((float(row["timestamp"]), oc_score))

    oc_ts_pairs.sort(key=lambda p: p[0])
    oc_ts_keys = [ts for ts, _ in oc_ts_pairs]
    oc_ts_values = [score for _, score in oc_ts_pairs]
    return oc_ts_keys, oc_ts_values, oc_by_sec


def _detect_game_format(game_jsonl_path):
    """Return True if the recording is in MetaDrive format."""
    with open(game_jsonl_path, "r") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Empty game recording file: {game_jsonl_path}")
    first = json.loads(first_line)
    return "features" in first


def _write_csv_rows(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_dataset(game_jsonl_path=None, oc_scores_csv_path=None, output_path=None):
    """
    Merge game recording JSONL with OC scores CSV.

    Supports both Velocity (lane/obs/decision) and MetaDrive (features/action) formats.

    Returns: path to full dataset CSV
    """
    game_jsonl_path = game_jsonl_path or config.GAME_JSONL
    oc_scores_csv_path = oc_scores_csv_path or config.OC_SCORES_CSV
    output_path = output_path or (config.DATA_DIR / "dataset_full.csv")

    # Read OC scores — index by wall-clock timestamp for accurate matching
    oc_ts_keys, oc_ts_values, oc_by_sec = _load_oc_lookup(oc_scores_csv_path)

    # Detect format from first JSONL record
    is_metadrive = _detect_game_format(game_jsonl_path)

    rows = []
    unmatched = 0
    skipped = 0
    with open(game_jsonl_path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue
            oc, matched = _match_oc_score(record, oc_ts_keys, oc_ts_values, oc_by_sec)
            if not matched:
                unmatched += 1

            if is_metadrive:
                row = dict(record["features"])
                row["action"] = record["action"]
                row["oc_score"] = round(oc, 4)
            else:
                distances = _compute_nearest_obstacle_distances(record.get('obs', []))
                row = {
                    'lane': record['lane'],
                    'obs_d0': round(distances[0], 4),
                    'obs_d1': round(distances[1], 4),
                    'obs_d2': round(distances[2], 4),
                    'decision': record['decision'],
                    'oc_score': round(oc, 4),
                }
            rows.append(row)

    if is_metadrive:
        from metadrive_wrapper import FEATURE_NAMES
        fieldnames = FEATURE_NAMES + ["action", "oc_score"]
    else:
        fieldnames = VELOCITY_FIELDNAMES

    output_path = _write_csv_rows(output_path, fieldnames, rows)

    if skipped > 0:
        print(f"  ⚠ Skipped {skipped} corrupted JSONL lines")
    if unmatched > 0:
        print(f"  ⚠ {unmatched}/{len(rows)} game records had no matching OC score (assigned 0.5)")
    print(f"  Dataset built: {output_path} ({len(rows)} rows)")
    return output_path


def filter_dataset(dataset_path=None, clean_path=None, cutoff=None, oc_cutoff=None, dirty_path=None):
    """
    Split dataset into dirty (all data) and clean (OC >= cutoff).

    Returns: (dirty_path, clean_path)
    """
    dataset_path = dataset_path or (config.DATA_DIR / "dataset_full.csv")
    # Support both 'cutoff' and 'oc_cutoff' keyword arguments
    if oc_cutoff is not None:
        cutoff = oc_cutoff
    cutoff = cutoff if cutoff is not None else config.OC_CUTOFF

    # Read full dataset (rows + fieldnames in one pass)
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Split
    clean_rows = [r for r in rows if float(r['oc_score']) >= cutoff]

    # Write dirty (all data)
    dirty_path = Path(dirty_path) if dirty_path is not None else config.DATASET_DIRTY
    dirty_path = _write_csv_rows(dirty_path, fieldnames, rows)

    # Write clean (filtered)
    clean_path = Path(clean_path) if clean_path is not None else config.DATASET_CLEAN
    clean_path = _write_csv_rows(clean_path, fieldnames, clean_rows)

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
