from __future__ import annotations

"""Velocity recording -> dataset row conversion helpers."""

from game_engine import HEIGHT, LANE_COUNT

PLAYER_NORM_Y = (HEIGHT - 120) / HEIGHT  # matches game_engine.Player default y


def compute_nearest_obstacle_distances(obstacles_list):
    """Match `game_engine.GameState.get_nearest_obstacles()` semantics."""
    distances = [1.0] * LANE_COUNT
    for obs in obstacles_list:
        lane = int(obs[0])
        norm_y = float(obs[1])
        if norm_y < PLAYER_NORM_Y:
            dist = PLAYER_NORM_Y - norm_y
            if dist < distances[lane]:
                distances[lane] = dist
    return distances


def record_to_dataset_row(record: dict, oc_score: float) -> dict:
    distances = compute_nearest_obstacle_distances(record.get("obs", []))
    return {
        "lane": record["lane"],
        "obs_d0": round(distances[0], 4),
        "obs_d1": round(distances[1], 4),
        "obs_d2": round(distances[2], 4),
        "decision": record["decision"],
        "oc_score": round(float(oc_score), 4),
    }
