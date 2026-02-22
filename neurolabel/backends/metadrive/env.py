"""MetaDrive wrapper - adapts MetaDrive env to NeuroLabel pipeline interface."""

from __future__ import annotations

import numpy as np
from metadrive import MetaDriveEnv
import config

# Observation layout for default MetaDrive (259-dim):
#   [0]  dist_left_boundary (normalized)
#   [1]  dist_right_boundary (normalized)
#   [2]  heading_diff (angle between vehicle heading and lane)
#   [3]  speed (normalized: speed_km_h / max_speed_km_h)
#   [4]  steering (normalized)
#   [5]  last_throttle_action (normalized)
#   [6]  last_steering_action (normalized)
#   [7]  yaw_rate (normalized)
#   [8]  lateral_offset (normalized, 0.5 = centered)
#   [9:19]  navigation info (2 checkpoints x 5 values)
#   [19:259] lidar cloud points (240 rays, clockwise from front)

EGO_DIM = 9
NAVI_DIM = 10
LIDAR_START = EGO_DIM + NAVI_DIM  # 19
LIDAR_RAYS = 240

# Lidar sector binning: 240 rays covering 360 degrees (clockwise from front)
# Each ray covers 1.5 degrees. We bin into 8 sectors of 30 rays each.
SECTORS = {
    "front":       (0, 30),      # 0-45 degrees (front)
    "front_right": (30, 60),     # 45-90
    "right":       (60, 90),     # 90-135
    "rear_right":  (90, 120),    # 135-180
    "rear":        (120, 150),   # 180-225
    "rear_left":   (150, 180),   # 225-270
    "left":        (180, 210),   # 270-315
    "front_left":  (210, 240),   # 315-360
}

FEATURE_NAMES = [
    # Ego state (7)
    "dist_left", "dist_right", "heading_diff", "speed",
    "steering", "yaw_rate", "lateral_offset",
    # Lidar sectors (8) — min distance in each sector (1.0 = clear, 0.0 = obstacle)
    "lidar_front", "lidar_front_right", "lidar_right", "lidar_rear_right",
    "lidar_rear", "lidar_rear_left", "lidar_left", "lidar_front_left",
    # Navigation (4)
    "navi_forward_dist", "navi_side_dist", "navi_lane_radius", "navi_direction",
    # Engineered (3)
    "min_front_dist", "danger_score", "heading_alignment",
]

NUM_FEATURES = len(FEATURE_NAMES)
NUM_ACTIONS = config.METADRIVE_STEERING_DIM * config.METADRIVE_THROTTLE_DIM


def _build_env_config(seed=0, headless=True):
    return {
        "use_render": not headless,
        "discrete_action": True,
        "discrete_steering_dim": config.METADRIVE_STEERING_DIM,
        "discrete_throttle_dim": config.METADRIVE_THROTTLE_DIM,
        "num_scenarios": config.METADRIVE_NUM_SCENARIOS,
        "start_seed": seed,
        "traffic_density": config.METADRIVE_TRAFFIC_DENSITY,
        "map": config.METADRIVE_MAP,
        "horizon": config.METADRIVE_HORIZON,
        "crash_vehicle_done": True,
        "out_of_road_done": True,
        "log_level": 50,
    }


def extract_features(obs):
    """Extract curated feature dict from raw 259-dim MetaDrive observation."""
    # Ego state
    dist_left = float(obs[0])
    dist_right = float(obs[1])
    heading_diff = float(obs[2])
    speed = float(obs[3])
    steering = float(obs[4])
    yaw_rate = float(obs[7])
    lateral_offset = float(obs[8])

    # Lidar sectors — min distance per sector (lower = closer obstacle)
    lidar = obs[LIDAR_START:LIDAR_START + LIDAR_RAYS]
    sectors = {}
    for name, (start, end) in SECTORS.items():
        sector_rays = lidar[start:end]
        sectors[name] = float(np.min(sector_rays)) if len(sector_rays) > 0 else 1.0

    # Navigation (first checkpoint)
    navi_forward_dist = float(obs[9])
    navi_side_dist = float(obs[10])
    navi_lane_radius = float(obs[11])
    navi_direction = float(obs[12])

    # Engineered
    min_front_dist = min(sectors["front"], sectors["front_left"], sectors["front_right"])
    danger_score = min(1.0 / (min_front_dist + 0.01), 100.0) / 100.0  # normalized
    heading_alignment = 1.0 - abs(heading_diff - 0.5) * 2.0  # 1.0 = aligned

    return {
        "dist_left": round(dist_left, 4),
        "dist_right": round(dist_right, 4),
        "heading_diff": round(heading_diff, 4),
        "speed": round(speed, 4),
        "steering": round(steering, 4),
        "yaw_rate": round(yaw_rate, 4),
        "lateral_offset": round(lateral_offset, 4),
        "lidar_front": round(sectors["front"], 4),
        "lidar_front_right": round(sectors["front_right"], 4),
        "lidar_right": round(sectors["right"], 4),
        "lidar_rear_right": round(sectors["rear_right"], 4),
        "lidar_rear": round(sectors["rear"], 4),
        "lidar_rear_left": round(sectors["rear_left"], 4),
        "lidar_left": round(sectors["left"], 4),
        "lidar_front_left": round(sectors["front_left"], 4),
        "navi_forward_dist": round(navi_forward_dist, 4),
        "navi_side_dist": round(navi_side_dist, 4),
        "navi_lane_radius": round(navi_lane_radius, 4),
        "navi_direction": round(navi_direction, 4),
        "min_front_dist": round(min_front_dist, 4),
        "danger_score": round(danger_score, 4),
        "heading_alignment": round(heading_alignment, 4),
    }


class MetaDriveGame:
    """Wraps MetaDrive env into a GameState-like interface for NeuroLabel."""

    def __init__(self, seed=0, headless=True):
        self.env = MetaDriveEnv(_build_env_config(seed, headless))
        self.obs, self.info = self.env.reset()
        self.alive = True
        self.frame = 0
        self.total_reward = 0.0
        self.seed = seed

    def step(self, action_index):
        """Advance one step. action_index: int in [0, NUM_ACTIONS)."""
        self.obs, reward, terminated, truncated, self.info = self.env.step(action_index)
        self.frame += 1
        self.total_reward += reward
        self.alive = not (terminated or truncated)

    def extract_features(self):
        """Extract curated feature dict from current observation."""
        return extract_features(self.obs)

    def encode(self):
        """Full state snapshot for recording/replay."""
        features = self.extract_features()
        return {
            "features": features,
            "alive": self.alive,
            "frame": self.frame,
            "total_reward": round(self.total_reward, 4),
            "velocity_kmh": round(self.info.get("velocity", 0) * 3.6, 2),
            "crash_vehicle": self.info.get("crash_vehicle", False),
            "crash_object": self.info.get("crash_object", False),
            "out_of_road": self.info.get("out_of_road", False),
            "arrive_dest": self.info.get("arrive_dest", False),
            "route_completion": round(self.info.get("route_completion", 0), 4),
        }

    def close(self):
        self.env.close()
