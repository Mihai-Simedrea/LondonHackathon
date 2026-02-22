"""
ML model training, prediction, and persistence for VELOCITY.
Uses RandomForestClassifier with engineered features and symmetry augmentation.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import csv
from pathlib import Path


def _engineer_features(lane_onehot, obs_d0, obs_d1, obs_d2):
    """
    Build the full 14-feature vector from raw inputs.

    Base features (6): lane_onehot[3] + obs_d0, obs_d1, obs_d2
    Engineered (8): current_lane_dist, left_advantage, right_advantage,
                     min_dist, safest_lane, danger_score, can_go_left, can_go_right
    """
    lane = lane_onehot.index(1) if 1 in lane_onehot else 1

    current_lane_dist = sum(lane_onehot[i] * d for i, d in enumerate([obs_d0, obs_d1, obs_d2]))
    left_advantage = obs_d0 - current_lane_dist
    right_advantage = obs_d2 - current_lane_dist
    min_dist = min(obs_d0, obs_d1, obs_d2)
    dists = [obs_d0, obs_d1, obs_d2]
    safest_lane = int(np.argmax(dists))
    danger_score = 1.0 / (current_lane_dist + 0.01)
    can_go_left = 0 if lane == 0 else 1
    can_go_right = 0 if lane == 2 else 1

    return lane_onehot + [obs_d0, obs_d1, obs_d2,
                          current_lane_dist, left_advantage, right_advantage,
                          min_dist, safest_lane, danger_score,
                          can_go_left, can_go_right]


def train_model(dataset_csv_path):
    """
    Train RandomForestClassifier on dataset CSV with symmetry augmentation.

    CSV columns: lane, obs_d0, obs_d1, obs_d2, decision, oc_score
    Input features (14): lane one-hot [3] + obs distances [3] + engineered [8]
    Output: decision class (0=left, 1=stay, 2=right)
    """
    rows = []
    with open(dataset_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if len(rows) < 10:
        print(f"  WARNING: very few training samples ({len(rows)}). Model may be unreliable.")

    X = []
    y = []
    for row in rows:
        lane = int(row["lane"])
        lane_onehot = [0, 0, 0]
        lane_onehot[lane] = 1

        obs_d0 = float(row["obs_d0"])
        obs_d1 = float(row["obs_d1"])
        obs_d2 = float(row["obs_d2"])

        X.append(_engineer_features(lane_onehot, obs_d0, obs_d1, obs_d2))

        decision = int(row["decision"])
        # Map: -1 -> 0, 0 -> 1, 1 -> 2
        y.append(decision + 1)

    # --- Symmetry mirroring: swap lane 0 <-> 2, obs_d0 <-> obs_d2, action left <-> right ---
    mirror_X = []
    mirror_y = []
    for row in rows:
        lane = int(row["lane"])
        mirrored_lane = 2 - lane  # 0->2, 1->1, 2->0
        mirror_onehot = [0, 0, 0]
        mirror_onehot[mirrored_lane] = 1

        obs_d0 = float(row["obs_d0"])
        obs_d1 = float(row["obs_d1"])
        obs_d2 = float(row["obs_d2"])
        # Swap d0 and d2
        mirror_X.append(_engineer_features(mirror_onehot, obs_d2, obs_d1, obs_d0))

        decision = int(row["decision"])
        # Map original: -1->0, 0->1, 1->2
        action = decision + 1
        # Mirror action: left(0) <-> right(2), stay(1) stays
        if action == 0:
            mirrored_action = 2
        elif action == 2:
            mirrored_action = 0
        else:
            mirrored_action = 1
        mirror_y.append(mirrored_action)

    X = np.array(X + mirror_X)
    y = np.array(y + mirror_y)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X, y)

    accuracy = model.score(X, y)
    print(f"  Training accuracy: {accuracy:.3f}")
    print(f"  Training samples: {len(X)} ({len(X) // 2} original + {len(X) // 2} mirrored)")
    return model


def predict(model, game_state_dict):
    """
    Predict decision from game state.

    Returns:
        int: -1 (left), 0 (stay), 1 (right)
    """
    lane = game_state_dict["lane"]
    lane_onehot = [0, 0, 0]
    lane_onehot[lane] = 1

    if "nearest_obstacles" in game_state_dict:
        obs_d = game_state_dict["nearest_obstacles"]
    else:
        obs_d = [
            game_state_dict.get("obs_d0", 1.0),
            game_state_dict.get("obs_d1", 1.0),
            game_state_dict.get("obs_d2", 1.0),
        ]

    features = np.array([_engineer_features(lane_onehot, obs_d[0], obs_d[1], obs_d[2])])
    pred_class = model.predict(features)[0]
    # Map back: 0 -> -1, 1 -> 0, 2 -> 1
    return pred_class - 1


def save_model(model, path):
    """Save model to file using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved: {path}")


def load_model(path):
    """Load model from file."""
    return joblib.load(path)
