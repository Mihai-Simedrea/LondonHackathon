from __future__ import annotations

"""
ML model training, prediction, and persistence for VELOCITY.
Uses RandomForestClassifier with engineered features and symmetry augmentation.
"""

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import csv
from pathlib import Path

LANE_DIM = 3
DECISION_CLASS_OFFSET = 1  # maps {-1,0,1} <-> {0,1,2}


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


def _read_dataset_rows(dataset_csv_path):
    """Load training rows from CSV into memory."""
    with open(dataset_csv_path, "r", newline="") as f:
        return list(csv.DictReader(f))


def _lane_to_onehot(lane):
    lane_onehot = [0] * LANE_DIM
    lane_onehot[int(lane)] = 1
    return lane_onehot


def _decision_to_class(decision):
    """Map decision {-1, 0, 1} to classifier class {0, 1, 2}."""
    return int(decision) + DECISION_CLASS_OFFSET


def _class_to_decision(pred_class):
    """Map classifier class {0, 1, 2} back to decision {-1, 0, 1}."""
    return int(pred_class) - DECISION_CLASS_OFFSET


def _mirror_decision_class(action_class):
    """Mirror lane-change class: left <-> right, stay unchanged."""
    if action_class == 0:
        return 2
    if action_class == 2:
        return 0
    return 1


def _row_to_features_and_class(row):
    lane = int(row["lane"])
    obs_d0 = float(row["obs_d0"])
    obs_d1 = float(row["obs_d1"])
    obs_d2 = float(row["obs_d2"])
    features = _engineer_features(_lane_to_onehot(lane), obs_d0, obs_d1, obs_d2)
    action_class = _decision_to_class(row["decision"])
    return features, action_class


def _mirror_row_to_features_and_class(row):
    lane = int(row["lane"])
    mirrored_lane = 2 - lane
    obs_d0 = float(row["obs_d0"])
    obs_d1 = float(row["obs_d1"])
    obs_d2 = float(row["obs_d2"])
    features = _engineer_features(_lane_to_onehot(mirrored_lane), obs_d2, obs_d1, obs_d0)
    action_class = _mirror_decision_class(_decision_to_class(row["decision"]))
    return features, action_class


def _extract_obs_distances(game_state_dict):
    if "nearest_obstacles" in game_state_dict:
        return game_state_dict["nearest_obstacles"]
    return [
        game_state_dict.get("obs_d0", 1.0),
        game_state_dict.get("obs_d1", 1.0),
        game_state_dict.get("obs_d2", 1.0),
    ]


def train_model(dataset_csv_path):
    """
    Train RandomForestClassifier on dataset CSV with symmetry augmentation.

    CSV columns: lane, obs_d0, obs_d1, obs_d2, decision, oc_score
    Input features (14): lane one-hot [3] + obs distances [3] + engineered [8]
    Output: decision class (0=left, 1=stay, 2=right)
    """
    rows = _read_dataset_rows(dataset_csv_path)

    if len(rows) < 10:
        print(f"  WARNING: very few training samples ({len(rows)}). Model may be unreliable.")

    # Build training set + mirrored augmentation.
    X = []
    y = []
    for row in rows:
        features, action_class = _row_to_features_and_class(row)
        X.append(features)
        y.append(action_class)

        mirror_features, mirror_action_class = _mirror_row_to_features_and_class(row)
        X.append(mirror_features)
        y.append(mirror_action_class)

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    model.fit(X, y)

    from sklearn.model_selection import cross_val_score
    cv_folds = min(5, len(X) // 2)
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
        print(f"  Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    else:
        print("  Cross-val accuracy: skipped (not enough samples)")
    print(f"  Training samples: {len(X)} ({len(X) // 2} original + {len(X) // 2} mirrored)")
    return model


def predict(model, game_state_dict):
    """
    Predict decision from game state.

    Returns:
        int: -1 (left), 0 (stay), 1 (right)
    """
    lane = int(game_state_dict["lane"])
    obs_d = _extract_obs_distances(game_state_dict)
    features = np.array([_engineer_features(_lane_to_onehot(lane), obs_d[0], obs_d[1], obs_d[2])])
    pred_class = model.predict(features)[0]
    return _class_to_decision(pred_class)


def save_model(model, path):
    """Save model to file using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved: {path}")


def load_model(path):
    """Load model from file."""
    return joblib.load(path)
