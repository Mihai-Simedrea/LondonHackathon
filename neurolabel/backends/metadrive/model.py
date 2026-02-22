"""ML model training and prediction for MetaDrive backend."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import csv
from pathlib import Path

from neurolabel.backends.metadrive.env import FEATURE_NAMES, NUM_ACTIONS


def _features_to_vector(features_dict):
    """Convert feature dict to ordered numpy array."""
    return [features_dict[name] for name in FEATURE_NAMES]


def train_model(dataset_csv_path):
    """
    Train RandomForestClassifier on MetaDrive dataset CSV.

    CSV columns: FEATURE_NAMES + ["action", "oc_score"]
    Output: action class (0 through NUM_ACTIONS-1)
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
        features = {name: float(row[name]) for name in FEATURE_NAMES}
        X.append(_features_to_vector(features))
        y.append(int(row["action"]))

    X = np.array(X)
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=15,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    model.fit(X, y)

    from sklearn.model_selection import cross_val_score
    cv_folds = min(5, max(2, len(X) // 10))
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="accuracy")
    print(f"  Cross-val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    print(f"  Training samples: {len(X)}")
    return model


def predict(model, features_dict):
    """Predict action index from feature dict. Returns int 0 to NUM_ACTIONS-1."""
    features = np.array([_features_to_vector(features_dict)])
    return int(model.predict(features)[0])


def save_model(model, path):
    """Save model to file using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  Model saved: {path}")


def load_model(path):
    """Load model from file."""
    return joblib.load(path)
