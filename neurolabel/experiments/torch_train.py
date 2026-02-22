#!/usr/bin/env python3
"""
PyTorch neural network model with MPS GPU acceleration for Apple Silicon.
Drop-in replacement for model.py with live training progress display.
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neurolabel.models.velocity_data import load_velocity_raw_xy


def get_device():
    """Get best available device (MPS for Apple Silicon, else CPU)."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


class GameClassifier(nn.Module):
    """Neural network for lane-change decisions."""
    def __init__(self, in_features=6, n_classes=3, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _load_csv(csv_path):
    """Load dataset CSV, return X (float32), y (int64) numpy arrays."""
    X, y, _rows = load_velocity_raw_xy(csv_path, x_dtype=np.float32, y_dtype=np.int64)
    return X, y


def _print_progress(epoch, total_epochs, loss, acc, lr, elapsed, bar_width=30):
    """Print live training progress bar."""
    pct = epoch / total_epochs
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(
        f"\r  [{bar}] {epoch:3d}/{total_epochs} | "
        f"loss: {loss:.4f} | acc: {acc:.1%} | lr: {lr:.1e} | {elapsed:.1f}s"
    )
    sys.stdout.flush()


def train_model(dataset_csv_path, epochs=200, lr=1e-3, batch_size=32, hidden=128):
    """
    Train PyTorch classifier on dataset CSV with MPS GPU acceleration.

    Returns: dict with 'model' (state_dict), 'config' (hyperparams), 'device' str
    """
    X_np, y_np = _load_csv(dataset_csv_path)
    n_samples = len(X_np)

    if n_samples < 10:
        print(f"  WARNING: very few training samples ({n_samples}). Model may be unreliable.")

    # Move data to GPU
    X = torch.from_numpy(X_np).to(DEVICE)
    y = torch.from_numpy(y_np).to(DEVICE)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        pin_memory=False, num_workers=0)

    # Build model on GPU
    model = GameClassifier(in_features=6, n_classes=3, hidden=hidden).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Device: {DEVICE} | Samples: {n_samples} | Epochs: {epochs} | Hidden: {hidden}")

    best_acc = 0
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0

        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(X_batch)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()

        scheduler.step()
        avg_loss = total_loss / n_samples
        acc = correct / n_samples
        current_lr = scheduler.get_last_lr()[0]

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        _print_progress(epoch, epochs, avg_loss, acc, current_lr, time.time() - start)

    print()  # newline after progress bar
    print(f"  Training accuracy: {best_acc:.3f}")
    print(f"  Training samples: {n_samples}")
    print(f"  Training time: {time.time() - start:.1f}s")

    # Return wrapper that's compatible with sklearn-style predict
    return TorchModelWrapper(best_state, hidden=hidden)


class TorchModelWrapper:
    """Wraps PyTorch model to be compatible with sklearn-style predict interface."""

    def __init__(self, state_dict, hidden=128):
        self.state_dict = state_dict
        self.hidden = hidden
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            self._model = GameClassifier(in_features=6, n_classes=3, hidden=self.hidden)
            self._model.load_state_dict(self.state_dict)
            self._model.to(DEVICE)
            self._model.eval()

    def predict(self, X):
        """Predict classes for feature array X. Returns numpy array of class indices."""
        self._ensure_model()
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        X = X.to(DEVICE)
        with torch.no_grad():
            logits = self._model(X)
            return logits.argmax(dim=1).cpu().numpy()

    def score(self, X, y):
        """Return accuracy on X, y."""
        preds = self.predict(X)
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        return (preds == y).mean()


def predict(model, game_state_dict):
    """
    Predict decision from game state. Compatible with simulator.py.

    Args:
        model: TorchModelWrapper or sklearn model
        game_state_dict: dict with 'lane' and obstacle distances

    Returns:
        int: -1 (left), 0 (stay), 1 (right)
    """
    lane = game_state_dict["lane"]
    lane_onehot = [0.0, 0.0, 0.0]
    lane_onehot[lane] = 1.0

    if "nearest_obstacles" in game_state_dict:
        obs_d = game_state_dict["nearest_obstacles"]
    else:
        obs_d = [
            game_state_dict.get("obs_d0", 1.0),
            game_state_dict.get("obs_d1", 1.0),
            game_state_dict.get("obs_d2", 1.0),
        ]

    features = np.array([lane_onehot + list(obs_d)], dtype=np.float32)

    if isinstance(model, TorchModelWrapper):
        pred_class = model.predict(features)[0]
    else:
        # sklearn model fallback
        pred_class = model.predict(features)[0]

    return int(pred_class) - 1


def save_model(model, path):
    """Save model to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, TorchModelWrapper):
        torch.save({"state_dict": model.state_dict, "hidden": model.hidden}, path)
    else:
        import joblib
        joblib.dump(model, path)
    print(f"  Model saved: {path}")


def load_model(path):
    """Load model from file."""
    path = Path(path)
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "state_dict" in data:
            return TorchModelWrapper(data["state_dict"], hidden=data.get("hidden", 128))
    except Exception:
        pass
    # Fallback to joblib (sklearn model)
    import joblib
    return joblib.load(path)


if __name__ == "__main__":
    import config
    print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
    if config.DATASET_CLEAN.exists():
        print("\nTraining on clean dataset...")
        model = train_model(str(config.DATASET_CLEAN), epochs=100)
    else:
        print("No dataset found. Run: python pipeline.py demo --synthetic --dev")
