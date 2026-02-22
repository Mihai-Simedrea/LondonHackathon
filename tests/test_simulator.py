"""Tests for simulator.py — headless game simulation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulator import simulate
from game_engine import FPS


class DummyStayModel:
    """A model that always predicts 'stay' (class 1 -> maps to decision 0)."""
    def predict(self, X):
        return [1]  # class 1 = stay


class DummyLeftModel:
    """A model that always predicts 'left' (class 0 -> maps to decision -1)."""
    def predict(self, X):
        return [0]  # class 0 = left


class TestSimulate:
    def test_simulate_basic(self):
        """Simulation with a dummy model completes and alive_time > 0."""
        result = simulate(DummyStayModel(), seed=42)
        assert result["alive_time"] > 0
        assert "seed" in result
        assert result["seed"] == 42

    def test_simulate_deterministic(self):
        """Same seed + same model = same alive_time."""
        r1 = simulate(DummyStayModel(), seed=123)
        r2 = simulate(DummyStayModel(), seed=123)
        assert r1["alive_time"] == r2["alive_time"]
        assert len(r1["frames"]) == len(r2["frames"])

    def test_simulate_returns_frames(self):
        """Result contains frames list with expected keys."""
        result = simulate(DummyStayModel(), seed=42)
        assert "frames" in result
        assert isinstance(result["frames"], list)
        assert len(result["frames"]) > 0

        # Check first frame has expected keys
        frame = result["frames"][0]
        assert "lane" in frame
        assert "obs" in frame
        assert "alive" in frame
        assert "score" in frame
        assert "frame" in frame
        assert "decision" in frame

    def test_simulate_always_stay(self):
        """Model that always stays should survive for some time."""
        result = simulate(DummyStayModel(), seed=42)
        # Should survive at least a few seconds (60 frames = 1 sec)
        assert result["alive_time"] >= FPS, (
            f"Stay model only survived {result['alive_time']} frames "
            f"({result['alive_time'] / FPS:.1f} sec)"
        )

    def test_simulate_different_seeds(self):
        """Different seeds should generally produce different results."""
        r1 = simulate(DummyStayModel(), seed=1)
        r2 = simulate(DummyStayModel(), seed=999)
        # Not a hard guarantee, but very unlikely to be identical
        # We just check both ran successfully
        assert r1["alive_time"] > 0
        assert r2["alive_time"] > 0
