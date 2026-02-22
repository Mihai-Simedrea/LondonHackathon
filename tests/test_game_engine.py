"""Tests for game_engine.py — Player, Obstacle, and GameState logic."""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from game_engine import Player, Obstacle, GameState, LANE_COUNT, LANE_CENTERS, HEIGHT


class TestPlayer:
    def test_player_initial_state(self):
        """Player starts in lane 1 (middle lane)."""
        p = Player()
        assert p.lane == 1
        assert p.x == float(LANE_CENTERS[1])

    def test_player_move_left(self):
        """move(-1) changes lane from 1 to 0."""
        p = Player()
        p.move(-1)
        assert p.lane == 0
        assert p.target_x == float(LANE_CENTERS[0])

    def test_player_move_right(self):
        """move(1) changes lane from 1 to 2."""
        p = Player()
        p.move(1)
        assert p.lane == 2
        assert p.target_x == float(LANE_CENTERS[2])

    def test_player_move_boundary_left(self):
        """Cannot move past lane 0."""
        p = Player()
        p.move(-1)  # lane 0
        p.move(-1)  # should stay at lane 0
        assert p.lane == 0

    def test_player_move_boundary_right(self):
        """Cannot move past lane 2."""
        p = Player()
        p.move(1)   # lane 2
        p.move(1)   # should stay at lane 2
        assert p.lane == 2


class TestGameState:
    def test_gamestate_step(self):
        """Advancing frames increases score and frame count."""
        gs = GameState(seed=42)
        gs.step(0)
        assert gs.score == 1
        assert gs.frame == 1
        gs.step(0)
        assert gs.score == 2
        assert gs.frame == 2

    def test_gamestate_collision(self):
        """Collision detection kills the player when obstacle overlaps."""
        gs = GameState(seed=42)
        # Place an obstacle directly on the player
        obs = Obstacle(gs.player.lane)
        obs.y = gs.player.y  # same vertical position
        gs.obstacles.append(obs)
        # Step should detect collision
        gs.step(0)
        assert gs.alive is False

    def test_gamestate_encode(self):
        """encode() returns dict with expected keys and correct types."""
        gs = GameState(seed=42)
        gs.step(0)
        encoded = gs.encode()
        assert "lane" in encoded
        assert "obs" in encoded
        assert "alive" in encoded
        assert "score" in encoded
        assert "frame" in encoded
        assert isinstance(encoded["lane"], int)
        assert isinstance(encoded["obs"], list)
        assert isinstance(encoded["alive"], bool)
        assert isinstance(encoded["score"], int)
        assert isinstance(encoded["frame"], int)

    def test_gamestate_deterministic(self):
        """Same seed produces same game sequence."""
        def run_game(seed):
            gs = GameState(seed=seed)
            for _ in range(200):
                gs.step(0)
                if not gs.alive:
                    break
            return gs.score, gs.frame, gs.alive

        r1 = run_game(123)
        r2 = run_game(123)
        assert r1 == r2

    def test_get_nearest_obstacles(self):
        """get_nearest_obstacles returns correct per-lane distances."""
        gs = GameState(seed=42)
        # No obstacles yet — all distances should be 1.0
        dists = gs.get_nearest_obstacles()
        assert len(dists) == LANE_COUNT
        assert all(d == 1.0 for d in dists)

        # Place an obstacle in lane 0 ahead of the player
        obs = Obstacle(0)
        obs.y = gs.player.y - HEIGHT * 0.5  # halfway up the screen
        gs.obstacles.append(obs)

        dists = gs.get_nearest_obstacles()
        assert dists[0] < 1.0  # lane 0 should have a closer obstacle
        assert dists[1] == 1.0  # lanes 1 and 2 still clear
        assert dists[2] == 1.0
