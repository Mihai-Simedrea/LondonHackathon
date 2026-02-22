#!/usr/bin/env python3
"""Headless game simulator — runs game with model decisions, records replay data."""

from game_engine import GameState, FPS

# Safety limit: stop if game exceeds this many frames (~3 minutes at 60fps)
MAX_FRAMES = 12_000


class SmoothDecisionFilter:
    """Prevents jittery lane changes by requiring consistent model output before switching."""

    def __init__(self, min_hold=2):
        self.current_action = 1  # stay (class 1)
        self.hold_counter = 0
        self.min_hold = min_hold

    def filter(self, raw_action, min_dist=1.0):
        # Emergency override: if closest obstacle < 0.15, allow immediate change
        if min_dist < 0.15:
            self.current_action = raw_action
            self.hold_counter = 0
            return raw_action

        if raw_action != self.current_action:
            self.hold_counter += 1
            if self.hold_counter >= self.min_hold:
                self.current_action = raw_action
                self.hold_counter = 0
                return raw_action
            return self.current_action  # hold previous action
        else:
            self.hold_counter = 0
            return raw_action


def simulate(model, seed=0):
    """
    Run one headless game simulation.

    Args:
        model: trained sklearn model (or any object with predict method
               compatible with model.predict)
        seed: random seed for deterministic replay

    Returns:
        dict: {
            'alive_time': int (frames survived),
            'seed': int,
            'frames': list of frame dicts for replay
        }

    Each frame dict contains the output of GameState.encode() plus a
    'decision' key with the model's current decision (-1, 0, or 1).
    """
    from model import predict as model_predict

    game = GameState(seed=seed)
    frames = []
    current_decision = 0
    smooth_filter = SmoothDecisionFilter(min_hold=2)

    # 8 decisions per second = every FPS/8 frames ≈ every 7-8 frames
    decision_interval = max(1, FPS // 8)

    while game.alive and game.frame < MAX_FRAMES:
        # Decide 8 times per second
        if game.frame % decision_interval == 0:
            nearest = game.get_nearest_obstacles()
            state_input = {
                "lane": game.player.lane,
                "nearest_obstacles": nearest,
            }
            try:
                raw_decision = model_predict(model, state_input)
                current_decision = smooth_filter.filter(raw_decision, min(nearest))
            except Exception:
                current_decision = 0  # default to stay on error

        # Record every other frame for replay (keeps data manageable)
        if game.frame % 2 == 0:
            state = game.encode()
            state["decision"] = current_decision
            frames.append(state)

        # Apply decision on decision frames, stay otherwise
        if game.frame % decision_interval == 0:
            game.step(current_decision)
        else:
            game.step(0)

    # Record final frame on death
    if frames and frames[-1].get("frame") != game.frame:
        final = game.encode()
        final["decision"] = current_decision
        frames.append(final)

    return {
        "alive_time": game.frame,
        "seed": seed,
        "frames": frames,
    }


def simulate_batch(model, seeds):
    """Run multiple simulations sequentially.

    Used by parallel_runner in worker processes to avoid per-seed overhead
    of importing / loading the model.
    """
    return [simulate(model, seed) for seed in seeds]


if __name__ == "__main__":
    # Quick smoke test with a dummy model that always stays
    class DummyModel:
        def predict(self, X):
            return [1]  # class 1 = stay (mapped to 0 by model.predict)

    result = simulate(DummyModel(), seed=42)
    print(f"Alive time: {result['alive_time']} frames ({result['alive_time'] / FPS:.1f} sec)")
    print(f"Frames recorded: {len(result['frames'])}")
