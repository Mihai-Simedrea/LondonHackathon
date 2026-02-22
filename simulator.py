#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.backends.velocity.simulation`."""

from neurolabel.backends.velocity.simulation import *  # noqa: F401,F403


if __name__ == "__main__":
    class DummyModel:
        def predict(self, X):
            return [1]

    from neurolabel.backends.velocity.simulation import simulate
    from game_engine import FPS

    result = simulate(DummyModel(), seed=42)
    print(f"Alive time: {result['alive_time']} frames ({result['alive_time'] / FPS:.1f} sec)")
    print(f"Frames recorded: {len(result['frames'])}")
