"""Headless MetaDrive simulation - runs trained model against MetaDrive env."""

from __future__ import annotations

from neurolabel.backends.metadrive.env import MetaDriveGame, NUM_ACTIONS
from neurolabel.backends.metadrive.model import predict as model_predict

MAX_STEPS = 2000
RECORD_INTERVAL = 5  # record every 5th step


def simulate(model, seed=0):
    """
    Run one headless MetaDrive simulation with the trained model.

    Returns:
        dict: {alive_time, seed, distance, total_reward, crash_vehicle,
               out_of_road, arrive_dest, route_completion, frames}
    """
    game = MetaDriveGame(seed=seed, headless=True)
    frames = []
    # Default action: go straight + coast (middle of 5x3 grid)
    default_action = (NUM_ACTIONS // 2)

    while game.alive and game.frame < MAX_STEPS:
        features = game.extract_features()
        try:
            action = model_predict(model, features)
        except Exception:
            action = default_action

        if game.frame % RECORD_INTERVAL == 0:
            state = game.encode()
            state["action"] = action
            frames.append(state)

        game.step(action)

    result = {
        "alive_time": game.frame,
        "seed": seed,
        "total_reward": round(game.total_reward, 4),
        "route_completion": round(game.info.get("route_completion", 0), 4),
        "crash_vehicle": game.info.get("crash_vehicle", False),
        "out_of_road": game.info.get("out_of_road", False),
        "arrive_dest": game.info.get("arrive_dest", False),
        "frames": frames,
    }

    game.close()
    return result
