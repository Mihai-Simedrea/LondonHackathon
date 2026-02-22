"""Generate synthetic MetaDrive + brain data by running actual simulations."""

from __future__ import annotations

import json
import csv
import time
import numpy as np
import config
from neurolabel.backends.metadrive.env import MetaDriveGame, NUM_ACTIONS


def _heuristic_action(features):
    """Simple heuristic policy: follow road, avoid obstacles, maintain speed."""
    heading = features["heading_diff"]
    speed = features["speed"]
    front = features["lidar_front"]
    front_left = features["lidar_front_left"]
    front_right = features["lidar_front_right"]

    # Steering: 0=hard left, 1=left, 2=straight, 3=right, 4=hard right
    if front < 0.5:
        if front_left > front_right:
            steer = 0 if front_left > 0.6 else 1
        else:
            steer = 4 if front_right > 0.6 else 3
    elif heading > 0.55:
        steer = 1
    elif heading < 0.45:
        steer = 3
    else:
        steer = 2

    # Throttle: 0=brake, 1=coast, 2=accelerate
    if front < 0.3:
        throttle = 0
    elif speed < 0.05:
        throttle = 2  # always accelerate when slow
    elif speed > 0.15:
        throttle = 1
    else:
        throttle = 2

    return steer * config.METADRIVE_THROTTLE_DIM + throttle


def _collect_episodes(n_records, policy="heuristic", seed_start=0):
    """Run MetaDrive episodes and collect (features, action) records.

    policy: "heuristic" for good driving, "random" for bad driving
    """
    records = []
    rng = np.random.RandomState(seed_start)
    seed = seed_start
    record_every = 3  # record every 3rd frame for variety

    while len(records) < n_records:
        game = MetaDriveGame(seed=seed, headless=True)
        step = 0

        while game.alive and game.frame < config.METADRIVE_HORIZON and len(records) < n_records:
            features = game.extract_features()

            if policy == "heuristic":
                action = int(_heuristic_action(features))
            elif policy == "random":
                action = int(rng.randint(NUM_ACTIONS))
            else:  # "mixed" — 50% random
                if rng.random() < 0.5:
                    action = int(rng.randint(NUM_ACTIONS))
                else:
                    action = int(_heuristic_action(features))

            if step % record_every == 0:
                records.append((features, action, float(game.total_reward)))

            game.step(action)
            step += 1

        game.close()
        seed += 1

    return records[:n_records]


def generate_synthetic_metadrive(duration_seconds=600):
    """Generate synthetic MetaDrive game + brain data using real simulations.

    High OC (first 60%): heuristic policy — good driving decisions
    Low OC (last 40%): random policy — erratic decisions
    """
    print(f"Generating {duration_seconds}s of synthetic MetaDrive data...")
    base_time = time.time()
    record_rate = 10  # 10 Hz
    total_records = duration_seconds * record_rate

    transition = int(0.6 * total_records)
    n_low = total_records - transition

    print(f"  Collecting {transition} high-OC records (heuristic policy)...")
    t0 = time.time()
    high_oc_records = _collect_episodes(transition, policy="heuristic", seed_start=0)
    print(f"    Done in {time.time() - t0:.1f}s")

    print(f"  Collecting {n_low} low-OC records (random policy)...")
    t0 = time.time()
    low_oc_records = _collect_episodes(n_low, policy="random", seed_start=1000)
    print(f"    Done in {time.time() - t0:.1f}s")

    all_records = high_oc_records + low_oc_records

    # Write game JSONL
    game_path = config.GAME_JSONL
    with open(game_path, "w") as f:
        for i, (features, action, reward) in enumerate(all_records):
            record = {
                "t": round(base_time + i / record_rate, 2),
                "sec": i,
                "features": features,
                "action": action,
                "alive": True,
                "reward": round(float(reward), 4),
            }
            f.write(json.dumps(record) + "\n")

    print(f"  Saved game: {game_path} ({total_records} records)")

    # Generate brain data (fNIRS or EEG based on config)
    if config.DEVICE_MODE == "fnirs":
        from neurolabel.backends.velocity.synthetic import generate_synthetic_fnirs
        fnirs_data = generate_synthetic_fnirs(duration_seconds, base_time=base_time)
        brain_path = config.FNIRS_CSV
        with open(brain_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + config.FNIRS_CHANNEL_NAMES)
            for row in fnirs_data:
                writer.writerow([round(v, 4) for v in row.tolist()])
        print(f"  Saved fNIRS: {brain_path} ({len(fnirs_data)} samples)")
    else:
        from neurolabel.backends.velocity.synthetic import generate_synthetic_eeg
        eeg_data = generate_synthetic_eeg(duration_seconds, base_time=base_time)
        brain_path = config.EEG_CSV
        with open(brain_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + config.CHANNEL_NAMES)
            for row in eeg_data:
                writer.writerow(row.tolist())
        print(f"  Saved EEG: {brain_path} ({len(eeg_data)} samples)")

    print(f"  Summary: {transition} high-OC + {n_low} low-OC records")
    return brain_path, game_path


if __name__ == "__main__":
    generate_synthetic_metadrive()
