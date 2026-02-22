#!/usr/bin/env python3
"""Generate synthetic EEG + game data for testing the NeuroLabel pipeline without hardware."""

import json
import csv
import time
import numpy as np
from pathlib import Path
from game_engine import GameState, LANE_COUNT, HEIGHT, FPS

# Import config paths
import config


def heuristic_decision(game_state):
    """
    Simple heuristic AI: move toward the lane with the most distant nearest obstacle.
    Returns: -1 (left), 0 (stay), 1 (right)
    """
    distances = game_state.get_nearest_obstacles()
    current_lane = game_state.player.lane

    # Find the safest lane (most distant nearest obstacle)
    best_lane = max(range(LANE_COUNT), key=lambda i: distances[i])

    if best_lane < current_lane:
        return -1  # move left
    elif best_lane > current_lane:
        return 1   # move right
    else:
        return 0   # stay


def generate_synthetic_eeg(duration_seconds, sample_rate=250):
    """
    Generate synthetic EEG data (20 channels) with OC-correlated spectral properties.

    First 60% of duration: high OC (high beta, low alpha/theta)
    Last 40%: low OC (high alpha/theta, low beta) — simulating fatigue

    Returns: numpy array shape (duration_seconds * sample_rate, 21) — timestamp + 20 channels
    """
    n_samples = duration_seconds * sample_rate
    n_channels = 20

    data = np.zeros((n_samples, n_channels + 1))  # +1 for timestamp

    transition_point = int(0.6 * n_samples)
    t = np.arange(n_samples) / sample_rate

    # Base timestamp
    base_time = time.time()
    data[:, 0] = base_time + t

    for ch in range(n_channels):
        # All channels get some noise
        noise = np.random.randn(n_samples) * 5  # 5 uV noise

        # Theta (4-8 Hz) component
        theta_freq = 6 + np.random.rand() * 2
        theta = np.sin(2 * np.pi * theta_freq * t)

        # Alpha (8-13 Hz) component
        alpha_freq = 10 + np.random.rand() * 2
        alpha = np.sin(2 * np.pi * alpha_freq * t)

        # Beta (13-30 Hz) component
        beta_freq = 20 + np.random.rand() * 5
        beta = np.sin(2 * np.pi * beta_freq * t)

        # First 60%: HIGH OC — strong beta, weak alpha/theta
        # High focus: high engagement (beta/(alpha+theta)), low fatigue (theta/alpha)
        signal_high = beta * 15 + alpha * 3 + theta * 2 + noise

        # Last 40%: LOW OC — weak beta, strong alpha/theta
        # Low focus: low engagement, high fatigue
        signal_low = beta * 3 + alpha * 12 + theta * 15 + noise

        # Smooth transition
        transition_width = int(0.05 * n_samples)
        blend = np.ones(n_samples)
        blend[transition_point:transition_point + transition_width] = np.linspace(1, 0, transition_width)
        blend[transition_point + transition_width:] = 0

        signal = signal_high * blend + signal_low * (1 - blend)

        # Convert to microvolts range (typical EEG: 10-100 uV)
        data[:, ch + 1] = signal

    return data


def generate_synthetic_game(duration_seconds):
    """
    Generate synthetic game recording using heuristic AI with OC-correlated noise.

    First 60%: good decisions (heuristic AI, ~90% optimal)
    Last 40%: bad decisions (70% corrupted toward nearest obstacle)

    Returns: list of per-second game state dicts
    """
    game = GameState(seed=42)
    records = []
    transition_sec = int(0.6 * duration_seconds)
    base_time = time.time()
    rng = np.random.RandomState(42)

    for sec in range(duration_seconds):
        if not game.alive:
            # Restart the game to keep generating data
            game = GameState(seed=42 + sec)

        # Get heuristic decision
        decision = heuristic_decision(game)

        # In low-OC period (last 40%), corrupt 70% of decisions with
        # actively bad choices (move toward nearest obstacle)
        if sec >= transition_sec:
            if rng.random() < 0.70:
                distances = game.get_nearest_obstacles()
                worst_lane = min(range(LANE_COUNT), key=lambda i: distances[i])
                current_lane = game.player.lane
                if worst_lane < current_lane:
                    decision = -1
                elif worst_lane > current_lane:
                    decision = 1
                else:
                    decision = 0

        # Record state BEFORE stepping
        state = game.encode()
        record = {
            "t": base_time + sec,
            "sec": sec,
            "lane": state["lane"],
            "obs": state["obs"],
            "decision": decision,
            "score": state["score"],
            "alive": state["alive"]
        }
        records.append(record)

        # Step game for 1 second (FPS frames)
        for frame in range(FPS):
            if frame == 0:
                game.step(decision)
            else:
                game.step(0)

            if not game.alive:
                break

    return records


def generate_synthetic(duration_seconds=600):
    """
    Generate complete synthetic dataset.

    Args:
        duration_seconds: total seconds of simulated gameplay (default 600 = 10 min)

    Returns:
        tuple: (eeg_csv_path, game_jsonl_path)
    """
    print(f"Generating {duration_seconds}s of synthetic data...")

    # Generate EEG data
    print("  Generating synthetic EEG...")
    eeg_data = generate_synthetic_eeg(duration_seconds)

    eeg_path = config.DATA_DIR / "eeg_recording.csv"
    with open(eeg_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + config.CHANNEL_NAMES)
        for row in eeg_data:
            writer.writerow(row.tolist())
    print(f"  Saved EEG: {eeg_path} ({len(eeg_data)} samples)")

    # Generate game data
    print("  Generating synthetic game recordings...")
    game_records = generate_synthetic_game(duration_seconds)

    game_path = config.DATA_DIR / "game_recording.jsonl"
    with open(game_path, 'w') as f:
        for record in game_records:
            f.write(json.dumps(record) + '\n')
    print(f"  Saved game: {game_path} ({len(game_records)} records)")

    # Print summary
    high_oc_count = int(0.6 * len(game_records))
    low_oc_count = len(game_records) - high_oc_count
    print(f"\n  Summary:")
    print(f"    Total seconds: {len(game_records)}")
    print(f"    High OC (first 60%): {high_oc_count}s")
    print(f"    Low OC (last 40%): {low_oc_count}s")
    print(f"    Corrupted decisions in low-OC: ~{int(low_oc_count * 0.70)}")

    return eeg_path, game_path


if __name__ == "__main__":
    generate_synthetic()
