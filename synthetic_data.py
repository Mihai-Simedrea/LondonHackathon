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


def generate_synthetic_eeg(duration_seconds, sample_rate=250, base_time=None):
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

    if base_time is None:
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


def generate_synthetic_game(duration_seconds, base_time=None):
    """
    Generate synthetic game recording using heuristic AI with OC-correlated noise.

    Records at ~10 Hz (every 6 frames) to capture close-obstacle scenarios.
    First 60%: good decisions (heuristic AI, ~90% optimal)
    Last 40%: bad decisions (70% corrupted toward nearest obstacle)

    Returns: list of game state dicts
    """
    RECORD_INTERVAL = 6  # match data_recorder.py
    game = GameState(seed=42)
    records = []
    transition_sec = int(0.6 * duration_seconds)
    if base_time is None:
        base_time = time.time()
    rng = np.random.RandomState(42)

    total_frames = duration_seconds * FPS
    decision = 0
    record_idx = 0

    for frame in range(total_frames):
        if not game.alive:
            game = GameState(seed=42 + frame)

        current_sec = frame // FPS

        # Make a new decision every second (every FPS frames)
        if frame % FPS == 0:
            decision = heuristic_decision(game)

            if current_sec >= transition_sec:
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

        # Record at RECORD_INTERVAL
        if frame % RECORD_INTERVAL == 0:
            state = game.encode()
            record = {
                "t": round(base_time + frame / FPS, 2),
                "sec": record_idx,
                "lane": state["lane"],
                "obs": state["obs"],
                "decision": decision,
                "score": state["score"],
                "alive": state["alive"]
            }
            records.append(record)
            record_idx += 1

        # Step game
        if frame % FPS == 0:
            game.step(decision)
        else:
            game.step(0)

        if not game.alive:
            continue

    return records


def generate_synthetic_fnirs(duration_seconds, sample_rate=11, base_time=None):
    """
    Generate synthetic fNIRS data with OC-correlated hemodynamic patterns.

    First 60%: high OC (strong IR/Red = high blood oxygenation)
    Last 40%: low OC (declining optical signals = fatigue)

    Returns: numpy array shape (n_samples, 11) — timestamp + 10 optical/temp channels
    """
    n_samples = duration_seconds * sample_rate

    if base_time is None:
        base_time = time.time()

    t = np.arange(n_samples) / sample_rate
    transition_point = int(0.6 * n_samples)
    transition_width = int(0.05 * n_samples)

    # Blend factor: 1.0 = high OC, 0.0 = low OC
    blend = np.ones(n_samples)
    blend[transition_point:transition_point + transition_width] = np.linspace(1, 0, transition_width)
    blend[transition_point + transition_width:] = 0

    noise = np.random.randn(n_samples) * 50

    # Typical Mendi ADC values: IR ~15000-25000, Red ~2000-4000, Amb ~-500-0
    # High OC: strong IR (high blood oxygenation), moderate Red
    # Low OC: declining IR (less oxygenation), rising Red (more deoxygenated Hb)
    ir_base = 20000
    red_base = 3000

    ir_l = ir_base + blend * 5000 - (1 - blend) * 3000 + noise
    red_l = red_base + (1 - blend) * 1500 - blend * 500 + noise * 0.3
    amb_l = -300 + np.random.randn(n_samples) * 50

    ir_r = ir_base + blend * 4500 - (1 - blend) * 2800 + noise * 0.9
    red_r = red_base + (1 - blend) * 1400 - blend * 450 + noise * 0.25
    amb_r = -250 + np.random.randn(n_samples) * 45

    # Pulse channel (short-distance, mostly motion artifact reference)
    ir_p = 180000 + np.random.randn(n_samples) * 500
    red_p = 75000 + np.random.randn(n_samples) * 300
    amb_p = -2300 + np.random.randn(n_samples) * 100

    temp = 32.0 + blend * 1.5 + np.random.randn(n_samples) * 0.2

    data = np.column_stack([
        base_time + t,
        ir_l, red_l, amb_l,
        ir_r, red_r, amb_r,
        ir_p, red_p, amb_p,
        temp,
    ])

    return data


def generate_synthetic(duration_seconds=600):
    """
    Generate complete synthetic dataset (EEG or fNIRS based on config.DEVICE_MODE).

    Args:
        duration_seconds: total seconds of simulated gameplay (default 600 = 10 min)

    Returns:
        tuple: (brain_csv_path, game_jsonl_path)
    """
    mode = config.DEVICE_MODE
    print(f"Generating {duration_seconds}s of synthetic data ({mode.upper()} mode)...")

    base_time = time.time()

    if mode == "fnirs":
        print("  Generating synthetic fNIRS...")
        fnirs_data = generate_synthetic_fnirs(duration_seconds, base_time=base_time)

        brain_path = config.FNIRS_CSV
        with open(brain_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + config.FNIRS_CHANNEL_NAMES)
            for row in fnirs_data:
                writer.writerow([round(v, 4) for v in row.tolist()])
        print(f"  Saved fNIRS: {brain_path} ({len(fnirs_data)} samples)")
    else:
        print("  Generating synthetic EEG...")
        eeg_data = generate_synthetic_eeg(duration_seconds, base_time=base_time)

        brain_path = config.EEG_CSV
        with open(brain_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + config.CHANNEL_NAMES)
            for row in eeg_data:
                writer.writerow(row.tolist())
        print(f"  Saved EEG: {brain_path} ({len(eeg_data)} samples)")

    # Generate game data
    print("  Generating synthetic game recordings...")
    game_records = generate_synthetic_game(duration_seconds, base_time=base_time)

    game_path = config.DATA_DIR / "game_recording.jsonl"
    with open(game_path, 'w') as f:
        for record in game_records:
            f.write(json.dumps(record) + '\n')
    print(f"  Saved game: {game_path} ({len(game_records)} records)")

    high_oc_count = int(0.6 * len(game_records))
    low_oc_count = len(game_records) - high_oc_count
    print(f"\n  Summary:")
    print(f"    Total records: {len(game_records)}")
    print(f"    High OC (first 60%): {high_oc_count}")
    print(f"    Low OC (last 40%): {low_oc_count}")

    return brain_path, game_path


if __name__ == "__main__":
    generate_synthetic()
