#!/usr/bin/env python3
from __future__ import annotations
"""
MetaDrive — Human Play + Brain Recorder

Launches MetaDrive with keyboard control and optionally records fNIRS brain
data from a local private headset integration in a background thread. Everything in one file.

Usage:
    python metadrive_recorder.py              # game + brain recording
    python metadrive_recorder.py --no-brain   # game only, no headband needed
    python metadrive_recorder.py 120          # stop after 120 seconds
"""

import json
import time
import signal
import sys
import csv
import threading

import numpy as np
from metadrive import MetaDriveEnv

import config
from neurolabel.brain.acquisition.fnirs_provider import get_private_fnirs_client_class
from neurolabel.backends.metadrive.env import extract_features, _build_env_config
from neurolabel.backends.metadrive.action_space import (
    continuous_to_discrete as _shared_continuous_to_discrete,
    steering_values as _shared_steering_values,
    throttle_values as _shared_throttle_values,
)

# ── Recording parameters ──────────────────────────────────
RECORD_HZ = 10                    # target recording rate
RECORD_INTERVAL_SEC = 1.0 / RECORD_HZ
STATUS_INTERVAL_SEC = 10.0        # print status every 10 seconds

# ── Discrete action grid ──────────────────────────────────
STEERING_VALUES = _shared_steering_values()
THROTTLE_VALUES = _shared_throttle_values()


def _build_manual_env_config(seed=0):
    """Build MetaDrive config with rendering + native keyboard control."""
    cfg = _build_env_config(seed=seed, headless=False)
    cfg["manual_control"] = True
    cfg.pop("discrete_action", None)
    cfg.pop("discrete_steering_dim", None)
    cfg.pop("discrete_throttle_dim", None)
    return cfg


def _continuous_to_discrete(steer, throttle):
    """Quantize continuous [steer, throttle] to a discrete index."""
    return _shared_continuous_to_discrete(float(steer), float(throttle))


def _get_human_action(env):
    """Read the continuous action the human applied this step."""
    try:
        vehicle = env.vehicle
        if hasattr(vehicle, "before_step_info"):
            raw = vehicle.before_step_info.get("raw_action", None)
            if raw is not None and hasattr(raw, "__len__") and len(raw) >= 2:
                return float(raw[0]), float(raw[1])
        if hasattr(vehicle, "last_current_action"):
            raw = vehicle.last_current_action
            if raw is not None and hasattr(raw, "__len__") and len(raw) >= 2:
                return float(raw[-2]), float(raw[-1])
    except Exception:
        pass
    return 0.0, 0.0


def _episode_end_reason(info):
    """Return a human-readable string for why the episode ended."""
    if info.get("crash_vehicle") or info.get("crash_object"):
        return "crash"
    if info.get("out_of_road"):
        return "out_of_road"
    if info.get("arrive_dest"):
        return "arrive"
    return "timeout"


# ── Brain recording (background thread) ──────────────────
def _run_brain_recording(output_path, stop_event):
    """Connect to a local private fNIRS device and stream data to CSV.

    Runs in a daemon thread with its own asyncio event loop.
    """
    import asyncio

    async def _record():
        FnirsClient = get_private_fnirs_client_class()

        async with FnirsClient() as device:
            print("[brain] Connected to local fNIRS device")

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp"] + config.FNIRS_CHANNEL_NAMES)
                count = 0

                def on_frame(_label, pkt):
                    nonlocal count
                    writer.writerow([
                        round(time.time(), 4),
                        pkt.ir_l, pkt.red_l, pkt.amb_l,
                        pkt.ir_r, pkt.red_r, pkt.amb_r,
                        pkt.ir_p, pkt.red_p, pkt.amb_p,
                        pkt.temp,
                    ])
                    count += 1
                    if count % 50 == 0:
                        f.flush()
                        print(f"  [brain] {count} frames recorded")

                device.on("frame", on_frame)
                await device.start_streaming()

                while not stop_event.is_set():
                    await asyncio.sleep(0.5)

                await device.stop_streaming()
                f.flush()

            print(f"[brain] Saved {count} frames to {output_path}")

    try:
        asyncio.run(_record())
    except Exception as e:
        print(f"[brain] Recording failed: {e}")


def record_session(seed=None, max_episodes=None, max_seconds=None, record_brain=True):
    """Run a human-play recording session with optional brain recording.

    Args:
        seed:          Starting scenario seed (default: time-based).
        max_episodes:  Stop after this many episodes (default: unlimited).
        max_seconds:   Stop after this many seconds (default: unlimited).
        record_brain:  If True, try to connect to a local private fNIRS device.

    Returns:
        dict with session summary.
    """
    if seed is None:
        seed = int(time.time() * 1000) % (2 ** 31)

    # ── Start brain recording in background thread ────────
    brain_stop = threading.Event()
    brain_thread = None
    if record_brain:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        brain_path = config.FNIRS_CSV
        print("[brain] Connecting to local fNIRS device...")
        brain_thread = threading.Thread(
            target=_run_brain_recording,
            args=(brain_path, brain_stop),
            daemon=True,
        )
        brain_thread.start()
        # Give BLE a moment to connect before opening MetaDrive window
        time.sleep(2)

    # ── Build environment ─────────────────────────────────
    env = MetaDriveEnv(_build_manual_env_config(seed=seed))

    # ── Psychedelic distraction overlay (initialized after first reset) ──
    overlay = None
    overlay_attempted = False

    # ── Prepare output file ───────────────────────────────
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.GAME_JSONL
    out_file = open(out_path, "w")

    # ── Session bookkeeping ───────────────────────────────
    session_start = time.time()
    total_records = 0
    episode_count = 0
    episode_rewards = []

    last_record_time = 0.0
    last_status_time = session_start

    # ── Graceful Ctrl+C handling ──────────────────────────
    shutdown_requested = False
    prev_handler = signal.getsignal(signal.SIGINT)

    def _on_sigint(_sig, _frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _on_sigint)

    print(f"[recorder] Session started (seed={seed})")
    print(f"[recorder] Recording to {out_path} at {RECORD_HZ} Hz")
    print("[recorder] Drive with arrow keys / WASD.  Ctrl+C to stop.")
    print()

    try:
        while True:
            if shutdown_requested:
                print("\n[recorder] Ctrl+C received, shutting down.")
                break
            if max_episodes is not None and episode_count >= max_episodes:
                print(f"[recorder] Reached {max_episodes} episodes, stopping.")
                break
            elapsed = time.time() - session_start
            if max_seconds is not None and elapsed >= max_seconds:
                print(f"[recorder] Reached {max_seconds}s, stopping.")
                break

            # ── Start episode ─────────────────────────────
            obs, info = env.reset()

            if not overlay_attempted:
                overlay_attempted = True
                try:
                    from psychedelic_overlay import PsychedelicOverlay
                    overlay = PsychedelicOverlay(env)
                    print("[recorder] Psychedelic distractions ENABLED")
                except Exception as exc:
                    print(f"[recorder] Overlay init failed ({exc}), continuing without effects")

            if overlay and not overlay._started:
                overlay.start()
            episode_count += 1
            episode_reward = 0.0
            episode_start = time.time()
            episode_steps = 0
            alive = True
            reward = 0.0

            print(f"[recorder] Episode {episode_count} started")

            while alive and not shutdown_requested:
                obs, reward, terminated, truncated, info = env.step([0.0, 0.0])
                episode_reward += reward
                episode_steps += 1
                alive = not (terminated or truncated)

                if overlay:
                    overlay.tick()

                now = time.time()

                if now - last_record_time >= RECORD_INTERVAL_SEC:
                    last_record_time = now

                    features = extract_features(obs)
                    steer, throttle = _get_human_action(env)
                    action_idx = _continuous_to_discrete(steer, throttle)

                    record = {
                        "t": round(now, 2),
                        "sec": round(now - session_start, 3),
                        "features": features,
                        "action": action_idx,
                        "alive": alive,
                        "reward": round(float(reward), 4),
                    }
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    total_records += 1

                if now - last_status_time >= STATUS_INTERVAL_SEC:
                    last_status_time = now
                    speed_kmh = info.get("velocity", 0) * 3.6
                    print(
                        f"  [status] ep={episode_count}  step={episode_steps}  "
                        f"time={now - episode_start:.0f}s  "
                        f"speed={speed_kmh:.1f}km/h  "
                        f"reward={episode_reward:.2f}  "
                        f"records={total_records}"
                    )

                if max_seconds is not None and (now - session_start) >= max_seconds:
                    break

            # ── Episode ended — write final record ────────
            now = time.time()
            features = extract_features(obs)
            steer, throttle = _get_human_action(env)
            action_idx = _continuous_to_discrete(steer, throttle)

            end_record = {
                "t": round(now, 2),
                "sec": round(now - session_start, 3),
                "features": features,
                "action": action_idx,
                "alive": False,
                "reward": round(float(reward), 4),
            }
            out_file.write(json.dumps(end_record) + "\n")
            out_file.flush()
            total_records += 1

            episode_rewards.append(episode_reward)
            reason = _episode_end_reason(info)
            ep_time = now - episode_start
            print(
                f"[recorder] Episode {episode_count} ended: "
                f"{reason}  reward={episode_reward:.2f}  "
                f"steps={episode_steps}  time={ep_time:.1f}s"
            )

    finally:
        if overlay:
            overlay.cleanup()
        out_file.close()
        env.close()
        signal.signal(signal.SIGINT, prev_handler)

        # Stop brain recording
        if brain_thread:
            print("[brain] Stopping...")
            brain_stop.set()
            brain_thread.join(timeout=5)

    # ── Summary ───────────────────────────────────────────
    total_elapsed = time.time() - session_start
    avg = round(float(np.mean(episode_rewards)), 2) if episode_rewards else 0.0

    summary = {
        "total_records": total_records,
        "episodes": episode_count,
        "elapsed_seconds": round(total_elapsed, 1),
        "episode_rewards": [round(r, 2) for r in episode_rewards],
        "avg_reward": avg,
        "output_file": str(out_path),
    }

    print()
    print("=" * 56)
    print("  MetaDrive Recording Session Summary")
    print("=" * 56)
    print(f"  Episodes:      {summary['episodes']}")
    print(f"  Total records: {summary['total_records']}")
    print(f"  Elapsed:       {summary['elapsed_seconds']}s")
    print(f"  Avg reward:    {summary['avg_reward']}")
    print(f"  Output:        {summary['output_file']}")
    print("=" * 56)

    return summary


if __name__ == "__main__":
    max_sec = None
    no_brain = "--no-brain" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-brain"]
    if args:
        max_sec = int(args[0])
    record_session(max_seconds=max_sec, record_brain=not no_brain)
