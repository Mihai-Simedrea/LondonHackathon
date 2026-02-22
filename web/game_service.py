"""MetaDrive game streaming service — runs MetaDrive in a separate PROCESS,
captures frames as JPEG, accepts keyboard input from the browser.

Uses multiprocessing because MetaDrive/Panda3D requires the main thread
for signal handling."""

from __future__ import annotations

import io
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

# ── Settings ────────────────────────────────────────────────────────
STEER_CENTER = config.METADRIVE_STEERING_DIM // 2
THROTTLE_CENTER = config.METADRIVE_THROTTLE_DIM // 2
STEERING_DIM = config.METADRIVE_STEERING_DIM

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
JPEG_QUALITY = 70
TARGET_FPS = 25


def _keys_to_action(keys: dict) -> int:
    """Map browser keyboard state to a discrete MetaDrive action index."""
    left = keys.get("left", False)
    right = keys.get("right", False)
    up = keys.get("up", False)
    down = keys.get("down", False)

    if left and not right:
        steer_idx = config.METADRIVE_STEERING_DIM - 1  # MetaDrive: +1 = left
    elif right and not left:
        steer_idx = 0  # MetaDrive: -1 = right
    else:
        steer_idx = STEER_CENTER

    if up and not down:
        throttle_idx = config.METADRIVE_THROTTLE_DIM - 1
    elif down and not up:
        throttle_idx = 0
    else:
        throttle_idx = THROTTLE_CENTER

    # MetaDrive decodes: steer = action % steering_dim, throttle = action // steering_dim
    return throttle_idx * STEERING_DIM + steer_idx


def _game_process(
    frame_queue: mp.Queue,
    action_value: mp.Value,
    stop_event: mp.Event,
    ready_event: mp.Event,
    session_seconds: float,
):
    """MetaDrive game loop — runs in a separate process."""
    from metadrive import MetaDriveEnv
    from psychedelic_overlay import PsychedelicOverlay

    env_config = {
        "use_render": True,
        "show_interface": False,
        "show_logo": False,
        "show_fps": False,
        "image_observation": True,
        "stack_size": 1,
        "window_size": (FRAME_WIDTH, FRAME_HEIGHT),
        "discrete_action": True,
        "discrete_steering_dim": config.METADRIVE_STEERING_DIM,
        "discrete_throttle_dim": config.METADRIVE_THROTTLE_DIM,
        "num_scenarios": config.METADRIVE_NUM_SCENARIOS,
        "start_seed": int(time.time()) % (2**31),
        "traffic_density": config.METADRIVE_TRAFFIC_DENSITY,
        "map": config.METADRIVE_MAP,
        "horizon": config.METADRIVE_HORIZON,
        "crash_vehicle_done": True,
        "out_of_road_done": True,
        "log_level": 50,
        "manual_control": False,
        "sensors": {
            "main_camera": (),
        },
        "vehicle_config": {
            "image_source": "main_camera",
        },
    }

    try:
        env = MetaDriveEnv(env_config)
        obs, info = env.reset()

        # Psychedelic distraction overlays
        overlay = None
        try:
            overlay = PsychedelicOverlay(env)
            overlay.start()
            print("[game] Psychedelic overlay ENABLED", flush=True)
        except Exception as exc:
            print(f"[game] Overlay init failed ({exc}), continuing without", flush=True)

        ready_event.set()

        default_action = THROTTLE_CENTER * STEERING_DIM + STEER_CENTER
        session_start = time.time()
        frame_interval = 1.0 / TARGET_FPS

        while not stop_event.is_set():
            loop_start = time.time()

            if loop_start - session_start >= session_seconds:
                break

            # Get action from shared value
            action = action_value.value
            if action != default_action:
                print(f"[game] action={action}", flush=True)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)

            # Tick psychedelic overlays
            if overlay:
                overlay.tick()

            # Encode frame
            img = obs["image"]
            if img.ndim == 4:
                img = img[:, :, :, -1]
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_uint8)
            buf = io.BytesIO()
            pil_img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            jpeg_bytes = buf.getvalue()

            # Put frame in queue (drop old frames if queue is full)
            try:
                # Clear old frames
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        break
                frame_queue.put_nowait(jpeg_bytes)
            except Exception:
                pass

            # Reset on episode end
            if terminated or truncated:
                obs, info = env.reset()

            # Frame rate limit
            elapsed = time.time() - loop_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

    except Exception as e:
        print(f"[game_service] Error: {e}", flush=True)
    finally:
        if overlay:
            try:
                overlay.cleanup()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass


class GameStreamService:
    """Manages MetaDrive in a separate process, exposes frames and input."""

    def __init__(self, session_seconds: float = 45.0):
        self.session_seconds = session_seconds
        self.alive = False
        self._process: mp.Process | None = None
        # Use spawn context — fork crashes on macOS when CoreFoundation is loaded
        ctx = mp.get_context("spawn")
        self._frame_queue: mp.Queue = ctx.Queue(maxsize=2)
        self._action_value = ctx.Value("i", _keys_to_action({}))
        self._stop_event = ctx.Event()
        self._ready_event = ctx.Event()
        self._ctx = ctx

    def start(self):
        """Launch MetaDrive in a separate process."""
        self._process = self._ctx.Process(
            target=_game_process,
            args=(
                self._frame_queue,
                self._action_value,
                self._stop_event,
                self._ready_event,
                self.session_seconds,
            ),
            daemon=True,
        )
        self._process.start()
        # Wait for environment to be ready
        if self._ready_event.wait(timeout=45.0):
            self.alive = True
        else:
            self.alive = False
            self.stop()

    def get_frame(self) -> bytes | None:
        """Return the latest JPEG frame."""
        frame = None
        try:
            while not self._frame_queue.empty():
                frame = self._frame_queue.get_nowait()
        except Exception:
            pass
        return frame

    def set_keys(self, keys: dict):
        """Update the current action from browser keyboard state."""
        self._action_value.value = _keys_to_action(keys)

    def stop(self):
        """Stop the game process."""
        self._stop_event.set()
        if self._process is not None and self._process.is_alive():
            self._process.join(timeout=5.0)
            if self._process.is_alive():
                self._process.terminate()
        self.alive = False
