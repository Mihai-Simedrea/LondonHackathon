from __future__ import annotations

"""Shared process/session helpers for acquisition flows."""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Callable


class KeyboardStop(Exception):
    """Raised when a recording session is interrupted by Ctrl+C."""


def launch_game(game_script: Path) -> subprocess.Popen | None:
    """Launch the configured game/recorder UI script."""
    if not game_script.exists():
        print(f"⚠️  Game script not found at: {game_script}")
        return None

    print(f"🎮 Launching game: {game_script}\n")
    return subprocess.Popen([sys.executable, str(game_script)])


def terminate_game(game_process: subprocess.Popen | None) -> None:
    if game_process and game_process.poll() is None:
        game_process.terminate()


async def monitor_connection(
    is_connected: Callable[[], bool],
    game_process: subprocess.Popen | None,
    *,
    poll_interval: float = 1.0,
) -> str:
    """Wait until connection ends, game closes, or Ctrl+C is pressed.

    Returns a reason string: "disconnect", "game_closed", or "keyboard_interrupt".
    """
    try:
        while is_connected():
            if game_process and game_process.poll() is not None:
                print("\n🎮 Game closed — stopping recording...")
                return "game_closed"
            await asyncio.sleep(poll_interval)
        return "disconnect"
    except KeyboardInterrupt:
        print("\n\n⏹️  Stopping recording...")
        return "keyboard_interrupt"
