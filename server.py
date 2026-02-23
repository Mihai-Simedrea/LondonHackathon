#!/usr/bin/env python3
"""Compatibility facade for legacy brain recorder entrypoint.

The implementation has moved to `neurolabel.brain.acquisition.*`. This module
preserves the historical `server.py --record` CLI and exports coroutine names
`record_eeg()` / `record_fnirs()` used by transitional adapters.
"""

from __future__ import annotations

from pathlib import Path

import config

# Legacy mutable globals kept for compatibility with the old module contract.
SAVE_FOLDER = Path.home() / "Medusa_Recordings"
GAME_SCRIPT = Path(__file__).parent / "car_game.py"
FIXED_FILENAME: str | None = None


async def record_eeg():
    from neurolabel.brain.acquisition.eeg_ble import record_eeg_session

    return await record_eeg_session(
        save_folder=SAVE_FOLDER,
        fixed_filename=FIXED_FILENAME,
        game_script=GAME_SCRIPT,
    )


async def record_fnirs():
    from neurolabel.brain.acquisition.fnirs_device import record_fnirs_session

    return await record_fnirs_session(
        save_folder=SAVE_FOLDER,
        fixed_filename=FIXED_FILENAME,
        game_script=GAME_SCRIPT,
    )


def main(argv=None):
    # Delegate CLI behavior to the package entrypoint (which still honors
    # config.GAME_BACKEND / config.DEVICE_MODE during migration).
    from neurolabel.brain.acquisition.main import main as acquisition_main

    return acquisition_main(argv)


if __name__ == "__main__":
    main()
