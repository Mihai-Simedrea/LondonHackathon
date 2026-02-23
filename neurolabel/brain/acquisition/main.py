from __future__ import annotations

"""CLI entrypoint for brain-acquisition flows (server.py replacement)."""

import argparse
import asyncio
from pathlib import Path

import config


def _resolve_record_mode_settings() -> tuple[Path, Path, str | None]:
    save_folder = Path(__file__).resolve().parents[3] / "data"
    save_folder.mkdir(exist_ok=True)

    project_dir = Path(__file__).resolve().parents[3]
    if config.GAME_BACKEND == "metadrive":
        game_script = project_dir / "metadrive_recorder.py"
    else:
        game_script = project_dir / "data_recorder.py"

    fixed_filename = "fnirs_recording.csv" if config.DEVICE_MODE == "fnirs" else "eeg_recording.csv"
    return save_folder, game_script, fixed_filename


async def record_eeg(*, save_folder: Path, game_script: Path, fixed_filename: str | None):
    from neurolabel.brain.acquisition.eeg_ble import record_eeg_session

    return await record_eeg_session(save_folder=save_folder, game_script=game_script, fixed_filename=fixed_filename)


async def record_fnirs(*, save_folder: Path, game_script: Path, fixed_filename: str | None):
    from neurolabel.brain.acquisition.fnirs_device import record_fnirs_session

    return await record_fnirs_session(save_folder=save_folder, game_script=game_script, fixed_filename=fixed_filename)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Recording mode for NeuroLabel pipeline")
    args = parser.parse_args(argv)

    if args.record:
        save_folder, game_script, fixed_filename = _resolve_record_mode_settings()
    else:
        project_dir = Path(__file__).resolve().parents[3]
        save_folder = Path.home() / "Medusa_Recordings"
        game_script = project_dir / "car_game.py"
        fixed_filename = None

    try:
        if config.DEVICE_MODE == "fnirs":
            asyncio.run(record_fnirs(save_folder=save_folder, game_script=game_script, fixed_filename=fixed_filename))
        else:
            asyncio.run(record_eeg(save_folder=save_folder, game_script=game_script, fixed_filename=fixed_filename))
    except KeyboardInterrupt:
        print("\n👋 Exiting...")


if __name__ == "__main__":
    main()
