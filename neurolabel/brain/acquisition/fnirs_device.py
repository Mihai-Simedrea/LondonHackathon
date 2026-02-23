from __future__ import annotations

"""fNIRS acquisition flow using an optional private local headset provider."""

import asyncio
import time
from pathlib import Path

from neurolabel.brain.acquisition.fnirs_provider import get_private_fnirs_client_class
from neurolabel.brain.acquisition.session import launch_game, monitor_connection, terminate_game
from neurolabel.brain.acquisition.sinks import RecordBuffer, SaveConfig, save_buffer_to_csv


async def record_fnirs_session(
    *,
    save_folder: Path,
    fixed_filename: str | None = None,
    game_script: Path | None = None,
) -> dict:
    buffer = RecordBuffer()
    save_cfg = SaveConfig(save_folder=save_folder, fixed_filename=fixed_filename, device_mode="fnirs")

    print("🔍 Scanning for local fNIRS device...")
    print("Make sure the headset is powered on and your private provider is configured.\n")

    game_process = None
    saved_path = None
    try:
        FnirsClient = get_private_fnirs_client_class()
        async with FnirsClient() as device:
            print("✅ Connected to local fNIRS device")
            print(f"📁 Save location: {save_folder}\n")

            def fnirs_callback(_label: str, pkt) -> None:
                buffer.append([
                    time.time(),
                    pkt.ir_l, pkt.red_l, pkt.amb_l,
                    pkt.ir_r, pkt.red_r, pkt.amb_r,
                    pkt.ir_p, pkt.red_p, pkt.amb_p,
                    pkt.temp,
                ])
                if buffer.packet_count % 50 == 0:
                    print(
                        f"📊 Recording: {buffer.packet_count} frames | "
                        f"{buffer.elapsed_seconds:.1f}s | IR_L: {pkt.ir_l}"
                    )

            device.on("frame", fnirs_callback)
            await device.start_streaming()

            game_process = launch_game(game_script) if game_script else None
            print("Press Ctrl+C to stop recording and save\n")
            reason = await monitor_connection(lambda: device.is_connected, game_process)
            if reason in {"game_closed", "keyboard_interrupt", "disconnect"}:
                try:
                    await device.stop_streaming()
                except Exception:
                    pass
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return {"ok": False, "error": str(exc), "samples": buffer.packet_count}
    finally:
        terminate_game(game_process)
        saved_path = save_buffer_to_csv(buffer, save_cfg)

    return {"ok": True, "samples": buffer.packet_count, "path": str(saved_path) if saved_path else None}


def run_record_fnirs(**kwargs) -> dict:
    return asyncio.run(record_fnirs_session(**kwargs))
