from __future__ import annotations

"""EEG BLE acquisition flow (extracted from legacy server.py)."""

import asyncio
import struct
import time
from pathlib import Path
from typing import Optional

try:
    from bleak import BleakClient, BleakScanner
except Exception:  # pragma: no cover - depends on environment
    BleakClient = None  # type: ignore[assignment]
    BleakScanner = None  # type: ignore[assignment]

from neurolabel.brain.acquisition.session import launch_game, monitor_connection, terminate_game
from neurolabel.brain.acquisition.sinks import RecordBuffer, SaveConfig, save_buffer_to_csv

DEVICE_KEYWORDS = ["symbiote", "bci", "medusa"]

PACKET_HEADER = 0xAA
PACKET_FOOTER = 0x55
EXPECTED_PACKET_SIZE = 84

VREF = 2.5
GAIN = 32
MAX_ADC = 2**31


class MissingBleakDependency(RuntimeError):
    pass


def adc_to_microvolts(adc_value: int) -> float:
    return (adc_value / MAX_ADC * VREF / GAIN) * 1e6


async def find_headset(device_keywords: list[str] | None = None) -> str | None:
    if BleakScanner is None:
        raise MissingBleakDependency("bleak is not installed; EEG BLE recording unavailable")

    keywords = device_keywords or DEVICE_KEYWORDS
    print("🔍 Scanning for BLE devices...")
    print("Make sure your EEG headset is powered ON\n")

    devices = await BleakScanner.discover(timeout=10.0)
    if not devices:
        print("❌ No BLE devices found")
        return None

    matched_devices = [d for d in devices if any(k in (d.name or "").lower() for k in keywords)]
    if matched_devices:
        print(f"✅ Found {len(matched_devices)} matching device(s):\n")
        for i, device in enumerate(matched_devices, 1):
            print(f"[{i}] {device.name or 'Unknown'}")
            print(f"    Address: {device.address}\n")
        if len(matched_devices) == 1:
            print(f"→ Auto-selecting: {matched_devices[0].name}\n")
            return matched_devices[0].address
        try:
            choice = input(f"Select device [1-{len(matched_devices)}]: ")
            idx = int(choice) - 1
            return matched_devices[idx].address if 0 <= idx < len(matched_devices) else None
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid selection")
            return None

    print(f"⚠️  No devices matched keywords: {keywords}")
    print(f"Showing all {len(devices)} device(s):\n")
    for i, device in enumerate(devices, 1):
        print(f"[{i}] {device.name or 'Unknown'}")
        print(f"    Address: {device.address}\n")

    try:
        choice = input(f"Select device [1-{len(devices)}] or 'q' to quit: ")
        if choice.lower() == "q":
            return None
        idx = int(choice) - 1
        return devices[idx].address if 0 <= idx < len(devices) else None
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid selection")
        return None


async def find_notify_characteristic(client: "BleakClient") -> Optional[str]:
    print("\n📋 Discovering services...\n")
    notify_chars: list[str] = []

    for service in client.services:
        print(f"  Service: {service.uuid}")
        if service.description and service.description != "Unknown":
            print(f"           {service.description}")
        for char in service.characteristics:
            props = ", ".join(char.properties)
            desc = f" — {char.description}" if char.description and char.description != "Unknown" else ""
            print(f"    Char:  {char.uuid}  [{props}]{desc}")
            if "notify" in char.properties or "indicate" in char.properties:
                notify_chars.append(char.uuid)
        print()

    if not notify_chars:
        print("⚠️  No notifiable characteristics found on this device.")
        return None
    if len(notify_chars) == 1:
        print(f"→ Auto-selecting characteristic: {notify_chars[0]}\n")
        return notify_chars[0]

    standard_suffix = "-0000-1000-8000-00805f9b34fb"
    custom_chars = [u for u in notify_chars if not u.lower().endswith(standard_suffix)]
    if len(custom_chars) == 1:
        print(f"→ Auto-selecting EEG characteristic: {custom_chars[0]}")
        print(f"  (skipped {len(notify_chars) - 1} standard BLE characteristic(s))\n")
        return custom_chars[0]

    select_from = custom_chars if custom_chars else notify_chars
    print("Found multiple notifiable characteristics:\n")
    for i, uuid in enumerate(select_from, 1):
        print(f"  [{i}] {uuid}")
    print()
    try:
        choice = input(f"Select characteristic [1-{len(select_from)}]: ")
        idx = int(choice) - 1
        return select_from[idx] if 0 <= idx < len(select_from) else None
    except (ValueError, KeyboardInterrupt):
        print("❌ Invalid selection")
        return None


async def record_eeg_session(
    *,
    save_folder: Path,
    fixed_filename: str | None = None,
    game_script: Path | None = None,
    device_keywords: list[str] | None = None,
) -> dict:
    if BleakClient is None:
        raise MissingBleakDependency("bleak is not installed; EEG BLE recording unavailable")

    buffer = RecordBuffer()
    save_cfg = SaveConfig(save_folder=save_folder, fixed_filename=fixed_filename, device_mode="eeg")

    device_address = await find_headset(device_keywords)
    if not device_address:
        print("❌ No device selected")
        return {"ok": False, "samples": 0}

    print(f"🔄 Connecting to: {device_address}")

    def handle_packet(_sender, data: bytearray) -> None:
        if len(data) != EXPECTED_PACKET_SIZE:
            return
        if data[0] != PACKET_HEADER or data[82] != PACKET_FOOTER:
            return
        try:
            channels = struct.unpack("<20i", data[2:82])
        except struct.error:
            return
        voltages = [adc_to_microvolts(ch) for ch in channels]
        buffer.append([time.time()] + voltages)
        if buffer.packet_count % 250 == 0:
            print(
                f"📊 Recording: {buffer.packet_count} packets | "
                f"{buffer.elapsed_seconds:.1f}s | CH0: {voltages[0]:.2f} µV"
            )

    game_process = None
    saved_path = None
    try:
        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                print("❌ Failed to connect")
                return {"ok": False, "samples": 0}

            print("✅ Connected to device")
            print(f"📁 Save location: {save_folder}\n")

            data_char_uuid = await find_notify_characteristic(client)
            if not data_char_uuid:
                print("❌ Could not find a usable characteristic")
                return {"ok": False, "samples": 0}

            game_process = launch_game(game_script) if game_script else None

            print("Press Ctrl+C to stop recording and save\n")
            await client.start_notify(data_char_uuid, handle_packet)
            reason = await monitor_connection(lambda: client.is_connected, game_process)
            if reason == "keyboard_interrupt":
                try:
                    await client.stop_notify(data_char_uuid)
                except Exception:
                    pass
    except Exception as exc:
        print(f"❌ Error: {exc}")
        return {"ok": False, "error": str(exc), "samples": buffer.packet_count}
    finally:
        terminate_game(game_process)
        saved_path = save_buffer_to_csv(buffer, save_cfg)

    return {"ok": True, "samples": buffer.packet_count, "path": str(saved_path) if saved_path else None}


def run_record_eeg(**kwargs) -> dict:
    return asyncio.run(record_eeg_session(**kwargs))
