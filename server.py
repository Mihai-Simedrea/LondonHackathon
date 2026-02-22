#!/usr/bin/env python3
"""
Lightweight EEG / fNIRS Recorder
Connects to BLE headset, discovers EEG service automatically, launches the car game,
and saves data to CSV in microvolts (EEG) or raw counts (fNIRS).
"""

import asyncio
import struct
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from bleak import BleakClient, BleakScanner

import config
from mendi.ble_client import MendiClient

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

DEVICE_KEYWORDS = ["symbiote", "bci", "medusa"]

TOTAL_CHANNELS = 20
PACKET_HEADER = 0xAA
PACKET_FOOTER = 0x55
EXPECTED_PACKET_SIZE = 84

# ADC to microvolts conversion
VREF = 2.5
GAIN = 32
MAX_ADC = 2**31

# Channel names
CHANNEL_NAMES = [
    "Fp1", "Fp2", "Fpz", "Cp1", "-", "-", "T7", "-", "O1", "Fz",
    "O2", "Cp2", "T8", "-", "Oz", "P3", "P4", "P7", "Cz", "P8"
]

# fNIRS channel names
FNIRS_CHANNEL_NAMES = [
    "ir_l", "red_l", "amb_l", "ir_r", "red_r", "amb_r",
    "ir_p", "red_p", "amb_p", "temp"
]

# Save location
SAVE_FOLDER = Path.home() / "Medusa_Recordings"
SAVE_FOLDER.mkdir(exist_ok=True)

# Path to the car game — same folder as this script by default
GAME_SCRIPT = Path(__file__).parent / "car_game.py"

# -------------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------------
all_data = []
packet_count = 0
start_time = None
FIXED_FILENAME = None

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
def adc_to_microvolts(adc_value: int) -> float:
    """Convert ADC reading to microvolts"""
    return (adc_value / MAX_ADC * VREF / GAIN) * 1e6


async def find_headset():
    """Scan for and select EEG headset"""
    print("🔍 Scanning for BLE devices...")
    print("Make sure your EEG headset is powered ON\n")

    devices = await BleakScanner.discover(timeout=10.0)

    if not devices:
        print("❌ No BLE devices found")
        return None

    # Try to find device by keyword
    matched_devices = []
    for device in devices:
        name = (device.name or "").lower()
        if any(keyword in name for keyword in DEVICE_KEYWORDS):
            matched_devices.append(device)

    if matched_devices:
        print(f"✅ Found {len(matched_devices)} matching device(s):\n")
        for i, device in enumerate(matched_devices, 1):
            print(f"[{i}] {device.name or 'Unknown'}")
            print(f"    Address: {device.address}\n")

        if len(matched_devices) == 1:
            print(f"→ Auto-selecting: {matched_devices[0].name}\n")
            return matched_devices[0].address
        else:
            while True:
                try:
                    choice = input(f"Select device [1-{len(matched_devices)}]: ")
                    idx = int(choice) - 1
                    if 0 <= idx < len(matched_devices):
                        return matched_devices[idx].address
                except (ValueError, KeyboardInterrupt):
                    print("❌ Invalid selection")
                    return None

    print(f"⚠️  No devices matched keywords: {DEVICE_KEYWORDS}")
    print(f"Showing all {len(devices)} device(s):\n")

    for i, device in enumerate(devices, 1):
        print(f"[{i}] {device.name or 'Unknown'}")
        print(f"    Address: {device.address}\n")

    while True:
        try:
            choice = input(f"Select device [1-{len(devices)}] or 'q' to quit: ")
            if choice.lower() == 'q':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(devices):
                return devices[idx].address
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid selection")
            return None


# -------------------------------------------------------------------------
# Game Launcher
# -------------------------------------------------------------------------
def launch_game() -> Optional[subprocess.Popen]:
    """Launch car_game.py as a separate process."""
    if not GAME_SCRIPT.exists():
        print(f"⚠️  Game script not found at: {GAME_SCRIPT}")
        print("    Make sure car_game.py is in the same folder as this script.")
        return None

    print(f"🎮 Launching game: {GAME_SCRIPT}\n")
    # Use the same Python interpreter that's running this script
    process = subprocess.Popen([sys.executable, str(GAME_SCRIPT)])
    return process


# -------------------------------------------------------------------------
# Service Discovery
# -------------------------------------------------------------------------
async def find_notify_characteristic(client: BleakClient) -> Optional[str]:
    """Walk all services and return UUID of first notifiable characteristic."""
    print("\n📋 Discovering services...\n")

    notify_chars = []

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

    # Filter out standard BLE service characteristics (battery, device info, etc.)
    # Standard 16-bit UUIDs follow pattern: 0000xxxx-0000-1000-8000-00805f9b34fb
    STANDARD_BLE_SUFFIX = "-0000-1000-8000-00805f9b34fb"
    custom_chars = [uuid for uuid in notify_chars if not uuid.lower().endswith(STANDARD_BLE_SUFFIX)]

    if len(custom_chars) == 1:
        print(f"→ Auto-selecting EEG characteristic: {custom_chars[0]}")
        print(f"  (skipped {len(notify_chars) - 1} standard BLE characteristic(s))\n")
        return custom_chars[0]

    # Fall back to manual selection
    select_from = custom_chars if custom_chars else notify_chars
    print("Found multiple notifiable characteristics:\n")
    for i, uuid in enumerate(select_from, 1):
        print(f"  [{i}] {uuid}")
    print()

    while True:
        try:
            choice = input(f"Select characteristic [1-{len(select_from)}]: ")
            idx = int(choice) - 1
            if 0 <= idx < len(select_from):
                return select_from[idx]
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid selection")
            return None


# -------------------------------------------------------------------------
# BLE Callback
# -------------------------------------------------------------------------
async def ble_callback(sender, data: bytearray):
    """Process incoming BLE data packets"""
    global packet_count, start_time

    if start_time is None:
        start_time = datetime.now()

    if len(data) != EXPECTED_PACKET_SIZE:
        return
    if data[0] != PACKET_HEADER or data[82] != PACKET_FOOTER:
        return

    try:
        channels = struct.unpack('<20i', data[2:82])
    except struct.error:
        return

    voltages = [adc_to_microvolts(ch) for ch in channels]
    all_data.append([time.time()] + voltages)
    packet_count += 1

    if packet_count % 250 == 0:
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"📊 Recording: {packet_count} packets | {elapsed:.1f}s | CH0: {voltages[0]:.2f} µV")


# -------------------------------------------------------------------------
# Save to CSV
# -------------------------------------------------------------------------
def save_to_csv():
    """Save all recorded data to CSV file"""
    if not all_data:
        print("❌ No data to save")
        return

    if FIXED_FILENAME:
        filepath = SAVE_FOLDER / FIXED_FILENAME
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config.DEVICE_MODE == "fnirs":
            filepath = SAVE_FOLDER / f"fnirs-recording-{timestamp}.csv"
        else:
            filepath = SAVE_FOLDER / f"eeg-recording-{timestamp}.csv"

    # Choose header based on device mode
    if config.DEVICE_MODE == "fnirs":
        header = FNIRS_CHANNEL_NAMES
    else:
        header = CHANNEL_NAMES

    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + header)
        for sample in all_data:
            writer.writerow(sample)

    elapsed = (datetime.now() - start_time).total_seconds() if start_time else 0
    n_channels = len(header)
    print(f"\n✅ Saved: {filepath}")
    print(f"📈 Total samples: {len(all_data):,}")
    print(f"⏱️  Duration: {elapsed:.1f}s")
    print(f"📊 Channels: {n_channels}")


# -------------------------------------------------------------------------
# Main Recording Function
# -------------------------------------------------------------------------
async def record_eeg():
    """Connect to BLE device, launch game, and record EEG data"""

    device_address = await find_headset()

    if not device_address:
        print("❌ No device selected")
        return

    print(f"🔄 Connecting to: {device_address}")

    try:
        async with BleakClient(device_address, timeout=20.0) as client:
            if not client.is_connected:
                print("❌ Failed to connect")
                return

            print(f"✅ Connected to device")
            print(f"📁 Save location: {SAVE_FOLDER}\n")

            data_char_uuid = await find_notify_characteristic(client)

            if not data_char_uuid:
                print("❌ Could not find a usable characteristic")
                return

            # ── Launch the game now that we're connected ──
            game_process = launch_game()

            print("Press Ctrl+C to stop recording and save\n")

            await client.start_notify(data_char_uuid, ble_callback)

            try:
                while client.is_connected:
                    # Also stop if the game window was closed
                    if game_process and game_process.poll() is not None:
                        print("\n🎮 Game closed — stopping recording...")
                        break
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping recording...")
                await client.stop_notify(data_char_uuid)

            # Clean up game process if still running
            if game_process and game_process.poll() is None:
                game_process.terminate()

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        save_to_csv()


# -------------------------------------------------------------------------
# fNIRS Recording Function
# -------------------------------------------------------------------------
async def record_fnirs():
    """Connect to Mendi fNIRS headband, launch game, and record fNIRS data"""
    global packet_count, start_time

    print("🔍 Scanning for Mendi fNIRS headband...")
    print("Make sure your Mendi is powered ON\n")

    try:
        async with MendiClient() as mendi:
            print(f"✅ Connected to Mendi")
            print(f"📁 Save location: {SAVE_FOLDER}\n")

            def fnirs_callback(label: str, pkt):
                global packet_count, start_time
                if start_time is None:
                    start_time = datetime.now()

                all_data.append([
                    time.time(),
                    pkt.ir_l, pkt.red_l, pkt.amb_l,
                    pkt.ir_r, pkt.red_r, pkt.amb_r,
                    pkt.ir_p, pkt.red_p, pkt.amb_p,
                    pkt.temp,
                ])
                packet_count += 1

                if packet_count % 50 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"📊 Recording: {packet_count} frames | {elapsed:.1f}s | IR_L: {pkt.ir_l}")

            mendi.on("frame", fnirs_callback)
            await mendi.start_streaming()

            # ── Launch the game now that we're connected ──
            game_process = launch_game()

            print("Press Ctrl+C to stop recording and save\n")

            try:
                while mendi.is_connected:
                    # Also stop if the game window was closed
                    if game_process and game_process.poll() is not None:
                        print("\n🎮 Game closed — stopping recording...")
                        break
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n\n⏹️  Stopping recording...")

            await mendi.stop_streaming()

            # Clean up game process if still running
            if game_process and game_process.poll() is None:
                game_process.terminate()

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        save_to_csv()


# -------------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", action="store_true", help="Recording mode for NeuroLabel pipeline")
    args = parser.parse_args()

    if args.record:
        SAVE_FOLDER = Path(__file__).parent / "data"
        SAVE_FOLDER.mkdir(exist_ok=True)
        GAME_SCRIPT = Path(__file__).parent / "data_recorder.py"

        if config.DEVICE_MODE == "fnirs":
            FIXED_FILENAME = "fnirs_recording.csv"
        else:
            FIXED_FILENAME = "eeg_recording.csv"

    try:
        if config.DEVICE_MODE == "fnirs":
            asyncio.run(record_fnirs())
        else:
            asyncio.run(record_eeg())
    except KeyboardInterrupt:
        print("\n👋 Exiting...")