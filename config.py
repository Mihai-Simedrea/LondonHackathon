"""Shared configuration for NeuroLabel pipeline."""
from pathlib import Path
import multiprocessing as _mp

from game_engine import HEIGHT, WIDTH, FPS, LANE_COUNT, ROAD_SPEED, SPAWN_INTERVAL

# Directories
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Create directories
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# ── Device mode: "fnirs" or "eeg" ──
DEVICE_MODE = "fnirs"

# ── Game backend: "velocity" or "metadrive" ──
GAME_BACKEND = "metadrive"

# EEG settings
SAMPLE_RATE = 250
FRONTAL_CHANNELS = [0, 1, 2, 9]  # Fp1, Fp2, Fpz, Fz
CHANNEL_NAMES = [
    "Fp1", "Fp2", "Fpz", "Cp1", "-", "-", "T7", "-", "O1", "Fz",
    "O2", "Cp2", "T8", "-", "Oz", "P3", "P4", "P7", "Cz", "P8"
]

# fNIRS settings
FNIRS_SAMPLE_RATE = 11
FNIRS_BASELINE_SEC = 30
FNIRS_CHANNEL_NAMES = [
    "ir_l", "red_l", "amb_l", "ir_r", "red_r", "amb_r",
    "ir_p", "red_p", "amb_p", "temp"
]

# OC score settings
OC_CUTOFF = 0.6
BASELINE_SECONDS = 120
WINDOW_SECONDS = 4
STRIDE_SECONDS = 1

# Game settings come from the canonical headless engine constants to avoid drift.

# MetaDrive settings
METADRIVE_MAP = 7
METADRIVE_TRAFFIC_DENSITY = 0.5
METADRIVE_NUM_SCENARIOS = 100
METADRIVE_HORIZON = 1000
METADRIVE_STEERING_DIM = 5
METADRIVE_THROTTLE_DIM = 3

# Simulation settings
SIMS_PER_MODEL = 20
SIM_WORKERS = max(2, _mp.cpu_count() - 2)
BATCH_SIZE = 10

# File paths
EEG_CSV = DATA_DIR / "eeg_recording.csv"
FNIRS_CSV = DATA_DIR / "fnirs_recording.csv"
BRAIN_CSV = FNIRS_CSV if DEVICE_MODE == "fnirs" else EEG_CSV
GAME_JSONL = DATA_DIR / "game_recording.jsonl"
OC_SCORES_CSV = DATA_DIR / "oc_scores.csv"
DATASET_DIRTY = DATA_DIR / "dataset_dirty.csv"
DATASET_CLEAN = DATA_DIR / "dataset_clean.csv"
MODEL_DIRTY = MODELS_DIR / "model_dirty.joblib"
MODEL_CLEAN = MODELS_DIR / "model_clean.joblib"
RESULTS_DIRTY = RESULTS_DIR / "results_dirty.json"
RESULTS_CLEAN = RESULTS_DIR / "results_clean.json"
