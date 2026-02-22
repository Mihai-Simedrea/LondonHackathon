"""Shared configuration for NeuroLabel pipeline."""
from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

# Create directories
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)

# EEG settings
SAMPLE_RATE = 250
FRONTAL_CHANNELS = [0, 1, 2, 9]  # Fp1, Fp2, Fpz, Fz
CHANNEL_NAMES = [
    "Fp1", "Fp2", "Fpz", "Cp1", "-", "-", "T7", "-", "O1", "Fz",
    "O2", "Cp2", "T8", "-", "Oz", "P3", "P4", "P7", "Cz", "P8"
]

# OC score settings
OC_CUTOFF = 0.6
BASELINE_SECONDS = 120
WINDOW_SECONDS = 4
STRIDE_SECONDS = 1

# Game settings (matching car_game.py)
LANE_COUNT = 3
FPS = 60
HEIGHT = 720
WIDTH = 480
ROAD_SPEED = 7.0
SPAWN_INTERVAL = 55

# Simulation settings
import multiprocessing as _mp
SIMS_PER_MODEL = 50
SIM_WORKERS = max(2, _mp.cpu_count() - 4)
BATCH_SIZE = 10

# File paths
EEG_CSV = DATA_DIR / "eeg_recording.csv"
GAME_JSONL = DATA_DIR / "game_recording.jsonl"
OC_SCORES_CSV = DATA_DIR / "oc_scores.csv"
DATASET_DIRTY = DATA_DIR / "dataset_dirty.csv"
DATASET_CLEAN = DATA_DIR / "dataset_clean.csv"
MODEL_DIRTY = MODELS_DIR / "model_dirty.joblib"
MODEL_CLEAN = MODELS_DIR / "model_clean.joblib"
RESULTS_DIRTY = RESULTS_DIR / "results_dirty.json"
RESULTS_CLEAN = RESULTS_DIR / "results_clean.json"
