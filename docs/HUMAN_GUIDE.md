# NeuroLabel (Human Guide)

## What This Repo Is Now

This repo now has a **clean package-based architecture** with a single main CLI:

- Canonical code lives under `neurolabel/`
- Old top-level scripts (like `pipeline.py`, `server.py`, `model.py`) are now mostly **compatibility wrappers**
- You can still use old commands, but new work should use the `neurolabel` CLI

## Main CLI (Recommended)

Use:

```bash
python3 scripts/neurolabel --help
```

Main commands:

- `collect` — record brain + game data
- `process` — compute OC scores and build/filter dataset
- `train` — train dirty + clean models
- `simulate` — run comparison simulations
- `visualize` — replay results (Velocity only)
- `demo` — end-to-end pipeline
- `doctor` — dependency/environment checks
- `report` — print saved result summaries
- `experiment` — compare/tune/torch experiments

## Typical Workflows

### 1. Full synthetic demo (fastest way to test everything)

```bash
python3 scripts/neurolabel demo --synthetic --dev
```

- `--synthetic` generates fake brain + game data
- `--dev` skips visualization (useful on headless machines)

### 2. Step-by-step pipeline

```bash
python3 scripts/neurolabel collect
python3 scripts/neurolabel process
python3 scripts/neurolabel train
python3 scripts/neurolabel simulate
python3 scripts/neurolabel report
python3 scripts/neurolabel visualize
```

### 3. Switch backend or device

```bash
python3 scripts/neurolabel demo --synthetic --backend velocity --device eeg
python3 scripts/neurolabel demo --synthetic --backend metadrive --device fnirs --dev
```

## Backends / Device Support

### Backends

- `velocity` — fully supported (train/simulate/visualize)
- `metadrive` — train/simulate supported, visualization intentionally disabled for now

### Devices

- `eeg` — BLE headset recording path supported
- `fnirs` — recording via optional local private integration (public repo ships mock/synthetic paths)

## Where Outputs Go

Defaults come from `config.py`, but the new runtime wraps them in `Settings`.

Common outputs:

- Brain CSV: `data/eeg_recording.csv` or `data/fnirs_recording.csv`
- Game recording: `data/game_recording.jsonl`
- OC scores: `data/oc_scores.csv`
- Datasets: `data/dataset_dirty.csv`, `data/dataset_clean.csv`
- Models: `models/model_dirty.joblib`, `models/model_clean.joblib`
- Results: `results/results_dirty.json`, `results/results_clean.json`

## Quick Troubleshooting

### Check environment/deps

```bash
python3 scripts/neurolabel doctor
```

This reports whether modules like `pygame`, `bleak`, `metadrive`, `torch`, `catboost`, and `xgboost` are installed.

### Visualization fails on MetaDrive

Expected for now. Replay visualization is currently Velocity-only.

## Compatibility Notes

Old commands still work (wrappers):

- `python3 pipeline.py ...`
- `python3 server.py --record`
- `python3 model_compare.py`
- `python3 model_tuning.py`
- `python3 model_torch.py`

They now delegate to the package implementation.

## If You’re Editing the Code

Start here:

- Orchestration: `neurolabel/core/orchestration.py`
- CLI: `neurolabel/ui/cli/main.py`
- Backend registry: `neurolabel/backends/registry.py`
- Dataset build/filter: `neurolabel/data/dataset_builder.py`, `neurolabel/data/dataset_filter.py`
- Simulation runner: `neurolabel/simulation/runner.py`

For a full architecture reference, see `docs/AI_SYSTEM_REFERENCE.md`.
