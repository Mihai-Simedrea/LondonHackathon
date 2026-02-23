# NeuroLabel Architecture Reference (AI-Facing)

## Purpose

This document describes the **current post-refactor architecture** of the NeuroLabel repo as of this snapshot. It is intended for an AI agent (or a human doing AI-assisted work) that needs:

- an accurate mental model of the codebase,
- the canonical module paths,
- the compatibility layer boundaries,
- the end-to-end data flow,
- the extension points,
- and the invariants needed to modify the repo safely.

This is the authoritative high-level map for the refactored code.

## Executive Summary

The repo has been refactored from a flat-script layout into a package-centric architecture built around `neurolabel/`.

### What is now canonical

- CLI: `scripts/neurolabel`
- CLI parser: `neurolabel/ui/cli/main.py`
- Runtime orchestration: `neurolabel/core/orchestration.py`
- Backend selection: `neurolabel/backends/registry.py`
- Data processing: `neurolabel/data/*`
- Simulation orchestration: `neurolabel/simulation/runner.py`
- Brain acquisition split: `neurolabel/brain/acquisition/*`
- Experiments package: `neurolabel/experiments/*`
- Velocity / MetaDrive backend modules: `neurolabel/backends/velocity/*`, `neurolabel/backends/metadrive/*`

### What remains as compatibility/legacy dependencies (intentional)

These still exist and are still used, but mostly behind package wrappers/bridges:

- `config.py` (legacy global config source; wrapped by `Settings` + `legacy_bridge`)
- `game_engine.py` (Velocity engine primitives; still imported by Velocity backend modules)
- `oc_scorer.py` (wrapped by `neurolabel/brain/scoring/oc.py`)

Most old top-level scripts (`pipeline.py`, `server.py`, `model.py`, `simulator.py`, etc.) are now **compatibility wrappers** that delegate to package modules.

## Design Goals Achieved

1. Centralized orchestration (package API + single CLI)
2. Centralized backend selection (registry)
3. Centralized device/scoring dispatch (brain registry + scorer wrapper)
4. Backend-explicit dataset building (no payload sniffing in package path)
5. Versioned result serialization (`schema_version`)
6. Monolith split for acquisition (`server.py` → `brain/acquisition/*`)
7. Canonical package implementations for Velocity + MetaDrive backend modules
8. Compatibility preserved via thin top-level wrappers

## Current Top-Level Layout (What Matters)

### Canonical code

- `neurolabel/`
- `scripts/neurolabel`
- `docs/HUMAN_GUIDE.md`
- `docs/AI_SYSTEM_REFERENCE.md`

### Compatibility wrappers (examples)

- `pipeline.py` → package CLI
- `server.py` → brain acquisition CLI
- `model.py` → `neurolabel.backends.velocity.model`
- `simulator.py` → `neurolabel.backends.velocity.simulation`
- `synthetic_data.py` → `neurolabel.backends.velocity.synthetic`
- `visualizer.py` → `neurolabel.ui.replay.velocity_viewer`
- `metadrive_*.py` → `neurolabel.backends.metadrive.*`
- `model_compare.py`, `model_tuning.py`, `model_torch.py` → `neurolabel.experiments.*`
- `parallel_runner.py` → `neurolabel.simulation.runner`

## Package Map (Canonical)

### `neurolabel/config`

#### `neurolabel/config/schema.py`
Defines immutable runtime configuration dataclasses:

- `Paths`
  - `project_dir`, `data_dir`, `models_dir`, `results_dir`
  - artifact file paths (`eeg_csv`, `fnirs_csv`, `game_jsonl`, `oc_scores_csv`, `dataset_dirty`, `dataset_clean`, `model_dirty`, `model_clean`, `results_dirty`, `results_clean`)
  - `ensure_dirs()` creates only the main directories
- `Settings`
  - `backend`
  - `device_mode`
  - `paths`
  - `oc_cutoff`, `sims_per_model`, `sim_workers`, `batch_size`
  - `with_overrides(...)`
  - `brain_csv` property (switches on `device_mode`)

#### `neurolabel/config/defaults.py`
Bridges from the legacy `config.py` globals into `Settings`.

- `from_legacy_config()` returns `Settings`
- This is the main reason `config.py` can still exist without blocking the new architecture

#### `neurolabel/config/loader.py`
Runtime loader used by CLI.

- `load_settings(backend=None, device_mode=None)`
- Applies overrides and calls `settings.paths.ensure_dirs()`

## `neurolabel/core`

### `neurolabel/core/contracts.py`
Defines simple summary DTOs used across orchestration and CLI output:

- `CollectSummary`
- `ProcessSummary`
- `TrainSummary`
- `SimSummary`
- `DemoSummary`

### `neurolabel/core/orchestration.py`
Package-level orchestration entrypoints. These are the main high-level operations used by the CLI.

Functions:

- `collect(settings)`
- `generate_synthetic(settings, duration_seconds)`
- `process(settings)`
- `train(settings)`
- `simulate(settings)`
- `visualize(settings, dirty_results=None, clean_results=None)`
- `demo(settings, synthetic=False, dev=False, duration_seconds=600)`

Important behavior:

- Calls `apply_legacy_settings(settings)` before delegating to backend/legacy-sensitive modules.
- `process()` reads game start timestamp and trims brain scoring via `read_game_start_timestamp(...)`.
- `process()` now uses package-native dataset build/filter (`neurolabel/data/*`).
- `visualize()` is backend-aware and raises cleanly if unsupported.

### `neurolabel/core/legacy_bridge.py`
Critical compatibility bridge.

- Synchronizes `Settings` into legacy `config.py` module globals
- This allows package orchestration to remain authoritative while legacy modules continue to run unchanged
- This is the main compatibility mechanism for the transition period

### `neurolabel/core/results.py`
Versioned JSON result I/O.

- `SCHEMA_VERSION = 1`
- `save_result_json(path, payload)` injects `schema_version`
- `load_result_json(path)` backfills `schema_version=0` for older summaries

### `neurolabel/core/doctor.py`
Environment/dependency checks.

Checks include:

- backend validity
- device mode validity
- `pygame`, `numpy`, `scipy`, `sklearn`, `joblib`, `bleak`, `metadrive`, `torch`, `catboost`, `xgboost`
- existence of output directories

## `neurolabel/backends`

### Shared registry and protocol

#### `neurolabel/backends/base.py`
Protocol `BackendAdapter` defines the expected backend interface:

- `name`
- `supports_visualizer`
- `collect_live(settings)`
- `generate_synthetic(settings, duration_seconds=...)`
- `train_models(settings)`
- `simulate_comparison(settings)`
- `visualize(settings, dirty_results=..., clean_results=...)`

#### `neurolabel/backends/registry.py`
Central backend selection.

- `load_backend(name)`
- `available_backends()`

Current registered backends:

- `velocity`
- `metadrive`

### Velocity backend (`neurolabel/backends/velocity`)

#### Canonical modules

- `adapter.py` — backend adapter implementation
- `model.py` — sklearn model training/predict/persistence (package-native copy)
- `simulation.py` — headless game simulation (package-native copy)
- `synthetic.py` — synthetic EEG/fNIRS + game generation (package-native copy)
- `recording.py` — interactive game recorder UI (package-native copy)
- `app_game.py` — playable Velocity game UI (package-native copy)
- `render.py` — shared rendering primitives (de-dup across game/recorder/UI)
- `dataset_parser.py` — Velocity JSONL record → dataset row conversion
- `replay_adapter.py` — replay compatibility helper (currently trivial)

#### `VelocityBackend` (`adapter.py`)
Behavior:

- `collect_live()` currently shells out to `server.py --record` to preserve end-user behavior
- `generate_synthetic()` calls `neurolabel.backends.velocity.synthetic.generate_synthetic`
- `train_models()` trains dirty + clean models and saves both
- `simulate_comparison()` calls `neurolabel.simulation.runner.run_comparison`
- `visualize()` calls `neurolabel.ui.replay.velocity_viewer.run_from_results`

### MetaDrive backend (`neurolabel/backends/metadrive`)

#### Canonical modules

- `adapter.py`
- `env.py` (canonical replacement for `metadrive_wrapper.py`)
- `model.py` (canonical replacement for `metadrive_model.py`)
- `simulation.py` (canonical replacement for `metadrive_simulator.py`)
- `synthetic.py` (canonical replacement for `metadrive_synthetic.py`)
- `recording.py` (canonical replacement for `metadrive_recorder.py`)
- `action_space.py` (shared continuous→discrete quantization helpers)
- `dataset_parser.py`
- `replay_adapter.py` (currently `supports_replay() == False`)

#### `MetaDriveBackend` (`adapter.py`)
Behavior:

- `collect_live()` calls `neurolabel.backends.metadrive.recording.record_session()`
- `generate_synthetic()` calls `generate_synthetic_metadrive(...)`
- `train_models()` trains/saves dirty+clean models using MetaDrive feature schema
- `simulate_comparison()` uses common package simulation runner
- `visualize()` intentionally raises unsupported replay error

## `neurolabel/brain`

### Registry / protocol

#### `neurolabel/brain/base.py`
Defines `BrainScorer` protocol with `compute_scores(...)`.

#### `neurolabel/brain/registry.py`
Currently returns a single scorer wrapper (`OcScorer`) regardless of device mode because `oc_scorer.py` auto-detects EEG vs fNIRS from CSV header.

### Scoring

#### `neurolabel/brain/scoring/oc.py`
Wrapper over legacy `oc_scorer.compute_oc_scores(...)`.

Why it still wraps legacy:

- `oc_scorer.py` is large and domain-heavy; wrapping preserves behavior while isolating it behind a package contract.
- The rest of the package no longer depends on the internals of `oc_scorer.py`.

Inputs:

- `settings.brain_csv`
- output path `settings.paths.oc_scores_csv`
- optional `trim_before`
- `include_timestamp_in_csv` (enabled in the package pipeline)

### Acquisition (`neurolabel/brain/acquisition`)

This is the split of the old `server.py` monolith.

#### `main.py`
CLI for acquisition workflows (server replacement logic).

- `--record` mode selects the game recorder script based on backend
  - Velocity → `data_recorder.py`
  - MetaDrive → `metadrive_recorder.py`
- non-`--record` mode uses `car_game.py` and saves to `~/Medusa_Recordings`
- dispatches by `config.DEVICE_MODE`

#### `eeg_ble.py`
EEG BLE path (extracted from `server.py`):

- BLE discovery (`find_headset`)
- characteristic selection (`find_notify_characteristic`)
- packet parsing + channel conversion (`adc_to_microvolts`)
- background buffering via `RecordBuffer`
- CSV output via `save_buffer_to_csv`
- optional game process lifecycle integration

#### `fnirs_device.py`
fNIRS acquisition path (optional local private provider):

- provider client stream subscription
- sample buffering and CSV sink
- optional game process lifecycle integration

#### `session.py`
Shared process/session helpers:

- `launch_game(...)`
- `terminate_game(...)`
- `monitor_connection(...)`

#### `sinks.py`
Shared recording buffer and sink logic:

- `RecordBuffer`
- `SaveConfig`
- `save_buffer_to_csv(...)`

This split is one of the biggest architectural simplifications in the repo.

## `neurolabel/data`

### `neurolabel/data/schemas.py`
Defines backend-specific dataset field names.

- `VELOCITY_FIELDNAMES = ["lane", "obs_d0", "obs_d1", "obs_d2", "decision", "oc_score"]`
- `METADRIVE_SUFFIX_FIELDS = ["action", "oc_score"]`

### `neurolabel/data/dataset_builder.py`
Package-native dataset builder (replaces payload sniffing path in the new architecture).

Key behavior:

- explicit backend parser dispatch using `settings.backend`
- OC lookup loaded from CSV into:
  - timestamp-indexed arrays (`oc_ts_keys`, `oc_ts_values`)
  - second-indexed dict (`oc_by_sec`)
- timestamps are sorted before `bisect`
- timestamp match tolerance is effectively `2.5s`
- fallback OC assignment is `0.5` when no match is found
- warns about unmatched rows and corrupted JSONL lines

### `neurolabel/data/dataset_filter.py`
Backend-agnostic filter/split step.

- reads full dataset CSV
- writes:
  - dirty (all rows)
  - clean (`oc_score >= cutoff`)
- reports filtered percentages

### Backend parsers

- `neurolabel/backends/velocity/dataset_parser.py`
  - reconstructs nearest obstacle distances from `obs` list
- `neurolabel/backends/metadrive/dataset_parser.py`
  - expands `record["features"]` + `action` + `oc_score`

## `neurolabel/simulation`

### `neurolabel/simulation/runner.py`
Now package-native (no longer a wrapper).

Capabilities:

- multiprocessing batch simulation
- worker loads model once per seed batch (important performance fix)
- backend-specific load/predict/simulate function dispatch via `settings.backend`
- dirty vs clean comparison flow
- versioned summary saving via `neurolabel.core.results.save_result_json`

Key functions:

- `run_simulations(settings, model_path, n_sims=None, batch_size=None)`
- `run_comparison(settings)`

Saved summary schema (v1 wrapper):

- `schema_version`
- `backend`
- `label` (`dirty` / `clean`)
- `avg_alive`, `std_alive`, `min_alive`, `max_alive`
- `avg_reward`, `std_reward`
- `avg_route_completion`
- `alive_times`, `rewards`, `route_completions`

Note:

- Full replay frames are returned at runtime in `results["runs"]` but not persisted into the summary JSON (to keep files small).

## `neurolabel/ui`

### CLI

#### `neurolabel/ui/cli/main.py`
Defines CLI parser and subcommands.

Top-level commands:

- `collect`
- `process`
- `train`
- `simulate`
- `visualize`
- `demo`
- `doctor`
- `report`
- `experiment`

Common options for most commands:

- `--backend {velocity,metadrive}`
- `--device {eeg,fnirs}`

`demo` options:

- `--synthetic`
- `--dev`
- `--duration <int>`

#### `neurolabel/ui/cli/commands.py`
CLI function handlers.

Responsibilities:

- load `Settings`
- call orchestration
- print user-facing summaries
- handle visualization unsupported path gracefully

### Replay

#### `neurolabel/ui/replay/velocity_viewer.py`
Canonical Velocity replay visualizer implementation (migrated from `visualizer.py`) plus package helper:

- `run_visualizer(dirty_results, clean_results)`
- `generate_mock_results(...)`
- `run_from_results(settings, dirty_results=None, clean_results=None)`

`run_from_results(...)` behavior:

- uses in-memory results if provided
- otherwise loads versioned summaries from `settings.paths.results_dirty/clean`
- creates mock replay runs sized to match run counts (legacy visualizer requires frame data)
- overlays summary metrics (`avg_alive`, `std_alive`) onto mock payloads

This is a compatibility strategy until a fully schema-aware replay viewer exists.

#### `neurolabel/ui/replay/serialization.py`
Wrapper over versioned result JSON helpers.

#### `neurolabel/ui/replay/dto.py`
Defines simple replay DTOs:

- `ReplayRun`
- `ReplayBundle`

## `neurolabel/models`

### `neurolabel/models/velocity_data.py`
Shared Velocity dataset loading/feature conversion helpers used by experiments and models.

Includes helpers for:

- raw `X/y` loading
- engineered feature loading
- mirrored augmentation

This removed repeated CSV parsing and feature engineering code across experiments.

### `neurolabel/models/features.py`
Reserved/shared feature helpers module (exists as part of package organization; use this for future central feature engineering expansion).

## `neurolabel/experiments`

### Canonical experiment modules

- `neurolabel/experiments/compare.py`
- `neurolabel/experiments/tune.py`
- `neurolabel/experiments/torch_train.py`
- `neurolabel/experiments/commands.py`

### `neurolabel/experiments/commands.py`
CLI-facing wrappers that import package experiment modules (not top-level scripts anymore).

Subcommands available under `neurolabel experiment`:

- `compare`
- `tune`
- `torch-train`

### Important compatibility note

`neurolabel/experiments/tune.py` still intentionally uses some legacy modules (`dataset.py`, `oc_scorer.py`, `game_engine.py`) to preserve historical experiment behavior and signatures. This does not affect the main pipeline architecture.

## End-to-End Data Flow (Canonical Pipeline)

### Synthetic or live collection → process → train → simulate → report/visualize

1. CLI (`neurolabel/ui/cli/main.py`) parses command
2. CLI handler (`neurolabel/ui/cli/commands.py`) loads `Settings`
3. Orchestration (`neurolabel/core/orchestration.py`) applies `legacy_bridge`
4. Backend selected once via `neurolabel/backends/registry.py`
5. Data collection (live or synthetic) happens in backend module
6. Brain OC scoring via `neurolabel/brain/scoring/oc.py`
7. Dataset built via `neurolabel/data/dataset_builder.py`
8. Dataset filtered via `neurolabel/data/dataset_filter.py`
9. Models trained via backend model module
10. Simulations run via `neurolabel/simulation/runner.py`
11. Results saved with versioned schema (`neurolabel/core/results.py`)
12. `report` reads summaries; `visualize` delegates to backend UI support

## Compatibility Layer Strategy

This repo intentionally uses a compatibility layer instead of a hard break.

### Pattern used

- New canonical implementation lives in `neurolabel/...`
- Old top-level module becomes a thin wrapper that imports from package

Benefits:

- Existing scripts and user habits keep working
- Tests importing legacy module names keep passing
- New development gets a stable package API

### Examples

- `model.py` → `neurolabel.backends.velocity.model`
- `visualizer.py` → `neurolabel.ui.replay.velocity_viewer`
- `metadrive_wrapper.py` → `neurolabel.backends.metadrive.env`
- `parallel_runner.py` → `neurolabel.simulation.runner`

## Remaining Legacy Dependencies (Known + Intentional)

These are the main remaining non-package dependencies that the package still imports directly:

1. `config.py`
   - still used as legacy config source and mutable runtime bridge target
   - package wraps it with `Settings` + `legacy_bridge`
2. `game_engine.py`
   - still used by Velocity backend modules and parsers
   - stable, core logic module
3. `oc_scorer.py`
   - wrapped by `neurolabel/brain/scoring/oc.py`
   - domain-heavy and kept intact for behavior preservation

These are acceptable compatibility anchors at the current stage.

## Migration Map (Old → Canonical New)

### Core / CLI

- `pipeline.py` → `neurolabel.ui.cli.main`
- `server.py` → `neurolabel.brain.acquisition.main`

### Velocity

- `model.py` → `neurolabel.backends.velocity.model`
- `simulator.py` → `neurolabel.backends.velocity.simulation`
- `synthetic_data.py` → `neurolabel.backends.velocity.synthetic`
- `data_recorder.py` → `neurolabel.backends.velocity.recording`
- `car_game.py` → `neurolabel.backends.velocity.app_game`
- `visualizer.py` → `neurolabel.ui.replay.velocity_viewer`

### MetaDrive

- `metadrive_wrapper.py` → `neurolabel.backends.metadrive.env`
- `metadrive_model.py` → `neurolabel.backends.metadrive.model`
- `metadrive_simulator.py` → `neurolabel.backends.metadrive.simulation`
- `metadrive_synthetic.py` → `neurolabel.backends.metadrive.synthetic`
- `metadrive_recorder.py` → `neurolabel.backends.metadrive.recording`

### Simulation / Results

- `parallel_runner.py` → `neurolabel.simulation.runner`

### Experiments

- `model_compare.py` → `neurolabel.experiments.compare`
- `model_tuning.py` → `neurolabel.experiments.tune`
- `model_torch.py` → `neurolabel.experiments.torch_train`

## Invariants / Assumptions (Important for Future Refactors)

1. `Settings` is the package-level runtime truth.
2. `legacy_bridge.apply_legacy_settings(settings)` must run before calling legacy-sensitive modules.
3. Backend selection should happen once (via registry), not via scattered `if config.GAME_BACKEND` branches in orchestration.
4. Package data processing should be backend-explicit (no payload sniffing).
5. Result JSONs should include `schema_version`.
6. MetaDrive visualization should fail gracefully until a proper viewer exists.
7. Compatibility wrappers should stay thin (no business logic).

## How to Extend the System Safely

### Add a new backend

1. Create `neurolabel/backends/<name>/adapter.py`
2. Implement the `BackendAdapter` protocol methods
3. Add dataset parser (`dataset_parser.py`) if training schema differs
4. Add model/simulation/synthetic/recording modules
5. Register in `neurolabel/backends/registry.py`
6. If replay supported, add UI/replay adapter path

### Add a new brain scoring mode

1. Create scorer wrapper under `neurolabel/brain/scoring/`
2. Update `neurolabel/brain/registry.py`
3. Keep `compute_scores(...)` contract compatible with orchestration
4. Ensure OC score CSV fields remain compatible with dataset builder (or extend builder explicitly)

### Add new CLI command

1. Add parser entry in `neurolabel/ui/cli/main.py`
2. Implement handler in `neurolabel/ui/cli/commands.py`
3. Prefer calling orchestration/package APIs, not top-level wrappers

## Testing / Validation Expectations

Current baseline validation used during refactor:

- `python3 -m py_compile $(rg --files -g'*.py' | tr '\n' ' ')`
- `pytest -q` (currently `36 passed`)
- CLI smoke:
  - `python3 scripts/neurolabel --help`
  - `python3 pipeline.py --help`
  - `python3 server.py --help`

### Hardware/manual checks (not CI)

- EEG BLE connect/discover/notify/record/stop
- fNIRS connect/stream/record/stop (optional local private provider)
- game process lifecycle coupling in acquisition sessions

## Known Limitations (Current State)

1. `config.py` remains a legacy global source and mutation target.
2. `game_engine.py` remains outside the package and is imported directly by Velocity modules.
3. `oc_scorer.py` remains a wrapped legacy module (not yet split into EEG/fNIRS package submodules).
4. MetaDrive replay visualization is intentionally unsupported.
5. `VelocityBackend.collect_live()` still shells out to `server.py --record` for compatibility (instead of calling acquisition package APIs directly with explicit backend/device/session objects).
6. Some experiment modules intentionally retain legacy imports for historical behavior parity.

## Recommended Next Refactor Targets (If Continuing)

1. Move `game_engine.py` into `neurolabel/backends/velocity/engine.py` and wrap the top-level file.
2. Replace `config.py` globals with a package-native config source and keep `config.py` as a read-only compatibility shim.
3. Split `oc_scorer.py` into `neurolabel/brain/scoring/eeg.py` and `neurolabel/brain/scoring/fnirs.py` behind the same wrapper contract.
4. Make `VelocityBackend.collect_live()` call acquisition package APIs directly (no subprocess shell-out to `server.py`).
5. Replace mock replay reconstruction in `run_from_results()` with persisted replay bundles or a schema-aware viewer.

## Practical Guidance for AI Agents Working in This Repo

1. Prefer `neurolabel/...` modules over top-level scripts.
2. Treat top-level scripts as wrappers unless explicitly asked to preserve CLI behavior.
3. Before changing runtime behavior, inspect whether `legacy_bridge` synchronization is needed.
4. When adding backend-specific behavior, implement it in backend modules and wire it through the backend adapter/registry, not orchestration conditionals.
5. Use `doctor` and `report` for quick environment/state inspection before deeper debugging.
6. Validate with `pytest -q` after touching shared modules (`orchestration`, `simulation`, `dataset`, `model`, wrappers).

---

If this document and the code diverge, update this file first and treat the mismatch as an architecture regression signal.
