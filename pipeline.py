#!/usr/bin/env python3
"""NeuroLabel Pipeline — EEG-guided game AI training demo."""

import argparse
import sys
import time

import config


def cmd_collect(args):
    """Record EEG + game data from live session."""
    import subprocess
    print("Starting EEG recording + game...")
    print("Press Ctrl+C to stop recording.\n")
    result = subprocess.run([sys.executable, "server.py", "--record"])
    if result.returncode != 0:
        print(f"\nCollection failed (exit code {result.returncode}). Aborting.")
        sys.exit(1)


def cmd_process(args):
    """Process brain data (EEG or fNIRS) and build datasets."""
    print(f"\n--- PROCESSING ({config.DEVICE_MODE.upper()} mode) ---")

    # Read game start time so we can trim idle data before gameplay
    import json as _json
    trim_before = None
    if config.GAME_JSONL.exists():
        with open(config.GAME_JSONL, 'r') as _f:
            first_line = _f.readline().strip()
            if first_line:
                trim_before = _json.loads(first_line).get('t')
                if trim_before:
                    print(f"\n  Game start timestamp: {trim_before:.2f}")

    # Step 1: Compute OC scores from brain data
    print("\n[1/3] Computing OC scores...")
    from oc_scorer import compute_oc_scores
    oc_results = compute_oc_scores(
        str(config.BRAIN_CSV),
        output_path=str(config.OC_SCORES_CSV),
        trim_before=trim_before,
    )
    print(f"  Computed {len(oc_results)} OC scores")

    # Step 2: Build merged dataset
    print("\n[2/3] Building dataset...")
    from dataset import build_dataset, filter_dataset
    full_path = build_dataset(
        str(config.GAME_JSONL),
        str(config.OC_SCORES_CSV)
    )

    # Step 3: Filter into dirty/clean
    print("\n[3/3] Splitting dirty/clean datasets...")
    dirty_path, clean_path = filter_dataset(str(full_path), cutoff=config.OC_CUTOFF)

    print("\nProcessing complete!")


def cmd_train(args):
    """Train dirty and clean models."""
    print("\n--- TRAINING ---")
    from model import train_model, save_model

    print("\n[1/2] Training DIRTY model (all data)...")
    dirty_model = train_model(str(config.DATASET_DIRTY))
    save_model(dirty_model, str(config.MODEL_DIRTY))

    print("\n[2/2] Training CLEAN model (OC-filtered)...")
    clean_model = train_model(str(config.DATASET_CLEAN))
    save_model(clean_model, str(config.MODEL_CLEAN))

    print("\nTraining complete!")


def cmd_simulate(args):
    """Run 50 simulations per model."""
    print("\n--- SIMULATION ---")
    from parallel_runner import run_comparison
    dirty_results, clean_results = run_comparison(
        str(config.MODEL_DIRTY),
        str(config.MODEL_CLEAN)
    )
    return dirty_results, clean_results


def cmd_visualize(args):
    """Open side-by-side comparison visualization."""
    import json
    from visualizer import run_visualizer

    # Load results summaries
    with open(config.RESULTS_DIRTY, 'r') as f:
        dirty_summary = json.load(f)
    with open(config.RESULTS_CLEAN, 'r') as f:
        clean_summary = json.load(f)

    # The saved JSON summaries don't include frame-level replay data.
    # Generate mock replay runs using the visualizer's built-in generator,
    # but preserve the real aggregate statistics.
    from visualizer import generate_mock_results
    dirty_mock = generate_mock_results(num_runs=len(dirty_summary.get("alive_times", [5])))
    clean_mock = generate_mock_results(num_runs=len(clean_summary.get("alive_times", [5])))

    # Overlay real stats onto mock replay data
    dirty_mock["avg_alive"] = dirty_summary["avg_alive"]
    dirty_mock["std_alive"] = dirty_summary["std_alive"]
    clean_mock["avg_alive"] = clean_summary["avg_alive"]
    clean_mock["std_alive"] = clean_summary["std_alive"]

    run_visualizer(dirty_mock, clean_mock)


def cmd_demo(args):
    """Run full demo pipeline."""
    start = time.time()

    print("=" * 60)
    if args.synthetic:
        print("  NEUROLABEL DEMO (Synthetic Data)")
    else:
        print("  NEUROLABEL DEMO (Live EEG)")
    print("=" * 60)

    if args.synthetic:
        # Step 0: Generate synthetic data
        step_start = time.time()
        print("\n[0/5] Generating synthetic data...")
        from synthetic_data import generate_synthetic
        generate_synthetic(duration_seconds=600)
        print(f"  Step time: {time.time() - step_start:.1f}s")
    else:
        # Step 0: Collect live data
        print("\n[0/5] Collecting live EEG + game data...")
        cmd_collect(args)

    # Step 1: Process
    step_start = time.time()
    print("\n[1/5] Processing EEG data...")
    cmd_process(args)
    print(f"  Step time: {time.time() - step_start:.1f}s")

    # Check for insufficient clean data
    import csv as _csv
    with open(config.DATASET_CLEAN, 'r') as _f:
        clean_count = sum(1 for _ in _csv.reader(_f)) - 1  # minus header
    if clean_count < 20:
        print(f"\n  ⚠ Only {clean_count} clean rows — model will be unreliable.")
        print("    Try recording for at least 60 seconds for usable results.")

    # Step 2: Train
    step_start = time.time()
    print("\n[2/5] Training models...")
    cmd_train(args)
    print(f"  Step time: {time.time() - step_start:.1f}s")

    # Step 3: Simulate
    step_start = time.time()
    print("\n[3/5] Running simulations...")
    dirty_results, clean_results = cmd_simulate(args)
    print(f"  Step time: {time.time() - step_start:.1f}s")

    # Step 4: Summary
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("  NEUROLABEL DEMO — RESULTS")
    print("=" * 60)
    print(f"\n  DIRTY MODEL (fat):  avg = {dirty_results['avg_alive']:.0f} frames ({dirty_results['avg_alive']/60:.1f}s)")
    print(f"  CLEAN MODEL (slim): avg = {clean_results['avg_alive']:.0f} frames ({clean_results['avg_alive']/60:.1f}s)")
    if dirty_results['avg_alive'] > 0:
        improvement = ((clean_results['avg_alive'] - dirty_results['avg_alive']) / dirty_results['avg_alive']) * 100
        print(f"\n  Improvement: {improvement:+.1f}%")
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60)

    # Step 5: Visualize (pass full in-memory results with frame data)
    if not getattr(args, 'dev', False):
        print("\n[5/5] Opening visualization...")
        from visualizer import run_visualizer
        run_visualizer(dirty_results, clean_results)
    else:
        print("\n[5/5] Skipping visualization (--dev mode)")


def main():
    parser = argparse.ArgumentParser(description="NeuroLabel Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    sub = subparsers.add_parser("collect", help="Record EEG + game data")
    sub.set_defaults(func=cmd_collect)

    sub = subparsers.add_parser("process", help="Process EEG and build datasets")
    sub.set_defaults(func=cmd_process)

    sub = subparsers.add_parser("train", help="Train models")
    sub.set_defaults(func=cmd_train)

    sub = subparsers.add_parser("simulate", help="Run simulations")
    sub.set_defaults(func=cmd_simulate)

    sub = subparsers.add_parser("visualize", help="Open visualization")
    sub.set_defaults(func=cmd_visualize)

    sub = subparsers.add_parser("demo", help="Run full demo pipeline")
    sub.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    sub.add_argument("--dev", action="store_true", help="Skip visualization (dev mode)")
    sub.set_defaults(func=cmd_demo)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
