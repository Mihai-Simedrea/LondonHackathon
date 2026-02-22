from __future__ import annotations

import time
from pathlib import Path

from neurolabel.config.loader import load_settings
from neurolabel.core import orchestration
from neurolabel.core.doctor import run_doctor
from neurolabel.experiments import commands as experiment_commands
from neurolabel.ui.replay.serialization import load_summary


def cmd_collect(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    orchestration.collect(settings)


def cmd_process(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    summary = orchestration.process(settings)
    print(f"Computed {summary.oc_windows} OC windows")


def cmd_train(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    orchestration.train(settings)


def cmd_simulate(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    orchestration.simulate(settings)


def cmd_visualize(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    try:
        orchestration.visualize(settings)
    except RuntimeError as exc:
        print(f"[visualize] {exc}")


def cmd_demo(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    start = time.time()
    summary = orchestration.demo(
        settings,
        synthetic=args.synthetic,
        dev=args.dev,
        duration_seconds=args.duration,
    )
    dirty = summary.sim.dirty_results
    clean = summary.sim.clean_results
    print("\n" + "=" * 60)
    print("  NEUROLABEL DEMO — RESULTS")
    print("=" * 60)
    print(f"DIRTY avg_alive: {dirty.get('avg_alive', 0):.2f}")
    print(f"CLEAN avg_alive: {clean.get('avg_alive', 0):.2f}")
    print(f"Elapsed: {time.time() - start:.1f}s")


def cmd_doctor(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    checks = run_doctor(settings)
    ok_count = 0
    for chk in checks:
        mark = "OK" if chk.ok else "FAIL"
        print(f"[{mark}] {chk.name}: {chk.detail}")
        ok_count += int(chk.ok)
    print(f"\n{ok_count}/{len(checks)} checks passing")


def cmd_report(args):
    settings = load_settings(backend=args.backend, device_mode=args.device)
    targets = [
        ("dirty", settings.paths.results_dirty),
        ("clean", settings.paths.results_clean),
    ]
    for label, path in targets:
        path = Path(path)
        if not path.exists():
            print(f"[report] Missing {label} results: {path}")
            continue
        data = load_summary(path)
        print(f"\n{label.upper()} ({path})")
        print(f"  schema_version: {data.get('schema_version', 'n/a')}")
        print(f"  backend: {data.get('backend', settings.backend)}")
        for key in ("avg_alive", "std_alive", "avg_reward", "std_reward", "avg_route_completion"):
            if key in data:
                print(f"  {key}: {data[key]}")


def cmd_experiment_compare(args):
    raise SystemExit(experiment_commands.run_compare())


def cmd_experiment_tune(args):
    raise SystemExit(experiment_commands.run_tune())


def cmd_experiment_torch_train(args):
    raise SystemExit(experiment_commands.run_torch_train())
