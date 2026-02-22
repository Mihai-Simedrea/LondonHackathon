from __future__ import annotations

import argparse

from neurolabel.ui.cli import commands
from neurolabel.backends.registry import available_backends


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NeuroLabel")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parent = argparse.ArgumentParser(add_help=False)
    common_parent.add_argument("--backend", choices=available_backends(), default=None)
    common_parent.add_argument("--device", choices=["eeg", "fnirs"], default=None)

    sub = subparsers.add_parser("collect", parents=[common_parent], help="Record brain + game data")
    sub.set_defaults(func=commands.cmd_collect)

    sub = subparsers.add_parser("process", parents=[common_parent], help="Compute OC and build dataset")
    sub.set_defaults(func=commands.cmd_process)

    sub = subparsers.add_parser("train", parents=[common_parent], help="Train models")
    sub.set_defaults(func=commands.cmd_train)

    sub = subparsers.add_parser("simulate", parents=[common_parent], help="Run simulations")
    sub.set_defaults(func=commands.cmd_simulate)

    sub = subparsers.add_parser("visualize", parents=[common_parent], help="Visualize results")
    sub.set_defaults(func=commands.cmd_visualize)

    sub = subparsers.add_parser("demo", parents=[common_parent], help="Run end-to-end demo")
    sub.add_argument("--synthetic", action="store_true")
    sub.add_argument("--dev", action="store_true")
    sub.add_argument("--duration", type=int, default=600)
    sub.set_defaults(func=commands.cmd_demo)

    sub = subparsers.add_parser("doctor", parents=[common_parent], help="Check environment/dependencies")
    sub.set_defaults(func=commands.cmd_doctor)

    sub = subparsers.add_parser("report", parents=[common_parent], help="Print saved simulation summaries")
    sub.set_defaults(func=commands.cmd_report)

    exp = subparsers.add_parser("experiment", help="Run experiment workflows")
    exp_subparsers = exp.add_subparsers(dest="experiment_command", required=True)

    sub = exp_subparsers.add_parser("compare", help="Run model comparison experiment")
    sub.set_defaults(func=commands.cmd_experiment_compare)

    sub = exp_subparsers.add_parser("tune", help="Run hyperparameter tuning experiment")
    sub.set_defaults(func=commands.cmd_experiment_tune)

    sub = exp_subparsers.add_parser("torch-train", help="Train torch model on clean dataset")
    sub.set_defaults(func=commands.cmd_experiment_torch_train)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
