from __future__ import annotations

import config


def run_compare() -> int:
    try:
        from neurolabel.experiments import compare
    except ImportError as exc:
        print(f"[experiment compare] Import failed: {exc}")
        return 1
    compare.main()
    return 0


def run_tune() -> int:
    try:
        from neurolabel.experiments import tune
    except ImportError as exc:
        print(f"[experiment tune] Import failed: {exc}")
        return 1
    tune.main()
    return 0


def run_torch_train() -> int:
    try:
        import torch  # noqa: F401
        from neurolabel.experiments import torch_train
    except ImportError as exc:
        print(f"[experiment torch-train] Import failed: {exc}")
        return 1

    print(f"PyTorch {torch_train.torch.__version__} | Device: {torch_train.DEVICE}")
    if config.DATASET_CLEAN.exists():
        print("\nTraining on clean dataset...")
        torch_train.train_model(str(config.DATASET_CLEAN), epochs=100)
        return 0

    print("No dataset found. Run: neurolabel demo --synthetic --dev")
    return 1
