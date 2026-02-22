#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.experiments.torch_train`."""

from neurolabel.experiments.torch_train import *  # noqa: F401,F403
from neurolabel.experiments import torch_train as _impl


if __name__ == "__main__":
    import config

    print(f"PyTorch {_impl.torch.__version__} | Device: {_impl.DEVICE}")
    if config.DATASET_CLEAN.exists():
        print("\nTraining on clean dataset...")
        _impl.train_model(str(config.DATASET_CLEAN), epochs=100)
    else:
        print("No dataset found. Run: python pipeline.py demo --synthetic --dev")
