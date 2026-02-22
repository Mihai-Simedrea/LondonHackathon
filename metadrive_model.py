"""Compatibility wrapper for the canonical MetaDrive model module."""

from neurolabel.backends.metadrive.model import load_model, predict, save_model, train_model

__all__ = ["train_model", "save_model", "load_model", "predict"]
