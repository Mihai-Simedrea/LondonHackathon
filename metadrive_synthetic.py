"""Compatibility wrapper for the canonical MetaDrive synthetic data module."""

from neurolabel.backends.metadrive.synthetic import generate_synthetic_metadrive

__all__ = ["generate_synthetic_metadrive"]


if __name__ == "__main__":
    generate_synthetic_metadrive()
