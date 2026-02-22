#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.backends.velocity.synthetic`."""

from neurolabel.backends.velocity.synthetic import *  # noqa: F401,F403


if __name__ == "__main__":
    from neurolabel.backends.velocity.synthetic import generate_synthetic

    generate_synthetic()
