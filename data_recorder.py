#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.backends.velocity.recording`."""

from neurolabel.backends.velocity.recording import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
