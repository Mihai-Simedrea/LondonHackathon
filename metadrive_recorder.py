#!/usr/bin/env python3
"""Compatibility wrapper for the canonical MetaDrive recorder module."""

import sys

from neurolabel.backends.metadrive.recording import record_session

__all__ = ["record_session"]


if __name__ == "__main__":
    max_sec = None
    no_brain = "--no-brain" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--no-brain"]
    if args:
        max_sec = int(args[0])
    record_session(max_seconds=max_sec, record_brain=not no_brain)
