#!/usr/bin/env python3
"""Compatibility wrapper for the new NeuroLabel CLI package.

The canonical entrypoint is `neurolabel.ui.cli.main`, but this file is kept so
existing commands like `python pipeline.py demo --synthetic --dev` still work
during the migration.
"""

from neurolabel.ui.cli.main import main


if __name__ == "__main__":
    main()

