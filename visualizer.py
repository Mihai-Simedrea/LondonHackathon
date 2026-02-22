#!/usr/bin/env python3
"""Compatibility wrapper for `neurolabel.ui.replay.velocity_viewer`."""

from neurolabel.ui.replay.velocity_viewer import *  # noqa: F401,F403


if __name__ == "__main__":
    from neurolabel.ui.replay.velocity_viewer import generate_mock_results, run_visualizer

    mock_dirty = generate_mock_results()
    mock_clean = generate_mock_results()
    run_visualizer(mock_dirty, mock_clean)
