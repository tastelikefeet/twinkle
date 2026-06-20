# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle TUI - Terminal User Interface for training control."""

__version__ = '0.1.0'


def main():
    """Entry point for `twinkle-tui` command."""
    from twinkle_tui.app import TwinkleTUI
    app = TwinkleTUI()
    app.run()
