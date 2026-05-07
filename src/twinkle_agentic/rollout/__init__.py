# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic multi-turn agentic rollout primitives.

The rollout loop here is compression-agnostic: callers plug in
compression (or any other per-turn context edit) via the
``display_builder`` callback. Compression machinery itself lives in
:mod:`twinkle_agentic.condenser`.
"""
from .agentic import (
    DisplayBuilder,
    OutputSanitizer,
    Rollout,
    ToolFactory,
    TurnHook,
    run_agentic_rollouts,
)

__all__ = [
    'DisplayBuilder',
    'OutputSanitizer',
    'Rollout',
    'ToolFactory',
    'TurnHook',
    'run_agentic_rollouts',
]
