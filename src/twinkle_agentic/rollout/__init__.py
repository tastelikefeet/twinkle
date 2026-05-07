# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic multi-turn agentic rollout primitives.

Extracted from the HotpotQA cookbooks so every agentic-RL training script
can reuse the same chunk → condense → sample → tool loop without
copy-pasting hundreds of lines.
"""
from .agentic import (
    FrozenContext,
    Rollout,
    batch_freeze_delta_pairs,
    clean_assistant_output,
    ensure_context_header,
    parse_tool_calls,
    run_agentic_rollouts,
    strip_passage_prefix,
)

__all__ = [
    'FrozenContext',
    'Rollout',
    'batch_freeze_delta_pairs',
    'clean_assistant_output',
    'ensure_context_header',
    'parse_tool_calls',
    'run_agentic_rollouts',
    'strip_passage_prefix',
]
