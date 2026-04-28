# Copyright (c) ModelScope Contributors. All rights reserved.
"""``ExtractCompressed``: recall original text for ``<block_N>`` markers.

This tool is the RL-time counterpart of the ``block_wrapper`` parameter of
:meth:`twinkle_agentic.data_format.chunk.Chunks.to_trajectory`.  The typical
pipeline is::

    full_chunks = chunker.chunk(trajectory)            # before compression
    compressed  = condenser.condense(full_chunks)      # after compression
    prompt      = compressed.to_trajectory()           # LLM sees <block_N>...

    mgr = ToolManager([ExtractCompressed(full_chunks)])

During the rollout the model reads the compressed trajectory with each
chunk fenced by ``<block_{n}>...</block_{n}>`` markers.  When it needs
details that were pruned by the condenser, it emits a tool call referencing
one or more block numbers and this tool expands them back to their
pre-compression text.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.data_format import Chunks

from .base import Tool


class ExtractCompressed(Tool):
    """Recall the original (pre-compression) text of ``<block_N>`` markers.

    The tool is bound to a *single* :class:`Chunks` object -- the full,
    uncompressed output of the chunker.  Block numbers are 0-based indices
    into ``original_chunks.chunks`` (the same ``{n}`` substituted into the
    ``block_wrapper`` template at prompt time), so the LLM can cross-
    reference markers it saw in the prompt directly.

    Args:
        original_chunks: The *pre-compression* chunks that block numbers
            index into.  Capture this before calling the condenser.
        max_blocks_per_call: Upper bound on how many blocks one call can
            expand, to prevent the model ballooning the context by
            requesting everything at once.  Defaults to ``16``.

    Example:
        >>> full = chunker.chunk(traj)
        >>> compressed = condenser.condense(full)
        >>> # model sees compressed.to_trajectory() with <block_N> markers
        >>> tool = ExtractCompressed(full)
        >>> tool('extract_compressed', {'blocks': [1, 3]})
        '<block_1>...original text of chunk 1...</block_1>\\n'
        '<block_3>...original text of chunk 3...</block_3>'
    """

    name = 'extract_compressed'

    def __init__(self, original_chunks: Chunks, max_blocks_per_call: int = 16) -> None:
        if max_blocks_per_call < 1:
            raise ValueError(
                f'max_blocks_per_call must be >=1, got {max_blocks_per_call}')
        self.original_chunks = original_chunks
        self.max_blocks_per_call = max_blocks_per_call

    # -- Dispatch entry point -------------------------------------------------

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            return f'Error: arguments must be an object, got {type(arguments).__name__}.'

        blocks = self._parse_blocks(arguments.get('blocks'))
        if not blocks:
            return 'Error: empty or invalid "blocks" argument. Pass e.g. {"blocks": [1, 3]}.'
        if len(blocks) > self.max_blocks_per_call:
            return (f'Error: too many blocks ({len(blocks)} > '
                    f'{self.max_blocks_per_call}). Split the request.')

        n_chunks = len(self.original_chunks.chunks)
        rendered: List[str] = []
        for idx in blocks:
            if not 0 <= idx < n_chunks:
                rendered.append(
                    f'<block_{idx}>ERROR: index out of range [0, {n_chunks})</block_{idx}>')
                continue
            chunk = self.original_chunks.chunks[idx]
            content = chunk.get('content')
            if not isinstance(content, str):
                # Non-text chunks (image / video / audio / structural): describe
                # the shape instead of dumping a raw dict into the prompt.
                rendered.append(
                    f'<block_{idx}>[non-text chunk, type={chunk.get("type")!r}]</block_{idx}>')
                continue
            rendered.append(f'<block_{idx}>{content}</block_{idx}>')
        return '\n'.join(rendered)

    # -- Tool advertisement ---------------------------------------------------

    def tool_info(self) -> ToolInfo:
        return {
            'tool_name': self.name,
            'description': (
                'Recall the original (pre-compression) text of one or more '
                '<block_N> markers observed in the current prompt. Use this '
                'when the compressed content omits details you need to '
                'answer accurately.'),
            'parameters': json.dumps({
                'blocks': {
                    'type': 'array',
                    'items': {'type': 'integer', 'minimum': 0},
                    'description': (
                        'List of block numbers (the N in <block_N>) to '
                        f'expand. Up to {self.max_blocks_per_call} per call.'),
                },
            }),
        }

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _parse_blocks(value: Any) -> List[int]:
        """Normalise the ``blocks`` argument into a deduplicated list of ints.

        Accepts an int, a list/tuple of ints or numeric strings, or a
        comma-separated string such as ``"1,3, 5"``.  Silently drops
        items that cannot be coerced to ``int``.
        """
        # Reject None and bools up-front; bool is a subclass of int but not a block id.
        if value is None or isinstance(value, bool):
            return []
        if isinstance(value, int):
            items: Iterable[Any] = (value,)
        elif isinstance(value, str):
            items = value.split(',')
        elif isinstance(value, (list, tuple)):
            items = value
        else:
            return []

        parsed: List[int] = []
        for item in items:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        # dict.fromkeys preserves insertion order while deduplicating.
        return list(dict.fromkeys(parsed))
