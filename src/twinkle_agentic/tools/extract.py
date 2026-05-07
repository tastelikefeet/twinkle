# Copyright (c) ModelScope Contributors. All rights reserved.
"""ExtractCompressed: recall original text for <block_N> markers."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.data_format import Chunks

from .base import Tool


class ExtractCompressed(Tool):
    """Recall original (pre-compression) text of <block_N> markers.

    The ``<block_N>`` numbering emitted by :meth:`Chunks.to_trajectory` is a
    1-based consecutive counter over wrapped (condensed) chunks only. That
    displayed number does NOT equal the chunk's 0-based position in
    ``original_chunks.chunks``. Callers should pass the
    ``displayed_to_full`` mapping from
    :meth:`Chunks.displayed_block_mapping` so this tool can translate the
    displayed block number emitted by the model back into the concrete
    chunk that holds the original passage text.

    If ``displayed_to_full`` is omitted, the tool falls back to treating
    the block number as a direct 0-based index (legacy behaviour).
    """

    name = 'extract_compressed'

    def __init__(
        self,
        original_chunks: Chunks,
        max_blocks_per_call: int = 16,
        displayed_to_full: Optional[Dict[int, int]] = None,
    ) -> None:
        self.original_chunks = original_chunks
        self.max_blocks_per_call = max_blocks_per_call
        self.displayed_to_full = dict(displayed_to_full) if displayed_to_full else None

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            return f'Error: arguments must be an object, got {type(arguments).__name__}.'

        blocks = self._parse_blocks(arguments.get('blocks'))
        if not blocks:
            return 'Error: empty or invalid "blocks" argument. Pass e.g. {"blocks": [1, 3]}.'
        if len(blocks) > self.max_blocks_per_call:
            return (f'Error: too many blocks ({len(blocks)} > '
                    f'{self.max_blocks_per_call}). Split the request.')

        return '\n'.join(self._render_block(b) for b in blocks)

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_block(self, displayed: int) -> str:
        """Render a single ``<block_N>...</block_N>`` for the given number.

        Translates ``displayed`` (1-based, as the model emits it) to the
        underlying full-chunk index using ``displayed_to_full``, then
        returns either the original passage text or a structured error
        message wrapped in matching block tags. The displayed identifier
        is preserved in the response so the model sees what it asked for.
        """
        def wrap(body: str) -> str:
            return f'<block_{displayed}>{body}</block_{displayed}>'

        # 1. Resolve displayed -> full-chunk index.
        if self.displayed_to_full is not None:
            if displayed not in self.displayed_to_full:
                valid = sorted(self.displayed_to_full)
                return wrap(f'ERROR: no such block. Valid blocks: {valid}.')
            idx = self.displayed_to_full[displayed]
        else:
            idx = displayed

        # 2. Bounds check.
        n_chunks = len(self.original_chunks.chunks)
        if not 0 <= idx < n_chunks:
            return wrap(f'ERROR: index out of range [0, {n_chunks})')

        # 3. Skip structural / non-text chunks.
        chunk = self.original_chunks.chunks[idx]
        if self._is_structural(chunk):
            return wrap('This block is structural, not a passage. '
                        'Pick a different passage block instead.')
        content = chunk.get('content')
        if not isinstance(content, str):
            return wrap(f'[non-text chunk, type={chunk.get("type")!r}]')

        return wrap(content)

    def tool_info(self) -> ToolInfo:
        return {
            'tool_name': self.name,
            'description': (
                'Recall the original text of one or more <block_N> markers. '
                'Use when compressed content omits details you need.'),
            'parameters': json.dumps({
                'blocks': {
                    'type': 'array',
                    'items': {'type': 'integer', 'minimum': 1},
                    'description': f'Block numbers (as shown in <block_N> tags) to expand. Up to {self.max_blocks_per_call} per call.',
                },
            }),
        }

    @staticmethod
    def _is_structural(chunk: Dict[str, Any]) -> bool:
        """True if chunk is system/tool/question (not a passage)."""
        role = chunk.get('role')
        if role in ('system', 'tool'):
            return True
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in ('tool_call', 'tool_response', 'question'):
            return True
        content = chunk.get('content', '')
        if isinstance(content, str) and content.lstrip().startswith('Question:'):
            return True
        return False

    @staticmethod
    def _parse_blocks(value: Any) -> List[int]:
        """Normalise blocks argument into deduplicated list of ints."""
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
        return list(dict.fromkeys(parsed))
