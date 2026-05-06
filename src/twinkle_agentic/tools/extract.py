# Copyright (c) ModelScope Contributors. All rights reserved.
"""ExtractCompressed: recall original text for <block_N> markers."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.data_format import Chunks

from .base import Tool


class ExtractCompressed(Tool):
    """Recall original (pre-compression) text of <block_N> markers."""

    name = 'extract_compressed'

    def __init__(self, original_chunks: Chunks, max_blocks_per_call: int = 16) -> None:
        self.original_chunks = original_chunks
        self.max_blocks_per_call = max_blocks_per_call

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
            if self._is_structural(chunk):
                rendered.append(
                    f'<block_{idx}>This block is structural, not a passage. '
                    f'Pick a higher-numbered passage block instead.</block_{idx}>')
                continue
            content = chunk.get('content')
            if not isinstance(content, str):
                rendered.append(
                    f'<block_{idx}>[non-text chunk, type={chunk.get("type")!r}]</block_{idx}>')
                continue
            rendered.append(f'<block_{idx}>{content}</block_{idx}>')
        return '\n'.join(rendered)

    def tool_info(self) -> ToolInfo:
        return {
            'tool_name': self.name,
            'description': (
                'Recall the original text of one or more <block_N> markers. '
                'Use when compressed content omits details you need.'),
            'parameters': json.dumps({
                'blocks': {
                    'type': 'array',
                    'items': {'type': 'integer', 'minimum': 0},
                    'description': f'Block numbers to expand. Up to {self.max_blocks_per_call} per call.',
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
