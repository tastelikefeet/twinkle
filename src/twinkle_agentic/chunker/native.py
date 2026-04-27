# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native rule-based trajectory chunker.

This module implements a deterministic chunker that splits a
:class:`~twinkle.data_format.Trajectory` into a list of
:class:`~twinkle_agentic.data_format.chunk.Chunk` objects using traditional
text-processing rules (no LLM calls).  It produces semantically self-contained
chunks by respecting natural boundaries such as paragraph breaks, fenced code
blocks and sentence endings, while also covering the structured elements of a
trajectory (reasoning content, tool calls, multi-modal references).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message, ToolCall
from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Chunker

# ── Split helpers ─────────────────────────────────────────────────────────────
# Keep fenced code blocks (``` ... ```) intact when splitting.
_CODE_FENCE_RE = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
# Paragraph boundary = blank line.
_PARAGRAPH_RE = re.compile(r'\n\s*\n')
# Sentence boundary = end-of-sentence punctuation (CN+EN) or hard newline.
_SENTENCE_RE = re.compile(r'(?<=[.!?。！？])\s+|\n+')

# Multi-modal keys on a Trajectory map 1:1 to Chunk.type values.
_MEDIA_KEYS = (('images', 'image'), ('videos', 'video'), ('audios', 'audio'))


class NativeChunker(Chunker):
    """Rule-based chunker.

    The chunker walks every element of a :class:`Trajectory` and emits chunks
    that preserve as much semantic integrity as possible:

    * Each :class:`Message` is expanded into separate chunks for its
      ``reasoning_content``, ``content`` and every entry in ``tool_calls``.
    * Plain text is split on paragraph / code-fence / sentence boundaries so a
      chunk is never cut in the middle of a fenced code block when it can fit.
    * Tool calls and multi-modal references are emitted as single, intact
      chunks so they can be indexed or embedded as a whole.

    Args:
        chunk_size: Soft upper bound (in characters) for a single text chunk.
            Oversized but semantically indivisible segments (e.g. one huge
            fenced code block) are emitted as-is to keep semantics intact.
        chunk_overlap: Number of characters duplicated between adjacent chunks
            produced by hard-splitting an oversized segment via a sliding
            window.  Must satisfy ``0 <= chunk_overlap < chunk_size``.

    Example:
        >>> trajectory = {
        ...     'messages': [
        ...         {
        ...             'role': 'user',
        ...             'content': [
        ...                 {'type': 'image', 'image': '/tmp/plot.png'},
        ...                 {'type': 'text', 'text': 'Sort the list [3, 1, 2].'},
        ...             ],
        ...         },
        ...         {
        ...             'role': 'assistant',
        ...             'reasoning_content': 'I should call a python tool to sort it.',
        ...             'content': 'Let me run a snippet.\n\n```python\nsorted([3, 1, 2])\n```',
        ...             'tool_calls': [
        ...                 {'tool_name': 'python', 'arguments': '{"code": "sorted([3, 1, 2])"}'},
        ...             ],
        ...         },
        ...         {'role': 'tool', 'content': '[1, 2, 3]'},
        ...     ],
        ... }
        >>> chunker = NativeChunker(chunk_size=512)
        >>> chunks = chunker.chunk(trajectory).chunks
        >>> for c in chunks:
        ...     print(c['role'], c['type'], '|', repr(c['content']))
        user      image | '/tmp/plot.png'
        user      text  | 'Sort the list [3, 1, 2].'
        assistant text  | 'I should call a python tool to sort it.'
        assistant text  | 'Let me run a snippet.\n\n```python\nsorted([3, 1, 2])\n```'
        assistant text  | '[tool_call:python]\n{\n  "code": "sorted([3, 1, 2])"\n}'
        tool      text  | '[1, 2, 3]'

        Notes:
          * Multi-modal parts embedded in a message's structured ``content``
            inherit that message's role (e.g. the image above is attributed
            to ``user`` because it lives in a ``user`` message).
          * ``reasoning_content`` is emitted before ``content`` so retrieval
            can weight them differently (``raw.kind`` distinguishes them).
          * The fenced ``python`` block is preserved inside a single chunk
            rather than split across lines.
          * Each ``tool_call`` becomes one atomic chunk, rendered as
            ``[tool_call:<name>]\\n<pretty-json-args>``; the structured form
            is kept in ``chunk['raw']`` for downstream reconstruction.
          * Trajectory-level ``images`` / ``videos`` / ``audios`` are also
            supported as a fallback for references not bound to any specific
            message; those chunks are attributed to ``user``.
    """

    def __init__(self, model_id: str, chunk_size: int = 1024, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be positive, got {chunk_size}')
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f'chunk_overlap must satisfy 0 <= overlap < chunk_size, got {chunk_overlap}')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.template = Template(model_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, trajectory: Trajectory) -> Chunks:
        chunks: List[Chunk] = []
        trajectory = self.template.format_trajectory(trajectory)
        for message in trajectory.get('messages') or []:
            chunks.extend(self._chunk_message(message))

        chunks.extend(self._multimodal_chunks(trajectory))
        return Chunks(chunks=chunks)

    # ── Message-level dispatch ────────────────────────────────────────────────

    def _chunk_message(self, message: Message) -> Iterator[Chunk]:
        """Expand a single message into chunks for each of its payloads."""
        role = message.get('role', 'user')

        reasoning = message.get('reasoning_content')
        if reasoning:
            yield from self._chunk_text(reasoning, role=role, kind='reasoning_content')

        content = message.get('content')
        if isinstance(content, str):
            yield from self._chunk_text(content, role=role, kind='content')
        elif isinstance(content, list):
            yield from self._chunk_structured_content(content, role=role)

        for tool_call in message.get('tool_calls') or []:
            yield self._tool_call_to_chunk(tool_call, role=role)

    def _chunk_structured_content(self, parts: List[Any], role: str) -> Iterator[Chunk]:
        """Handle OpenAI-style list content: ``[{'type': 'text', 'text': ...}, ...]``."""
        for part in parts:
            if isinstance(part, str):
                yield from self._chunk_text(part, role=role, kind='content')
                continue
            if not isinstance(part, dict):
                continue

            p_type = part.get('type')
            if p_type == 'text':
                yield from self._chunk_text(part.get('text', ''), role=role, kind='content')
            elif p_type in {'image', 'image_url', 'video', 'video_url', 'audio', 'audio_url'}:
                media_type = p_type.split('_', 1)[0]
                media_value = part.get(f'{media_type}_url') or part.get(media_type) or part
                yield Chunk(type=media_type, content=media_value, raw=part, role=role)

    # ── Text chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, *, role: str, kind: str = 'content') -> Iterator[Chunk]:
        text = (text or '').strip()
        if not text:
            return
        for segment in self._iter_semantic_segments(text):
            yield Chunk(
                type='text',
                content=segment,
                raw={'kind': kind, 'text': segment},
                role=role,
            )

    def _iter_semantic_segments(self, text: str) -> Iterator[str]:
        """Yield text segments that honour paragraph and code-fence boundaries."""
        buffer: List[str] = []
        buffer_len = 0

        def flush() -> Iterator[str]:
            nonlocal buffer, buffer_len
            if buffer:
                yield '\n\n'.join(buffer)
                buffer, buffer_len = [], 0

        for paragraph in self._split_preserving_code(text):
            # Oversized non-code paragraph: flush buffer, then hard-split.
            # Code blocks are always kept intact to preserve semantics.
            if len(paragraph) > self.chunk_size and not self._is_code_block(paragraph):
                yield from flush()
                yield from self._hard_split(paragraph)
                continue

            separator_cost = 2 if buffer else 0
            if buffer and buffer_len + separator_cost + len(paragraph) > self.chunk_size:
                yield from flush()

            buffer.append(paragraph)
            buffer_len += len(paragraph) + (2 if len(buffer) > 1 else 0)

        yield from flush()

    @staticmethod
    def _split_preserving_code(text: str) -> List[str]:
        """Split on blank lines while keeping fenced code blocks as one unit."""
        paragraphs: List[str] = []
        for part in _CODE_FENCE_RE.split(text):
            if not part:
                continue
            if part.startswith('```'):
                paragraphs.append(part.strip())
            else:
                paragraphs.extend(p.strip() for p in _PARAGRAPH_RE.split(part) if p.strip())
        return paragraphs

    @staticmethod
    def _is_code_block(segment: str) -> bool:
        return segment.startswith('```') and segment.rstrip().endswith('```')

    def _hard_split(self, text: str) -> Iterator[str]:
        """Fallback splitter for segments exceeding ``chunk_size``.

        Splits on sentence boundaries first; falls back to a sliding window
        (with optional overlap) when a single sentence is still too long.
        """
        sentences = [s for s in _SENTENCE_RE.split(text) if s and s.strip()] or [text]
        buffer = ''

        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if buffer:
                    yield buffer
                    buffer = ''
                yield from self._sliding_window(sentence)
                continue

            if buffer and len(buffer) + 1 + len(sentence) > self.chunk_size:
                yield buffer
                buffer = sentence
            else:
                buffer = f'{buffer} {sentence}' if buffer else sentence

        if buffer:
            yield buffer

    def _sliding_window(self, text: str) -> Iterator[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            yield text[start:start + self.chunk_size]
            if start + self.chunk_size >= len(text):
                break

    # ── Tool calls & multi-modal ──────────────────────────────────────────────

    @staticmethod
    def _tool_call_to_chunk(tool_call: ToolCall, *, role: str) -> Chunk:
        """Render a tool call as a single indexable chunk, keeping it intact."""
        name = tool_call.get('tool_name', 'tool')
        arguments = tool_call.get('arguments', '')
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            rendered_args = json.dumps(parsed, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            rendered_args = str(arguments)

        return Chunk(
            type='text',
            content=f'[tool_call:{name}]\n{rendered_args}',
            raw={'kind': 'tool_call', 'tool_call': dict(tool_call)},
            role=role,
        )

    @staticmethod
    def _multimodal_chunks(trajectory: Trajectory) -> Iterator[Chunk]:
        """Emit one chunk per trajectory-level multi-modal reference."""
        for key, media_type in _MEDIA_KEYS:
            for item in trajectory.get(key) or []:
                yield Chunk(type=media_type, content=item, raw=item, role='user')
