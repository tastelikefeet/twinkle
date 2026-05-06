# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rule-based trajectory chunker: splits Trajectory into Chunks."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message, ToolCall
from twinkle.template import Template
from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Chunker

# Atomic spans: code fences and <block_N> tags must never be split.
_ATOMIC_SPAN_RE = re.compile(
    r'(```[\s\S]*?```|<block_\d+>[\s\S]*?</block_\d+>)', re.MULTILINE)
_BLOCK_TAG_RE = re.compile(r'(<block_\d+>[\s\S]*?</block_\d+>)')
_PARAGRAPH_RE = re.compile(r'\n\s*\n')
_SENTENCE_RE = re.compile(r'(?<=[.!?。！？])\s+|\n+')

_MEDIA_KEYS = (('images', 'image'), ('videos', 'video'), ('audios', 'audio'))
_MEDIA_PART_TYPES = frozenset(
    {'image', 'image_url', 'video', 'video_url', 'audio', 'audio_url'})


class NativeChunker(Chunker):
    """Rule-based chunker respecting code fences, block tags, and passage boundaries."""

    def __init__(self, model_id: str, chunk_size: int = 1024, chunk_overlap: int = 0,
                 passage_boundary_re: Optional[str] = None) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # When set, paragraphs matching this regex start a fresh chunk.
        # E.g. r'^\[\d+\]\s+' ensures [N] Title: passages align 1:1 with blocks.
        self.passage_boundary_re = (
            re.compile(passage_boundary_re) if passage_boundary_re else None)
        self.template = Template(model_id)

    def chunk(self, trajectory: Trajectory) -> Chunks:
        trajectory = self.template.format_trajectory(trajectory)
        chunks: List[Chunk] = []
        for message in trajectory.get('messages') or []:
            chunks.extend(self._chunk_message(message))
        chunks.extend(self._multimodal_chunks(trajectory))
        return Chunks(chunks=chunks)

    # ── Message dispatch ──────────────────────────────────────────────────────

    def _chunk_message(self, message: Message) -> Iterator[Chunk]:
        role = message.get('role', 'user')
        tool_call_id = message.get('tool_call_id')

        if reasoning := message.get('reasoning_content'):
            yield from self._chunk_text(reasoning, role=role, kind='reasoning_content',
                                        tool_call_id=tool_call_id)

        content = message.get('content')
        if isinstance(content, str):
            yield from self._chunk_text(content, role=role, kind='content',
                                        tool_call_id=tool_call_id)
        elif isinstance(content, list):
            yield from self._chunk_structured_content(content, role=role,
                                                     tool_call_id=tool_call_id)

        for tool_call in message.get('tool_calls') or []:
            yield self._tool_call_to_chunk(tool_call, role=role)

    def _chunk_structured_content(self, parts: List[Any], role: str, *,
                                  tool_call_id: Optional[str] = None) -> Iterator[Chunk]:
        for part in parts:
            if isinstance(part, str):
                yield from self._chunk_text(part, role=role, kind='content',
                                            tool_call_id=tool_call_id)
                continue
            if not isinstance(part, dict):
                continue
            p_type = part.get('type')
            if p_type == 'text':
                yield from self._chunk_text(part.get('text', ''), role=role, kind='content',
                                            tool_call_id=tool_call_id)
            elif p_type in _MEDIA_PART_TYPES:
                media_type = p_type.split('_', 1)[0]
                media_value = part.get(f'{media_type}_url') or part.get(media_type) or part
                yield Chunk(type=media_type, content=media_value, raw=part, role=role)

    # ── Text chunking ─────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, *, role: str, kind: str = 'content',
                    tool_call_id: Optional[str] = None) -> Iterator[Chunk]:
        text = (text or '').strip()
        if not text:
            return
        for segment in self._iter_segments(text):
            raw: Dict[str, Any] = {'kind': kind}
            if tool_call_id:
                raw['tool_call_id'] = tool_call_id
            yield Chunk(type='text', content=segment, raw=raw, role=role)

    def _iter_segments(self, text: str) -> Iterator[str]:
        """Pack paragraphs into segments respecting chunk_size and boundaries."""
        boundary = self.passage_boundary_re
        is_boundary = (lambda p: bool(boundary.match(p))) if boundary else None
        yield from self._greedy_pack(
            self._split_preserving_atomic(text),
            separator='\n\n',
            on_oversize=self._split_oversize,
            is_boundary=is_boundary,
        )

    def _split_oversize(self, p: str) -> Iterator[str]:
        """Route oversized paragraph: atomic spans pass through, others get sentence-split."""
        if p.startswith('```') or _BLOCK_TAG_RE.fullmatch(p):
            yield p
            return
        yield from self._hard_split(p)

    @staticmethod
    def _split_preserving_atomic(text: str) -> List[str]:
        """Split on blank lines, keeping code fences and block tags intact."""
        paragraphs: List[str] = []
        for part in _ATOMIC_SPAN_RE.split(text):
            if not part:
                continue
            if part.startswith('```') or _BLOCK_TAG_RE.fullmatch(part):
                paragraphs.append(part.strip())
            else:
                paragraphs.extend(p.strip() for p in _PARAGRAPH_RE.split(part) if p.strip())
        return paragraphs

    def _hard_split(self, text: str) -> Iterator[str]:
        """Split oversized text on sentence boundaries, then sliding window."""
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()] \
            or [text.strip()]
        yield from self._greedy_pack(sentences, separator=' ', on_oversize=self._sliding_window)

    def _greedy_pack(self, items: Iterable[str], *, separator: str,
                     on_oversize: Callable[[str], Iterator[str]],
                     is_boundary: Optional[Callable[[str], bool]] = None) -> Iterator[str]:
        """Greedily pack items into segments of at most chunk_size chars."""
        buf: List[str] = []
        buf_len = 0
        sep_len = len(separator)

        for item in items:
            # Boundary items force flush
            if buf and is_boundary is not None and is_boundary(item):
                yield separator.join(buf)
                buf, buf_len = [], 0
            # Oversized items flush + delegate
            if len(item) > self.chunk_size:
                if buf:
                    yield separator.join(buf)
                    buf, buf_len = [], 0
                yield from on_oversize(item)
                continue
            need = len(item) + (sep_len if buf else 0)
            if buf_len + need > self.chunk_size:
                yield separator.join(buf)
                buf, buf_len = [item], len(item)
            else:
                buf.append(item)
                buf_len += need

        if buf:
            yield separator.join(buf)

    def _sliding_window(self, text: str) -> Iterator[str]:
        step = max(1, self.chunk_size - self.chunk_overlap)
        for start in range(0, len(text), step):
            yield text[start:start + self.chunk_size]
            if start + self.chunk_size >= len(text):
                break

    # ── Tool calls & media ────────────────────────────────────────────────────

    @staticmethod
    def _tool_call_to_chunk(tool_call: ToolCall, *, role: str) -> Chunk:
        name = tool_call.get('tool_name', 'tool')
        arguments = tool_call.get('arguments', '')
        try:
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            rendered = json.dumps(parsed, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            rendered = str(arguments)
        return Chunk(
            type='text',
            content=f'[tool_call:{name}]\n{rendered}',
            raw={'kind': 'tool_call', 'tool_call': dict(tool_call)},
            role=role,
        )

    @staticmethod
    def _multimodal_chunks(trajectory: Trajectory) -> Iterator[Chunk]:
        for key, media_type in _MEDIA_KEYS:
            for item in trajectory.get(key) or []:
                yield Chunk(type=media_type, content=item, raw=item, role='user')
