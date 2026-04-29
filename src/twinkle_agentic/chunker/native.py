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
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message, ToolCall
from twinkle.template import Template
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
_MEDIA_PART_TYPES = frozenset(
    {'image', 'image_url', 'video', 'video_url', 'audio', 'audio_url'})


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

    def __init__(self, model_id: str, chunk_size: int = 1024, chunk_overlap: int = 50,
                 passage_boundary_re: Optional[str] = None) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError(
                f'invalid params: chunk_size={chunk_size} (expect >0), '
                f'chunk_overlap={chunk_overlap} (expect 0<=overlap<chunk_size)')
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Optional passage-level hard-boundary regex.  Paragraphs whose first
        # character matches this pattern are guaranteed to start a FRESH chunk
        # rather than being greedy-packed next to earlier paragraphs.  Typical
        # use: retrieval corpora that number their passages -- set to
        # ``r'^\[\d+\]\s+'`` so ``[N] Title: body`` aligns 1:1 with chunks and
        # ``ExtractCompressed(idx)`` returns exactly that passage's text.
        self.passage_boundary_re = (
            re.compile(passage_boundary_re) if passage_boundary_re else None)
        self.template = Template(model_id)

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, trajectory: Trajectory) -> Chunks:
        trajectory = self.template.format_trajectory(trajectory)
        chunks: List[Chunk] = []
        for message in trajectory.get('messages') or []:
            chunks.extend(self._chunk_message(message))
        chunks.extend(self._multimodal_chunks(trajectory))
        return Chunks(chunks=chunks)

    # ── Message-level dispatch ────────────────────────────────────────────────

    def _chunk_message(self, message: Message) -> Iterator[Chunk]:
        """Expand a single message into chunks for each of its payloads."""
        role = message.get('role', 'user')
        # Preserve OpenAI-style ``tool_call_id`` (present on ``role='tool'`` messages)
        # so ``Chunks.to_trajectory`` can re-pair tool results with their calls.
        tool_call_id = message.get('tool_call_id')

        if reasoning := message.get('reasoning_content'):
            yield from self._chunk_text(
                reasoning, role=role, kind='reasoning_content', tool_call_id=tool_call_id)

        content = message.get('content')
        if isinstance(content, str):
            yield from self._chunk_text(
                content, role=role, kind='content', tool_call_id=tool_call_id)
        elif isinstance(content, list):
            yield from self._chunk_structured_content(
                content, role=role, tool_call_id=tool_call_id)

        for tool_call in message.get('tool_calls') or []:
            yield self._tool_call_to_chunk(tool_call, role=role)

    def _chunk_structured_content(
        self,
        parts: List[Any],
        role: str,
        *,
        tool_call_id: Optional[str] = None,
    ) -> Iterator[Chunk]:
        """Handle OpenAI-style list content: ``[{'type': 'text', 'text': ...}, ...]``."""
        for part in parts:
            if isinstance(part, str):
                yield from self._chunk_text(
                    part, role=role, kind='content', tool_call_id=tool_call_id)
                continue
            if not isinstance(part, dict):
                continue
            p_type = part.get('type')
            if p_type == 'text':
                yield from self._chunk_text(
                    part.get('text', ''), role=role, kind='content',
                    tool_call_id=tool_call_id)
            elif p_type in _MEDIA_PART_TYPES:
                media_type = p_type.split('_', 1)[0]
                media_value = part.get(f'{media_type}_url') or part.get(media_type) or part
                yield Chunk(type=media_type, content=media_value, raw=part, role=role)

    # ── Text chunking ─────────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        *,
        role: str,
        kind: str = 'content',
        tool_call_id: Optional[str] = None,
    ) -> Iterator[Chunk]:
        text = (text or '').strip()
        if not text:
            return
        for segment in self._iter_semantic_segments(text):
            # ``content`` is the single source of truth; do not duplicate it in ``raw``
            # so downstream condensers that rewrite ``content`` cannot leave a stale
            # ``raw['text']`` behind.
            raw: Dict[str, Any] = {'kind': kind}
            if tool_call_id:
                raw['tool_call_id'] = tool_call_id
            yield Chunk(type='text', content=segment, raw=raw, role=role)

    def _iter_semantic_segments(self, text: str) -> Iterator[str]:
        """Pack paragraphs into segments; hard-split oversized non-code ones.

        When ``passage_boundary_re`` is set, any paragraph whose prefix
        matches is forced to start a fresh segment so numbered passages
        (``[N] ...``) align 1:1 with chunks -- that alignment is what lets
        ``<block_N>`` and ``extract_compressed(N)`` refer to the exact same
        passage text.
        """
        boundary = self.passage_boundary_re
        is_boundary = (lambda p: bool(boundary.match(p))) if boundary else None
        yield from self._greedy_pack(
            self._split_preserving_code(text),
            separator='\n\n',
            on_oversize=self._split_oversize_paragraph,
            is_boundary=is_boundary,
        )

    def _split_oversize_paragraph(self, p: str) -> Iterator[str]:
        """Route an over-size paragraph through the right splitter.

        Fenced code blocks are atomic even when they exceed ``chunk_size``
        (splitting them would break syntax).  Everything else goes through
        ``_hard_split`` which sentence-packs, with the ``[N] Title:`` prefix
        (if any) threaded in so continuation pieces carry a ``(cont.)`` anchor
        rather than becoming orphan chunks invisible to ``extract_compressed``.
        """
        if self._is_code_block(p):
            yield p
            return
        yield from self._hard_split(p, anchor_prefix=self._extract_anchor_prefix(p))

    def _extract_anchor_prefix(self, paragraph: str) -> Optional[str]:
        """Return ``[N] Title:``-style anchor if paragraph opens with one.

        Used by the hard-split path to prefix every continuation piece with
        ``<anchor> (cont.) ...`` so the passage stays identifiable end-to-end.
        Cap the prefix length so a colon-less first line (e.g. a whole
        paragraph on one line) cannot end up as the "anchor".
        """
        boundary = self.passage_boundary_re
        if boundary is None or not boundary.match(paragraph):
            return None
        first_line = paragraph.split('\n', 1)[0]
        colon_idx = first_line.find(':')
        if 0 < colon_idx <= 120:
            return first_line[:colon_idx + 1].strip()
        m = boundary.match(paragraph)
        return m.group(0).strip() if m else None

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

    def _hard_split(self, text: str, *,
                    anchor_prefix: Optional[str] = None) -> Iterator[str]:
        """Split oversized text on sentence boundaries; sliding window if still too long.

        When ``anchor_prefix`` is provided (e.g. ``"[3] Echosmith:"`` from a
        passage-boundary paragraph that overflowed ``chunk_size``), every
        piece *after the first* is rewritten as ``<anchor> (cont.) <piece>``
        so the passage identifier never gets orphaned.
        """
        sentences = [s.strip() for s in _SENTENCE_RE.split(text) if s and s.strip()] \
            or [text.strip()]
        pieces = list(self._greedy_pack(
            sentences, separator=' ', on_oversize=self._sliding_window))
        if not anchor_prefix or len(pieces) <= 1:
            yield from pieces
            return
        # First piece already carries the anchor (it starts with the passage
        # header sentence).  Later pieces get a continuation marker so they
        # remain attributable to the same passage.
        yield pieces[0]
        for p in pieces[1:]:
            if p.startswith(anchor_prefix):
                yield p
            else:
                yield f'{anchor_prefix} (cont.) {p}'

    def _greedy_pack(
        self,
        items: Iterable[str],
        *,
        separator: str,
        on_oversize: Callable[[str], Iterator[str]],
        is_boundary: Optional[Callable[[str], bool]] = None,
    ) -> Iterator[str]:
        """Greedily pack ``items`` into segments of at most ``chunk_size`` chars.

        Items individually exceeding ``chunk_size`` are routed through
        ``on_oversize`` after flushing the current buffer.  When
        ``is_boundary`` is provided, any item for which it returns ``True``
        forces the current buffer to flush first -- used to keep numbered
        passages (``[N] ...``) atomic at the chunk level.
        """
        buf: List[str] = []
        buf_len = 0
        sep_len = len(separator)

        for item in items:
            if buf and is_boundary is not None and is_boundary(item):
                yield separator.join(buf)
                buf, buf_len = [], 0
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

    # ── Tool calls & multi-modal ──────────────────────────────────────────────

    @staticmethod
    def _tool_call_to_chunk(tool_call: ToolCall, *, role: str) -> Chunk:
        """Render a tool call as a single indexable chunk, keeping it intact."""
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
        """Emit one chunk per trajectory-level multi-modal reference."""
        for key, media_type in _MEDIA_KEYS:
            for item in trajectory.get(key) or []:
                yield Chunk(type=media_type, content=item, raw=item, role='user')
