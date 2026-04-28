# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native TF-IDF based per-chunk condenser.

This module compresses each chunk's text content using extractive TF-IDF
summarisation *without changing the chunk count*.  For every text chunk we
split the content into sentence-sized units (keeping fenced code blocks
atomic), score each unit by how many distinctive terms of the trajectory it
carries, and keep just enough units to fit a target character budget.

Structural chunks (``tool_call``) and multi-modal chunks
(``image`` / ``video`` / ``audio``) pass through untouched: their
sequence position and metadata are essential for re-assembling the
trajectory, and they cannot be meaningfully summarised by text rules.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Union

RatioLike = Union[float, Tuple[float, float], List[float]]

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser

# ── Regex helpers ─────────────────────────────────────────────────────────────
# Bilingual tokenizer: Latin / digit words + single CJK characters.
_TOKEN_RE = re.compile(r'[A-Za-z0-9_]+|[\u4e00-\u9fff]')
# Keep fenced code blocks (``` ... ```) atomic when splitting content.
_CODE_FENCE_RE = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
# Sentence boundaries: end-of-sentence punctuation (CN+EN) or hard newline.
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?。！？])\s+|\n+')

# Chunks that bypass compression entirely.
_PROTECTED_KINDS: Tuple[str, ...] = ('tool_call',)
_PROTECTED_TYPES: Tuple[str, ...] = ('image', 'video', 'audio')


class NativeCondenser(Condenser):
    """Extractive per-chunk TF-IDF condenser.

    The condenser shortens each text chunk's ``content`` to roughly
    ``keep_ratio`` of its original character length by dropping the least
    informative sentences, while keeping:

    * the **chunk count** unchanged,
    * every chunk's ``role`` / ``type`` / ``raw`` metadata,
    * fenced code blocks as atomic units (never truncated mid-block),
    * structural and multi-modal chunks verbatim.

    The scoring corpus is the set of all compressible text chunks in the
    input, so terms that are distinctive to the current trajectory get high
    IDF weight.  Surviving sentences are emitted in original order so the
    shortened content still reads as a coherent narrative.

    Args:
        keep_ratio: Target character ratio per text chunk after compression,
            in ``(0.0, 1.0]``.  ``1.0`` disables compression.  May be either

            * a single ``float`` -- uniform ratio applied to every
              compressible chunk, or
            * a ``(min_keep_ratio, max_keep_ratio)`` tuple -- a gradient that
              **linearly interpolates** across the sequence of compressible
              chunks.  Because early context is typically stale / less
              valuable than recent context, the first compressible chunk gets
              ``min_keep_ratio`` (most aggressive compression) and the last
              gets ``max_keep_ratio`` (most preserved).  Non-compressible
              chunks (protected kinds / multi-modal) are skipped when counting
              positions, so the gradient spans only the chunks actually being
              shortened.

            Defaults to ``0.5`` (uniform).
        min_sentences: Lower bound on sentences retained per chunk regardless
            of the character budget.  Defaults to ``1``.
        min_chars: Chunks with content shorter than this threshold are kept
            verbatim (already compact).  Defaults to ``40``.

    Example:
        Given five chunks produced by :class:`NativeChunker` (a user question,
        an assistant reasoning trace, a tool call, an assistant reply
        containing a fenced code block, and a trajectory-level image), the
        condenser transforms ``content`` in place while keeping the chunk
        count, roles, types and structural chunks untouched::

            >>> condensed = NativeCondenser(keep_ratio=0.5).condense(chunks)
            count: 5 -> 5

            [0] role=user      type=text   len=102 -> 55
                BEFORE: 'Please compute the eigenvalues of matrix '
                        '[[2,1],[1,3]]. This matters for my thesis. '
                        'Thanks in advance.'
                AFTER : 'Please compute the eigenvalues of matrix '
                        '[[2,1],[1,3]].'

            [1] role=assistant type=text   len=288 -> 161
                BEFORE: 'Alright, so the user wants me to find eigenvalues. '
                        'Let me think about this carefully. First I need to '
                        'set up the characteristic equation. We compute '
                        'det(A - lambda I) = 0 for the given 2x2 matrix. '
                        'That expands into a quadratic in lambda. Solving '
                        'it yields the two eigenvalues numerically.'
                AFTER : 'Alright, so the user wants me to find eigenvalues. '
                        'First I need to set up the characteristic '
                        'equation. We compute det(A - lambda I) = 0 for '
                        'the given 2x2 matrix.'

            [2] role=assistant type=text   len=47  -> 47   (tool_call, protected)
                BEFORE: '[tool_call:numpy_eig]\n{"matrix": [[2,1],[1,3]]}'
                AFTER : '[tool_call:numpy_eig]\n{"matrix": [[2,1],[1,3]]}'

            [3] role=assistant type=text   len=133 -> 91
                BEFORE: 'Here is my code:\n\n```python\nimport numpy as np\n'
                        'np.linalg.eig([[2,1],[1,3]])\n```\n\nI hope it '
                        'makes sense. Happy to clarify if you need.'
                AFTER : '```python\nimport numpy as np\n'
                        'np.linalg.eig([[2,1],[1,3]])\n``` '
                        'Happy to clarify if you need.'

            [4] role=user      type=image  len=13  -> 13   (multimodal, protected)
                BEFORE: '/tmp/plot.png'
                AFTER : '/tmp/plot.png'

        Observations:
          * The chunk count stays at 5 -- nothing is dropped; each text chunk
            is simply shortened to roughly ``keep_ratio`` of its original
            length.
          * Filler phrases ("This matters for my thesis.", "Thanks in
            advance.", "Let me think about this carefully.", "Here is my
            code:", "I hope it makes sense.") are pruned because their TF-IDF
            mass is low.
          * Core sentences carrying distinctive terms (``eigenvalues``,
            ``characteristic equation``, ``det(A - lambda I) = 0``) survive.
          * The fenced ``` ``` ``` code block in ``[3]`` is treated as one
            atomic unit and scored ``+inf``, so it stays intact even when
            surrounding prose is trimmed.
          * Structural (``tool_call``) and multi-modal (``image``) chunks
            pass through verbatim; their ``raw`` metadata is unchanged.
          * Surviving sentences are emitted in original order, so the
            shortened text still reads as a coherent narrative.
    """

    def __init__(
        self,
        keep_ratio: RatioLike = 0.5,
        min_sentences: int = 1,
        min_chars: int = 40,
    ) -> None:
        low, high = self._normalize_ratio(keep_ratio)
        if min_sentences < 1 or min_chars < 0:
            raise ValueError(
                f'invalid params: min_sentences={min_sentences} (expect >=1), '
                f'min_chars={min_chars} (expect >=0)')
        self.min_keep_ratio = low
        self.max_keep_ratio = high
        # Backward-compat attribute: equals ``max_keep_ratio`` when uniform.
        self.keep_ratio = high if low == high else (low, high)
        self.min_sentences = min_sentences
        self.min_chars = min_chars

    @staticmethod
    def _normalize_ratio(keep_ratio: RatioLike) -> Tuple[float, float]:
        """Coerce ``keep_ratio`` to a ``(low, high)`` tuple and validate."""
        if isinstance(keep_ratio, (tuple, list)):
            if len(keep_ratio) != 2:
                raise ValueError(
                    f'invalid keep_ratio={keep_ratio!r}: expect 2-element '
                    f'(min, max) tuple')
            low, high = float(keep_ratio[0]), float(keep_ratio[1])
        else:
            low = high = float(keep_ratio)
        if not (0 < low <= 1) or not (0 < high <= 1) or low > high:
            raise ValueError(
                f'invalid keep_ratio={keep_ratio!r}: expect float in (0, 1] '
                f'or (min, max) with 0 < min <= max <= 1')
        return low, high

    # ── Public API ────────────────────────────────────────────────────────────

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        if 'keep_ratio' in kwargs:
            low, high = self._normalize_ratio(kwargs['keep_ratio'])
        else:
            low, high = self.min_keep_ratio, self.max_keep_ratio
        min_sentences = kwargs.get('min_sentences', self.min_sentences)
        min_chars = kwargs.get('min_chars', self.min_chars)

        items = list(chunks.chunks)
        if not items or (low >= 1.0 and high >= 1.0):
            return Chunks(chunks=items)

        # Locate compressible chunks; the gradient spans only these positions.
        compressible_indices = [
            i for i, c in enumerate(items) if self._is_compressible(c)
        ]
        idf = self._compute_idf(
            [items[i].get('content', '') for i in compressible_indices])

        # Linear interpolation across compressible chunks:
        #   rank 0 -> low  (earliest, most aggressive compression),
        #   rank N-1 -> high (latest, most preserved).
        # Non-compressible chunks get ratio 1.0 (a no-op; they are skipped by
        # ``_try_compress`` anyway, but keeps the call site uniform).
        ratio_by_index: Dict[int, float] = {}
        total = len(compressible_indices)
        for rank, idx in enumerate(compressible_indices):
            if total <= 1:
                ratio_by_index[idx] = high
            else:
                t = rank / (total - 1)
                ratio_by_index[idx] = low + (high - low) * t

        return Chunks(chunks=[
            self._try_compress(c, idf, ratio_by_index.get(i, 1.0),
                               min_sentences, min_chars)
            for i, c in enumerate(items)
        ])

    # ── Eligibility & rewriting ──────────────────────────────────────────────

    @staticmethod
    def _is_compressible(chunk: Chunk) -> bool:
        """True iff chunk is non-empty text and not structurally protected."""
        if chunk.get('type') in _PROTECTED_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
            return False
        content = chunk.get('content')
        return isinstance(content, str) and bool(content.strip())

    @classmethod
    def _try_compress(cls, chunk: Chunk, idf: Dict[str, float],
                      keep_ratio: float, min_sentences: int, min_chars: int) -> Chunk:
        """Compress ``chunk`` if eligible; otherwise return it unchanged."""
        content = chunk.get('content', '')
        if not cls._is_compressible(chunk) or len(content) <= min_chars:
            return chunk
        compressed = cls._compress_text(content, idf, keep_ratio, min_sentences)
        if not compressed or compressed == content:
            return chunk
        new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
        new_chunk['content'] = compressed
        raw = new_chunk.get('raw')
        if isinstance(raw, dict):
            new_chunk['raw'] = {**raw, 'condensed': True}
        return new_chunk

    # ── TF-IDF ────────────────────────────────────────────────────────────────

    @classmethod
    def _compute_idf(cls, docs: Sequence[str]) -> Dict[str, float]:
        """Smoothed IDF: ``log((N + 1) / (df + 1)) + 1``."""
        n = max(1, len(docs))
        df: Counter = Counter()
        for doc in docs:
            df.update(set(cls._tokenize(doc)))
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    @classmethod
    def _score_unit(cls, text: str, idf: Dict[str, float]) -> float:
        """Sub-linear TF × IDF over unique terms; OOV terms get weight 1.0."""
        tf = Counter(cls._tokenize(text))
        return sum((1 + math.log(c)) * idf.get(t, 1.0) for t, c in tf.items())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercased bilingual tokenizer (Latin words + CJK unigrams)."""
        return _TOKEN_RE.findall(text.lower())

    # ── Per-chunk compression ─────────────────────────────────────────────────

    @classmethod
    def _compress_text(cls, text: str, idf: Dict[str, float],
                       keep_ratio: float, min_sentences: int) -> str:
        """Greedy extractive compression respecting code fences and ordering."""
        units = cls._split_into_units(text)
        if len(units) <= min_sentences:
            return text

        target = max(1, math.ceil(len(text) * keep_ratio))
        # Code blocks get +inf so they are picked first and never dropped.
        scored = [
            (math.inf if is_code else cls._score_unit(u, idf), i, u)
            for i, (u, is_code) in enumerate(units)
        ]

        # Pick units by descending score until both budget and min are satisfied.
        selected: set = set()
        used = 0
        for _, i, u in sorted(scored, key=lambda x: (-x[0], x[1])):
            selected.add(i)
            used += len(u) + 1  # +1 for the joining space
            if used >= target and len(selected) >= min_sentences:
                break
        return ' '.join(u for i, (u, _) in enumerate(units) if i in selected).strip()

    @staticmethod
    def _split_into_units(text: str) -> List[Tuple[str, bool]]:
        """Yield ``(unit, is_code_block)`` preserving order; code stays atomic."""
        units: List[Tuple[str, bool]] = []
        for part in _CODE_FENCE_RE.split(text):
            if not part:
                continue
            if part.startswith('```'):
                units.append((part.strip(), True))
            else:
                units.extend((s.strip(), False) for s in _SENT_SPLIT_RE.split(part) if s.strip())
        return units
