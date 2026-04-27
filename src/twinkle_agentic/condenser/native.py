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
from typing import Dict, List, Sequence, Set, Tuple

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
            in ``(0.0, 1.0]``.  ``1.0`` disables compression.  Defaults to
            ``0.5``.
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
        keep_ratio: float = 0.5,
        min_sentences: int = 1,
        min_chars: int = 40,
    ) -> None:
        if not 0 < keep_ratio <= 1:
            raise ValueError(f'keep_ratio must be in (0, 1], got {keep_ratio}')
        if min_sentences < 1:
            raise ValueError(f'min_sentences must be >= 1, got {min_sentences}')
        if min_chars < 0:
            raise ValueError(f'min_chars must be >= 0, got {min_chars}')
        self.keep_ratio = keep_ratio
        self.min_sentences = min_sentences
        self.min_chars = min_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        keep_ratio = kwargs.get('keep_ratio', self.keep_ratio)
        min_sentences = kwargs.get('min_sentences', self.min_sentences)
        min_chars = kwargs.get('min_chars', self.min_chars)

        items = list(chunks.chunks)
        if not items or keep_ratio >= 1.0:
            return Chunks(chunks=items)

        # 1. Build corpus-wide IDF over all compressible text chunks.
        corpus = [c.get('content', '') for c in items if self._is_compressible(c)]
        idf = self._compute_idf(corpus)

        # 2. Compress each text chunk individually; pass-through the rest.
        result: List[Chunk] = []
        for chunk in items:
            if not self._is_compressible(chunk):
                result.append(chunk)
                continue
            content = chunk.get('content', '')
            if len(content) <= min_chars:
                result.append(chunk)
                continue
            compressed = self._compress_text(content, idf, keep_ratio, min_sentences)
            if compressed and compressed != content:
                result.append(self._replace_content(chunk, compressed))
            else:
                result.append(chunk)
        return Chunks(chunks=result)

    # ── Eligibility ──────────────────────────────────────────────────────────

    @staticmethod
    def _is_compressible(chunk: Chunk) -> bool:
        """A chunk is compressible iff it is text, non-empty, and not protected."""
        if chunk.get('type') in _PROTECTED_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
            return False
        content = chunk.get('content')
        return isinstance(content, str) and bool(content.strip())

    @staticmethod
    def _replace_content(chunk: Chunk, content: str) -> Chunk:
        """Return a shallow copy of ``chunk`` with a new ``content`` string."""
        new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
        new_chunk['content'] = content
        raw = new_chunk.get('raw')
        if isinstance(raw, dict):
            new_chunk['raw'] = {**raw, 'condensed': True}
        return new_chunk

    # ── TF-IDF ────────────────────────────────────────────────────────────────

    @classmethod
    def _compute_idf(cls, docs: Sequence[str]) -> Dict[str, float]:
        """Smoothed IDF: ``log((N + 1) / (df + 1)) + 1``."""
        n_docs = max(1, len(docs))
        df: Counter = Counter()
        for doc in docs:
            df.update(set(cls._tokenize(doc)))
        return {term: math.log((n_docs + 1) / (count + 1)) + 1 for term, count in df.items()}

    @classmethod
    def _score_unit(cls, text: str, idf: Dict[str, float]) -> float:
        """Sub-linear TF × IDF aggregation over unique terms.

        Terms unseen in ``idf`` receive a minimal default weight so that
        rare-but-novel vocabulary does not get zero credit.
        """
        tokens = cls._tokenize(text)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        default = 1.0
        return sum((1 + math.log(count)) * idf.get(term, default) for term, count in tf.items())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercased bilingual tokenizer (Latin words + CJK unigrams)."""
        return _TOKEN_RE.findall(text.lower())

    # ── Per-chunk compression ─────────────────────────────────────────────────

    @classmethod
    def _compress_text(
        cls,
        text: str,
        idf: Dict[str, float],
        keep_ratio: float,
        min_sentences: int,
    ) -> str:
        """Extractive compression that respects code fences and ordering."""
        units = cls._split_into_units(text)
        if len(units) <= min_sentences:
            return text

        target_chars = max(1, math.ceil(len(text) * keep_ratio))
        # Code blocks get +inf so they are picked first and never dropped.
        scored = [
            (math.inf if is_code else cls._score_unit(unit, idf), i, unit)
            for i, (unit, is_code) in enumerate(units)
        ]

        # Greedy selection by score descending; tie-broken by original order.
        selected: Set[int] = set()
        used = 0
        for score, i, unit in sorted(scored, key=lambda x: (-x[0], x[1])):
            selected.add(i)
            used += len(unit) + 1  # +1 for the joining space
            if used >= target_chars and len(selected) >= min_sentences:
                break

        parts = [unit for i, (unit, _) in enumerate(units) if i in selected]
        return ' '.join(parts).strip()

    @staticmethod
    def _split_into_units(text: str) -> List[Tuple[str, bool]]:
        """Split content into ``(unit_text, is_code_block)`` preserving order.

        Fenced code blocks remain single indivisible units; non-code parts
        are further split on sentence boundaries.
        """
        units: List[Tuple[str, bool]] = []
        for part in _CODE_FENCE_RE.split(text):
            if not part:
                continue
            if part.startswith('```'):
                units.append((part.strip(), True))
                continue
            for sent in _SENT_SPLIT_RE.split(part):
                sent = sent.strip()
                if sent:
                    units.append((sent, False))
        return units
