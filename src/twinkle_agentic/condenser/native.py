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
import random
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
# Segment-header anchors the scorer must NEVER drop:
#   - ``[N] Title...``  -> HotpotQA / retrieval passage numbering
#   - ``# ... / ## ... / ### ...`` -> markdown headings
# The anchor is the model's *only* reliable hook for deciding "this block
# might matter; let me call ``extract_compressed`` to recall it".  Without
# preserving it, a passage whose body sentences all got scored low becomes an
# invisible hole -- the model can't extract what it doesn't know is there.
#
# We ``search`` rather than ``match`` because ``_split_into_units`` merges
# short leading fragments (e.g. ``Context:``) into the next unit, so ``[1]``
# is often not at position 0.  To avoid false positives on inline citations
# (``as shown in [3] the text``), require ``[N]`` be followed by an uppercase
# letter OR a digit -- the two real-world passage-title prefixes in corpora
# like HotpotQA (``[7] 2014-15 Ukrainian Hockey Championship`` starts with a
# digit).  The negative lookbehind ``(?<![A-Za-z])`` further rules out
# accidental matches like ``footnote[3]``.
_ANCHOR_RE = re.compile(r'(?<![A-Za-z])\[\d+\]\s+[A-Z0-9]|(?:^|\n)\s*#{1,6}\s')
# Candidate sub-sentence boundaries (language-neutral; no conjunction list):
#   - ASCII strong punctuation / comma / semicolon followed by whitespace
#     (requiring whitespace preserves decimals like ``3.14`` and number
#     groupings like ``1,000`` -- they have no following space, so never split)
#   - CJK punctuation (。！？；，): zero-width split, since CJK text has
#     no reliable whitespace around punctuation
#   - Hard newlines
# Granularity is intentionally over-aggressive here; ``_split_into_units``
# merges fragments shorter than ``min_unit_chars`` back into a neighbour so
# appositives (``Barack Obama, the 44th president,``), number enumerations
# (``apples, bananas, oranges``), and short dependent clauses
# (``Despite the rain,``) are never dropped in isolation.
_PUNCT_SPLIT_RE = re.compile(
    r'(?<=[.!?;])\s+'
    r'|(?<=,)\s+'
    r'|(?<=[。！？；，])'
    r'|\n+',
)

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
        min_unit_chars: Weighted-length threshold below which a fragment
            produced by punctuation splitting is merged back into a
            neighbour (CJK chars count ×3). Defaults to ``20``.
        skip_system: When ``True`` (default), chunks whose ``role`` is
            ``'system'`` pass through verbatim and are excluded from the
            gradient / IDF corpus. System prompts typically carry tool
            schemas and load-bearing instructions that should not be
            summarised. Set to ``False`` to also compress system chunks.
        strategy: Per-unit scoring policy. Must be one of:

            * ``'tfidf'`` (default) -- sub-linear TF x IDF over the
              trajectory corpus. Best when surviving sentences should carry
              the most distinctive tokens (works well for code / technical
              content).
            * ``'random'`` -- assign ``random.random()`` to each candidate
              unit. Useful as a query-agnostic baseline that avoids IDF's
              well-known bias toward rare-but-irrelevant tokens (e.g. in
              multi-hop QA where the rare tokens tend to sit in distractor
              passages). Pair with an external ``random.seed(...)`` call
              for reproducible rollouts.

            Segment-header anchors (``[N] ...`` / markdown ``#``) and
            fenced code blocks are scored ``+inf`` under *both* strategies
            so they always survive.

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
        min_unit_chars: int = 20,
        skip_system: bool = True,
        strategy: str = 'tfidf',
    ) -> None:
        low, high = self._normalize_ratio(keep_ratio)
        if min_sentences < 1 or min_chars < 0 or min_unit_chars < 0:
            raise ValueError(
                f'invalid params: min_sentences={min_sentences} (expect >=1), '
                f'min_chars={min_chars} (expect >=0), '
                f'min_unit_chars={min_unit_chars} (expect >=0)')
        if strategy not in ('tfidf', 'random'):
            raise ValueError(
                f'invalid strategy={strategy!r}: expect "tfidf" or "random"')
        self.min_keep_ratio = low
        self.max_keep_ratio = high
        # Backward-compat attribute: equals ``max_keep_ratio`` when uniform.
        self.keep_ratio = high if low == high else (low, high)
        self.min_sentences = min_sentences
        self.min_chars = min_chars
        # Fragments shorter than this after punct split are merged back into a
        # neighbour, so appositives / enumerations / short dependent clauses
        # are never scored (and possibly dropped) in isolation.
        self.min_unit_chars = min_unit_chars
        # System-role chunks typically carry tool schemas / instructions whose
        # wording is load-bearing; compressing them can silently break the
        # trajectory. Default to pass-through.
        self.skip_system = skip_system
        self.strategy = strategy

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
        min_unit_chars = kwargs.get('min_unit_chars', self.min_unit_chars)
        skip_system = kwargs.get('skip_system', self.skip_system)
        strategy = kwargs.get('strategy', self.strategy)
        if strategy not in ('tfidf', 'random'):
            raise ValueError(
                f'invalid strategy={strategy!r}: expect "tfidf" or "random"')

        items = list(chunks.chunks)
        if not items or (low >= 1.0 and high >= 1.0):
            return Chunks(chunks=items)

        # Locate compressible chunks; the gradient spans only these positions.
        compressible_indices = [
            i for i, c in enumerate(items)
            if self._is_compressible(c, skip_system=skip_system)
        ]
        # IDF is only needed for the tfidf strategy; the random strategy
        # skips corpus aggregation entirely (saves work on long rollouts).
        if strategy == 'tfidf':
            idf = self._compute_idf(
                [items[i].get('content', '') for i in compressible_indices])
        else:
            idf = {}

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
                               min_sentences, min_chars, min_unit_chars,
                               skip_system=skip_system, strategy=strategy)
            for i, c in enumerate(items)
        ])

    # ── Eligibility & rewriting ──────────────────────────────────────────────

    @staticmethod
    def _is_compressible(chunk: Chunk, skip_system: bool = True) -> bool:
        """True iff chunk is non-empty text and not structurally protected.

        When ``skip_system`` is ``True`` (default), chunks whose ``role`` is
        ``'system'`` are treated as protected and pass through verbatim.
        """
        if skip_system and chunk.get('role') == 'system':
            return False
        if chunk.get('type') in _PROTECTED_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
            return False
        content = chunk.get('content')
        return isinstance(content, str) and bool(content.strip())

    @classmethod
    def _try_compress(cls, chunk: Chunk, idf: Dict[str, float],
                      keep_ratio: float, min_sentences: int, min_chars: int,
                      min_unit_chars: int,
                      skip_system: bool = True,
                      strategy: str = 'tfidf') -> Chunk:
        """Compress ``chunk`` if eligible; otherwise return it unchanged."""
        content = chunk.get('content', '')
        if (not cls._is_compressible(chunk, skip_system=skip_system)
                or len(content) <= min_chars):
            return chunk
        compressed = cls._compress_text(content, idf, keep_ratio, min_sentences,
                                        min_unit_chars, strategy=strategy)
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
                       keep_ratio: float, min_sentences: int,
                       min_unit_chars: int,
                       strategy: str = 'tfidf') -> str:
        """Greedy extractive compression respecting code fences and ordering."""
        units = cls._split_into_units(text, min_unit_chars)
        if len(units) <= min_sentences:
            return text

        target = max(1, math.ceil(len(text) * keep_ratio))
        # Scoring priority (shared across strategies):
        #   1. Code fences & segment-header anchors  -> +inf (never dropped)
        #   2. Body units                            -> TF-IDF or random
        def _score(u: str, is_code: bool) -> float:
            if is_code or _ANCHOR_RE.search(u):
                return math.inf
            if strategy == 'random':
                return random.random()
            return cls._score_unit(u, idf)

        scored = [(_score(u, is_code), i, u)
                  for i, (u, is_code) in enumerate(units)]

        # Pick units by descending score until both budget and min are satisfied.
        # ``+inf`` units (anchors + code fences) are *always* kept, even when
        # doing so pushes ``used`` past ``target`` -- losing a late-index
        # anchor because an earlier one already saturated the budget would
        # recreate exactly the "invisible hole" we added anchors to prevent.
        selected: set = set()
        used = 0
        for score, i, u in sorted(scored, key=lambda x: (-x[0], x[1])):
            if score == math.inf:
                selected.add(i)
                used += len(u) + 1  # +1 for the joining space
                continue
            if used >= target and len(selected) >= min_sentences:
                break
            selected.add(i)
            used += len(u) + 1
        return ' '.join(u for i, (u, _) in enumerate(units) if i in selected).strip()

    @staticmethod
    def _weighted_len(text: str) -> int:
        """Length proxy for info density: each CJK char counts as 3 ASCII chars.

        Without weighting, ``min_unit_chars=20`` would never trip for Chinese
        text (7 Chinese chars ≈ a full short sentence), so the merge-back
        would collapse everything into one unit. Weighting CJK ×3 aligns the
        threshold across languages.
        """
        return sum(3 if '\u4e00' <= c <= '\u9fff' else 1 for c in text)

    @staticmethod
    def _split_into_units(text: str, min_unit_chars: int = 20) -> List[Tuple[str, bool]]:
        """Yield ``(unit, is_code_block)`` preserving order; code stays atomic.

        Splits aggressively at every punctuation boundary, then **merges
        fragments whose weighted length is below ``min_unit_chars`` back into
        a neighbour**. This replaces a hand-curated transition-conjunction
        list with a purely length-driven rule that handles appositives,
        enumerations, and short dependent clauses uniformly across languages
        (CJK chars count ×3 via :meth:`_weighted_len`).

        Merge policy: accumulate short fragments into a forward buffer until
        they reach ``min_unit_chars``; a trailing short fragment at the end is
        glued back onto the previous emitted unit.
        """
        units: List[Tuple[str, bool]] = []
        for part in _CODE_FENCE_RE.split(text):
            if not part:
                continue
            if part.startswith('```'):
                units.append((part.strip(), True))
                continue
            raw = [s.strip() for s in _PUNCT_SPLIT_RE.split(part) if s and s.strip()]
            if not raw:
                continue
            merged: List[str] = []
            buffer = ''
            for r in raw:
                combined = (buffer + ' ' + r).strip() if buffer else r
                if NativeCondenser._weighted_len(combined) < min_unit_chars:
                    buffer = combined
                else:
                    merged.append(combined)
                    buffer = ''
            if buffer:
                if merged:
                    merged[-1] = (merged[-1] + ' ' + buffer).strip()
                else:
                    merged.append(buffer)
            units.extend((m, False) for m in merged)
        return units
