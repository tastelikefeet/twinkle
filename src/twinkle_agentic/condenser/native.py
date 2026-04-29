# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native TF-IDF based per-chunk condenser.

Compresses each chunk's text to roughly ``keep_ratio`` of its original
character length by dropping the least informative sentences, while keeping
chunk count, role / type / raw metadata, fenced code blocks (atomic), and
structural / multi-modal chunks verbatim.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser

RatioLike = Union[float, Tuple[float, float], List[float]]


# ── Regex helpers ────────────────────────────────────────────────────────────
# Multilingual tokenizer: ASCII + accented Latin / digit words as one token,
# plus single-char tokens for CJK Unified Ideographs, Japanese kana
# (Hiragana / Katakana) and Korean Hangul.  Single-char tokenization for
# non-space-segmented scripts mirrors a unigram baseline -- coarse but
# language-agnostic, so query-aware scoring still fires on zh / ja / ko
# corpora without a language-specific tokenizer dependency.
_TOKEN_RE = re.compile(
    r'[A-Za-z0-9_\u00c0-\u024f]+'
    r'|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
# Fenced code blocks stay atomic under all splitting.
_CODE_FENCE_RE = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
# ``<block_N>...</block_N>`` spans stay atomic too: their internal
# punctuation must never trigger a sub-sentence split that tears the
# closing tag off from the body.  When a tool response produced by
# ``ExtractCompressed`` re-enters the condenser, splitting through a
# tag pair would leave half of the block under one scoring unit and
# the other half under the next, letting the condenser drop either
# piece and corrupt the markup downstream.
_BLOCK_TAG_RE = re.compile(r'(<block_\d+>[\s\S]*?</block_\d+>)')
# Ordered union of atomic regions -- same role as ``_ATOMIC_SPAN_RE`` in
# the chunker: isolate these before punctuation-level splitting.
_ATOMIC_SPAN_RE = re.compile(
    r'(```[\s\S]*?```|<block_\d+>[\s\S]*?</block_\d+>)', re.MULTILINE)
# Segment-header anchors scored +inf: ``[N] Title...`` + markdown headings.
# Negative lookbehind avoids false positives on inline citations like
# ``footnote[3]``; the uppercase/digit tail rules out ``[3] the text``.
_ANCHOR_RE = re.compile(r'(?<![A-Za-z])\[\d+\]\s+[A-Z0-9]|(?:^|\n)\s*#{1,6}\s')
# Over-aggressive sub-sentence split (ASCII + CJK punctuation + hard newline);
# ``_merge_fragments`` re-combines short fragments afterwards.
_PUNCT_SPLIT_RE = re.compile(
    r'(?<=[.!?;])\s+|(?<=,)\s+|(?<=[。！？；，])|\n+')
# Subordinate-clause openers that must glue back to their antecedent.
_SUBORDINATE_START_RE = re.compile(
    r'^(which|that|who|whom|whose|where|when)\b', re.IGNORECASE)

_PROTECTED_KINDS: Tuple[str, ...] = ('tool_call',)
_PROTECTED_TYPES: Tuple[str, ...] = ('image', 'video', 'audio')

# Grammatical-noise stop list for query-aware boosting.  Three groups:
#   1. English function words (articles, auxiliaries, prepositions, ...).
#   2. English interrogatives (``what`` / ``when`` / ``where`` / ...).
#      These frame the question without carrying the factual target and
#      would otherwise force-keep every passage that merely echoes the
#      question's phrasing (e.g. every biography contains ``who``).
#   3. Common single-character CJK particles.  Under the unigram CJK
#      tokenizer these appear in virtually every Chinese passage; leaving
#      them in would neutralise query-awareness on zh corpora the same way
#      English articles would on en corpora.
# Content-bearing words (``first``, ``started``, ...) intentionally stay
# in: corpus-wide commonality is filtered adaptively via IDF inside
# :meth:`NativeCondenser._query_tokens` (``query_token_max_df``).
_QUERY_STOP_TOKENS: frozenset = frozenset([
    # English function words
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'of', 'to', 'in', 'on', 'at', 'by', 'as', 'for', 'from', 'with',
    'and', 'or', 'but', 'if', 'then', 'so',
    'it', 'its', 'this', 'that', 'these', 'those', 'such',
    'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'must', 'not', 'no',
    's', 't', 'd', 'm', 'll', 're', 've',
    # English interrogatives (question-framing, not content)
    'who', 'whom', 'whose', 'what', 'which', 'when', 'where', 'why', 'how',
    # CJK structural particles / common function chars
    '的', '了', '是', '在', '和', '与', '或', '及', '有', '为', '被',
    '这', '那', '其', '之', '也', '都', '就', '还', '又', '而',
    '不', '没', '要', '会', '能', '吗', '呢', '吧', '啊', '哦',
])


class NativeCondenser(Condenser):
    """Extractive per-chunk TF-IDF condenser.

    Args:
        keep_ratio: Target char ratio per chunk in ``(0, 1]``.  Float =
            uniform; ``(min, max)`` tuple = linear gradient across
            compressible chunks (oldest → ``min``, newest → ``max``).
        min_sentences: Minimum sentences retained per chunk regardless of
            budget.  Defaults to ``1``.
        min_chars: Chunks shorter than this are kept verbatim.  Defaults
            to ``40``.
        min_unit_chars: Weighted-length threshold below which a fragment is
            merged into a neighbour (CJK chars count ×3).  Defaults to ``20``.
        skip_system: System-role chunks pass through verbatim.  Defaults
            to ``True``.
        strategy: ``'tfidf'`` (default) or ``'random'``.  Anchors & code
            fences are ``+inf`` under both.
        query_boost: Additive score bonus per distinct query-token match,
            scaled by IDF.  ``0.0`` disables query-aware scoring.
            Defaults to ``2.0``.
        query_keep_cap: Max non-anchor units per chunk force-kept due to
            query overlap, independent of budget.  Defaults to ``2``.
        query_token_max_df: Query tokens whose corpus document-frequency
            exceeds this ratio are dropped as non-discriminative (e.g.
            ``started`` / ``first`` that echo the question's framing in
            every distractor passage).  Defaults to ``0.5``.

    Example:
        >>> condensed = NativeCondenser(keep_ratio=0.5).condense(chunks)
        # Each text chunk shortened; count and metadata preserved.
    """

    def __init__(
        self,
        keep_ratio: RatioLike = 0.5,
        min_sentences: int = 1,
        min_chars: int = 40,
        min_unit_chars: int = 20,
        skip_system: bool = True,
        strategy: str = 'tfidf',
        query_boost: float = 2.0,
        query_keep_cap: int = 2,
        query_token_max_df: float = 0.5,
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
        if query_boost < 0 or query_keep_cap < 0:
            raise ValueError(
                f'invalid params: query_boost={query_boost} (expect >=0), '
                f'query_keep_cap={query_keep_cap} (expect >=0)')
        if not 0 < query_token_max_df <= 1:
            raise ValueError(
                f'invalid query_token_max_df={query_token_max_df}: '
                f'expect float in (0, 1]')
        self.min_keep_ratio = low
        self.max_keep_ratio = high
        # Backward-compat alias: equals ``max_keep_ratio`` when uniform.
        self.keep_ratio = high if low == high else (low, high)
        self.min_sentences = min_sentences
        self.min_chars = min_chars
        self.min_unit_chars = min_unit_chars
        self.skip_system = skip_system
        self.strategy = strategy
        self.query_boost = query_boost
        self.query_keep_cap = query_keep_cap
        self.query_token_max_df = query_token_max_df

    # ── Public API ───────────────────────────────────────────────────────────

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        cfg = self._resolve_kwargs(kwargs)
        items = list(chunks.chunks)
        if not items or (cfg['low'] >= 1.0 and cfg['high'] >= 1.0):
            return Chunks(chunks=items)

        compressible_indices = [
            i for i, c in enumerate(items)
            if self._is_compressible(c, skip_system=cfg['skip_system'])
        ]
        compressible_docs = [items[i].get('content', '')
                             for i in compressible_indices]
        idf = (self._compute_idf(compressible_docs)
               if cfg['strategy'] == 'tfidf' else {})
        query_tokens = self._query_tokens(
            cfg['query_hint'], idf=idf,
            corpus_size=len(compressible_docs),
            max_df_ratio=cfg['query_token_max_df'])
        ratio_by_index = self._compute_ratio_gradient(
            compressible_indices, cfg['low'], cfg['high'])

        return Chunks(chunks=[
            self._try_compress(
                c, idf, ratio_by_index.get(i, 1.0),
                cfg['min_sentences'], cfg['min_chars'], cfg['min_unit_chars'],
                skip_system=cfg['skip_system'], strategy=cfg['strategy'],
                query_tokens=query_tokens, query_boost=cfg['query_boost'],
                query_keep_cap=cfg['query_keep_cap'])
            for i, c in enumerate(items)
        ])

    # ── Config / eligibility ─────────────────────────────────────────────────

    @staticmethod
    def _normalize_ratio(keep_ratio: RatioLike) -> Tuple[float, float]:
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

    def _resolve_kwargs(self, kwargs: Dict) -> Dict:
        """Merge per-call overrides with instance defaults."""
        if 'keep_ratio' in kwargs:
            low, high = self._normalize_ratio(kwargs['keep_ratio'])
        else:
            low, high = self.min_keep_ratio, self.max_keep_ratio
        strategy = kwargs.get('strategy', self.strategy)
        if strategy not in ('tfidf', 'random'):
            raise ValueError(
                f'invalid strategy={strategy!r}: expect "tfidf" or "random"')
        return {
            'low': low, 'high': high,
            'min_sentences': kwargs.get('min_sentences', self.min_sentences),
            'min_chars': kwargs.get('min_chars', self.min_chars),
            'min_unit_chars': kwargs.get('min_unit_chars', self.min_unit_chars),
            'skip_system': kwargs.get('skip_system', self.skip_system),
            'strategy': strategy,
            'query_boost': kwargs.get('query_boost', self.query_boost),
            'query_keep_cap': kwargs.get(
                'query_keep_cap', self.query_keep_cap),
            'query_token_max_df': kwargs.get(
                'query_token_max_df', self.query_token_max_df),
            'query_hint': kwargs.get('query_hint', ''),
        }

    @staticmethod
    def _compute_ratio_gradient(compressible_indices: List[int],
                                 low: float,
                                 high: float) -> Dict[int, float]:
        """Linear interpolation: rank 0 → low, rank N-1 → high."""
        out: Dict[int, float] = {}
        total = len(compressible_indices)
        for rank, idx in enumerate(compressible_indices):
            if total <= 1:
                out[idx] = high
            else:
                t = rank / (total - 1)
                out[idx] = low + (high - low) * t
        return out

    @staticmethod
    def _is_compressible(chunk: Chunk, skip_system: bool = True) -> bool:
        if skip_system and chunk.get('role') == 'system':
            return False
        if chunk.get('type') in _PROTECTED_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
            return False
        content = chunk.get('content')
        return isinstance(content, str) and bool(content.strip())

    # ── TF-IDF corpus ────────────────────────────────────────────────────────

    @classmethod
    def _compute_idf(cls, docs: Sequence[str]) -> Dict[str, float]:
        """Smoothed IDF: ``log((N + 1) / (df + 1)) + 1``."""
        n = max(1, len(docs))
        df: Counter = Counter()
        for doc in docs:
            df.update(set(cls._tokenize(doc)))
        return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _TOKEN_RE.findall(text.lower())

    @classmethod
    def _score_unit(cls, text: str, idf: Dict[str, float]) -> float:
        """Sub-linear TF × IDF; OOV terms get weight ``1.0``."""
        tf = Counter(cls._tokenize(text))
        return sum((1 + math.log(c)) * idf.get(t, 1.0) for t, c in tf.items())

    # ── Query awareness ──────────────────────────────────────────────────────

    @classmethod
    def _query_tokens(cls, query_hint: str,
                       idf: Optional[Dict[str, float]] = None,
                       corpus_size: int = 0,
                       max_df_ratio: float = 0.5) -> Set[str]:
        """Distill a question into discriminative content-bearing tokens.

        Filter order:
          1. ``_QUERY_STOP_TOKENS`` drops purely grammatical noise.
          2. Corpus-IDF guard drops tokens whose document-frequency ratio
             exceeds ``max_df_ratio``.  Without this, generic question
             verbs (``started`` / ``first``) would match nearly every
             distractor passage and trigger force-keep everywhere,
             destroying compression.  Tokens absent from the corpus are
             trivially discriminative and kept.
        """
        if not query_hint or not query_hint.strip():
            return set()
        tokens = {t for t in cls._tokenize(query_hint)
                  if t not in _QUERY_STOP_TOKENS}
        if not tokens or not idf or corpus_size <= 0:
            return tokens
        # Recover df from smoothed IDF:
        #   idf = log((N+1)/(df+1)) + 1
        #   => df = (N+1) / exp(idf - 1) - 1
        max_df = max_df_ratio * corpus_size
        return {
            t for t in tokens
            if t not in idf
            or ((corpus_size + 1) / math.exp(idf[t] - 1) - 1) <= max_df
        }

    # ── Per-chunk compression ────────────────────────────────────────────────

    @classmethod
    def _try_compress(cls, chunk: Chunk, idf: Dict[str, float],
                      keep_ratio: float, min_sentences: int, min_chars: int,
                      min_unit_chars: int, *,
                      skip_system: bool = True,
                      strategy: str = 'tfidf',
                      query_tokens: Optional[Set[str]] = None,
                      query_boost: float = 0.0,
                      query_keep_cap: int = 0) -> Chunk:
        """Compress ``chunk`` if eligible; otherwise return unchanged."""
        content = chunk.get('content', '')
        if (not cls._is_compressible(chunk, skip_system=skip_system)
                or len(content) <= min_chars):
            return chunk
        compressed = cls._compress_text(
            content, idf, keep_ratio, min_sentences, min_unit_chars,
            strategy=strategy, query_tokens=query_tokens,
            query_boost=query_boost, query_keep_cap=query_keep_cap)
        if not compressed or compressed == content:
            return chunk
        new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
        new_chunk['content'] = compressed
        raw = new_chunk.get('raw')
        if isinstance(raw, dict):
            new_chunk['raw'] = {**raw, 'condensed': True}
        return new_chunk

    @classmethod
    def _compress_text(cls, text: str, idf: Dict[str, float],
                       keep_ratio: float, min_sentences: int,
                       min_unit_chars: int, *,
                       strategy: str = 'tfidf',
                       query_tokens: Optional[Set[str]] = None,
                       query_boost: float = 0.0,
                       query_keep_cap: int = 0) -> str:
        """Split → score → two-pass select → reassemble in original order."""
        units = cls._split_into_units(text, min_unit_chars)
        if len(units) <= min_sentences:
            return text

        qtokens = query_tokens or set()
        unit_tokens: List[Set[str]] = [
            set(cls._tokenize(u)) if not is_code else set()
            for u, is_code in units
        ]
        scored = cls._score_units(
            units, unit_tokens, idf, strategy, qtokens, query_boost)
        forced = cls._pick_forced(
            scored, unit_tokens, qtokens, query_boost, query_keep_cap)
        target = max(1, math.ceil(len(text) * keep_ratio))
        selected = cls._select_with_budget(
            scored, forced, target, min_sentences)
        return ' '.join(
            u for i, (u, _) in enumerate(units) if i in selected).strip()

    @classmethod
    def _score_units(cls, units: List[Tuple[str, bool]],
                      unit_tokens: List[Set[str]],
                      idf: Dict[str, float],
                      strategy: str,
                      qtokens: Set[str],
                      query_boost: float) -> List[Tuple[float, int, str]]:
        """Score each unit.  Anchors & code fences → ``+inf``."""
        scored: List[Tuple[float, int, str]] = []
        for i, (u, is_code) in enumerate(units):
            if is_code or _ANCHOR_RE.search(u):
                scored.append((math.inf, i, u))
                continue
            if strategy == 'random':
                scored.append((random.random(), i, u))
                continue
            base = cls._score_unit(u, idf)
            if qtokens and query_boost > 0:
                matched = qtokens & unit_tokens[i]
                if matched:
                    base += query_boost * sum(
                        idf.get(t, 1.0) for t in matched)
            scored.append((base, i, u))
        return scored

    @staticmethod
    def _pick_forced(scored: List[Tuple[float, int, str]],
                      unit_tokens: List[Set[str]],
                      qtokens: Set[str],
                      query_boost: float,
                      query_keep_cap: int) -> Set[int]:
        """Top-K query-matched non-anchor units, force-kept regardless of budget.

        Without this, a passage's anchor alone can saturate ``target`` and
        every body sentence gets dropped -- including the one carrying the
        answer-critical fact (e.g. ``started in 1989`` in ``First for Women``).
        """
        if not (qtokens and query_keep_cap > 0 and query_boost > 0):
            return set()
        ranked = sorted(
            (s for s in scored
             if s[0] != math.inf and (qtokens & unit_tokens[s[1]])),
            key=lambda x: (-x[0], x[1]))
        return {i for _, i, _ in ranked[:query_keep_cap]}

    @staticmethod
    def _select_with_budget(scored: List[Tuple[float, int, str]],
                             forced: Set[int],
                             target: int,
                             min_sentences: int) -> Set[int]:
        """Two-pass selection: must-keep first, then score-descending fill.

        Pass 1 adds anchors (+inf) and ``forced`` unconditionally -- a
        single descending-order loop with early ``break`` would skip
        forced items whose raw score is below the budget cutoff,
        re-introducing the fact-loss ``forced`` exists to prevent.
        """
        selected: Set[int] = set()
        used = 0
        for score, i, u in scored:
            if score == math.inf or i in forced:
                selected.add(i)
                used += len(u) + 1  # +1 for joining space
        for score, i, u in sorted(scored, key=lambda x: (-x[0], x[1])):
            if i in selected:
                continue
            if used >= target and len(selected) >= min_sentences:
                break
            selected.add(i)
            used += len(u) + 1
        return selected

    # ── Sentence splitting ───────────────────────────────────────────────────

    @staticmethod
    def _weighted_len(text: str) -> int:
        """Each CJK char counts as 3 ASCII chars (info-density parity)."""
        return sum(3 if '\u4e00' <= c <= '\u9fff' else 1 for c in text)

    @classmethod
    def _split_into_units(cls, text: str,
                          min_unit_chars: int = 20) -> List[Tuple[str, bool]]:
        """Yield ``(unit, is_atomic)``; code fences and ``<block_N>`` atomic.

        Splits aggressively at every punctuation boundary, then merges short
        fragments via :meth:`_merge_fragments` so appositives, enumerations
        and short dependent clauses are never scored in isolation.  Atomic
        spans (fenced code + ``<block_N>...</block_N>`` pairs) bypass the
        punctuation splitter entirely so their internal ``.``/``,`` cannot
        tear the span apart.  The ``is_atomic`` flag is used by the scorer
        to assign ``+inf`` (never drop) and by the tokenizer to skip token
        extraction (atomic units are not scored on their TF-IDF content).
        """
        units: List[Tuple[str, bool]] = []
        for part in _ATOMIC_SPAN_RE.split(text):
            if not part:
                continue
            if part.startswith('```') or _BLOCK_TAG_RE.fullmatch(part):
                units.append((part.strip(), True))
                continue
            frags = [s.strip() for s in _PUNCT_SPLIT_RE.split(part)
                     if s and s.strip()]
            units.extend(
                (m, False) for m in cls._merge_fragments(frags, min_unit_chars))
        return units

    @classmethod
    def _merge_fragments(cls, frags: Iterable[str],
                          min_unit_chars: int) -> List[str]:
        """Merge short fragments with three cohesion rules.

        * **Subordinate clauses** (``which``/``that``/``who``/...) glue
          to their antecedent so TF-IDF cannot keep a dangling clause
          while dropping the main clause that grounds it.
        * **Short interstitials** with a prior unit glue *backward*
          rather than accumulating forward -- otherwise a fragment like
          ``New Jersey.`` gets prepended to the next sentence and
          produces a misleading sandwich when force-keep retains that
          neighbour.
        * **Trailing short buffer** attaches to the previous unit.
        """
        merged: List[str] = []
        buf = ''
        for f in frags:
            if not buf and merged and (
                    _SUBORDINATE_START_RE.match(f)
                    or cls._weighted_len(f) < min_unit_chars):
                merged[-1] = f'{merged[-1]} {f}'
                continue
            buf = f'{buf} {f}' if buf else f
            if cls._weighted_len(buf) >= min_unit_chars:
                merged.append(buf)
                buf = ''
        if buf:
            if merged:
                merged[-1] = f'{merged[-1]} {buf}'
            else:
                merged.append(buf)
        return merged
