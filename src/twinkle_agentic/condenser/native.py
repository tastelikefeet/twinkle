# Copyright (c) ModelScope Contributors. All rights reserved.
"""Native TF-IDF based per-chunk condenser.

Compresses each chunk's text by dropping low-scoring sentences while keeping
code fences, block tags, and structural chunks verbatim.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser

# Multilingual tokenizer: ASCII words + single CJK/kana/hangul chars.
_TOKEN_RE = re.compile(
    r'[A-Za-z0-9_\u00c0-\u024f]+'
    r'|[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]')
# Atomic spans that must never be split.
_ATOMIC_SPAN_RE = re.compile(
    r'(```[\s\S]*?```|<block_\d+>[\s\S]*?</block_\d+>)', re.MULTILINE)
_BLOCK_TAG_RE = re.compile(r'(<block_\d+>[\s\S]*?</block_\d+>)')
# Segment-header anchors: [N] Title... or markdown headings.
_ANCHOR_RE = re.compile(r'(?<![A-Za-z])\[\d+\]\s+[A-Z0-9]|(?:^|\n)\s*#{1,6}\s')
# Punctuation-based sub-sentence split.
_PUNCT_SPLIT_RE = re.compile(
    r'(?<=[.!?;])\s+|(?<=,)\s+|(?<=[。！？；，])|\n+')

_PROTECTED_KINDS: Tuple[str, ...] = ('tool_call',)
_PROTECTED_TYPES: Tuple[str, ...] = ('image', 'video', 'audio')

_QUERY_STOP_TOKENS: frozenset = frozenset([
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'of', 'to', 'in', 'on', 'at', 'by', 'as', 'for', 'from', 'with',
    'and', 'or', 'but', 'if', 'then', 'so',
    'it', 'its', 'this', 'that', 'these', 'those', 'such',
    'do', 'does', 'did', 'have', 'has', 'had', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'must', 'not', 'no',
    's', 't', 'd', 'm', 'll', 're', 've',
    'who', 'whom', 'whose', 'what', 'which', 'when', 'where', 'why', 'how',
    '的', '了', '是', '在', '和', '与', '或', '及', '有', '为', '被',
    '这', '那', '其', '之', '也', '都', '就', '还', '又', '而',
    '不', '没', '要', '会', '能', '吗', '呢', '吧', '啊', '哦',
])


class NativeCondenser(Condenser):
    """Extractive per-chunk TF-IDF condenser with optional query-awareness.

    Args:
        keep_ratio: Target char ratio per chunk in (0, 1].
        min_sentences: Minimum sentences retained per chunk.
        min_chars: Chunks shorter than this are kept verbatim.
        query_boost: Additive score bonus per query-token match. 0 disables.
    """

    def __init__(self, keep_ratio: float = 0.5, min_sentences: int = 1,
                 min_chars: int = 40, min_unit_chars: int = 20,
                 skip_system: bool = True, query_boost: float = 2.0) -> None:
        if not 0 < keep_ratio <= 1:
            raise ValueError(f'keep_ratio must be in (0, 1], got {keep_ratio}')
        self.keep_ratio = keep_ratio
        self.min_sentences = max(1, min_sentences)
        self.min_chars = min_chars
        self.min_unit_chars = min_unit_chars
        self.skip_system = skip_system
        self.query_boost = query_boost

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        items = list(chunks.chunks)
        if not items or self.keep_ratio >= 1.0:
            return Chunks(chunks=items)

        # Build corpus IDF from compressible chunks
        compressible_docs = [
            c.get('content', '') for c in items if self._is_compressible(c)]
        idf = self._compute_idf(compressible_docs)

        query_hint = kwargs.get('query_hint', '')
        query_tokens = self._query_tokens(query_hint) if query_hint else set()

        return Chunks(chunks=[
            self._try_compress(c, idf, query_tokens) for c in items
        ])

    # ── Eligibility ───────────────────────────────────────────────────────────

    def _is_compressible(self, chunk: Chunk) -> bool:
        if self.skip_system and chunk.get('role') == 'system':
            return False
        if chunk.get('type') in _PROTECTED_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
            return False
        content = chunk.get('content')
        return isinstance(content, str) and bool(content.strip())

    # ── TF-IDF ────────────────────────────────────────────────────────────────

    @classmethod
    def _compute_idf(cls, docs: Sequence[str]) -> Dict[str, float]:
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
        tf = Counter(cls._tokenize(text))
        return sum((1 + math.log(c)) * idf.get(t, 1.0) for t, c in tf.items())

    @classmethod
    def _query_tokens(cls, query_hint: str) -> Set[str]:
        if not query_hint.strip():
            return set()
        return {t for t in cls._tokenize(query_hint) if t not in _QUERY_STOP_TOKENS}

    # ── Per-chunk compression ─────────────────────────────────────────────────

    def _try_compress(self, chunk: Chunk, idf: Dict[str, float],
                      query_tokens: Set[str]) -> Chunk:
        content = chunk.get('content', '')
        if not self._is_compressible(chunk) or len(content) <= self.min_chars:
            return chunk
        compressed = self._compress_text(content, idf, query_tokens)
        if not compressed or compressed == content:
            return chunk
        new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
        new_chunk['content'] = compressed
        raw = new_chunk.get('raw')
        if isinstance(raw, dict):
            new_chunk['raw'] = {**raw, 'condensed': True}
        return new_chunk

    def _compress_text(self, text: str, idf: Dict[str, float],
                       query_tokens: Set[str]) -> str:
        """Split → score → select top units within budget."""
        units = self._split_into_units(text)
        if len(units) <= self.min_sentences:
            return text

        # Score each unit
        scored: List[Tuple[float, int, str]] = []
        for i, (u, is_atomic) in enumerate(units):
            if is_atomic or _ANCHOR_RE.search(u):
                scored.append((math.inf, i, u))
                continue
            base = self._score_unit(u, idf)
            if query_tokens and self.query_boost > 0:
                matched = query_tokens & set(self._tokenize(u))
                if matched:
                    base += self.query_boost * sum(idf.get(t, 1.0) for t in matched)
            scored.append((base, i, u))

        # Select: anchors always kept, then fill by score until budget
        target = max(1, math.ceil(len(text) * self.keep_ratio))
        selected: Set[int] = set()
        used = 0
        # Pass 1: must-keep (anchors/atomic)
        for score, i, u in scored:
            if score == math.inf:
                selected.add(i)
                used += len(u) + 1
        # Pass 2: fill by descending score
        for score, i, u in sorted(scored, key=lambda x: (-x[0], x[1])):
            if i in selected:
                continue
            if used >= target and len(selected) >= self.min_sentences:
                break
            selected.add(i)
            used += len(u) + 1

        return ' '.join(u for i, (u, _) in enumerate(units) if i in selected).strip()

    # ── Sentence splitting ────────────────────────────────────────────────────

    def _split_into_units(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into (unit, is_atomic) pairs."""
        units: List[Tuple[str, bool]] = []
        for part in _ATOMIC_SPAN_RE.split(text):
            if not part:
                continue
            if part.startswith('```') or _BLOCK_TAG_RE.fullmatch(part):
                units.append((part.strip(), True))
                continue
            frags = [s.strip() for s in _PUNCT_SPLIT_RE.split(part) if s and s.strip()]
            units.extend((m, False) for m in self._merge_fragments(frags))
        return units

    def _merge_fragments(self, frags: Iterable[str]) -> List[str]:
        """Merge short fragments so they're not scored in isolation."""
        merged: List[str] = []
        buf = ''
        for f in frags:
            buf = f'{buf} {f}' if buf else f
            if len(buf) >= self.min_unit_chars:
                merged.append(buf)
                buf = ''
        if buf:
            if merged:
                merged[-1] = f'{merged[-1]} {buf}'
            else:
                merged.append(buf)
        return merged


# ═══════════════════════════════════════════════════════════════════════════════
# PassageIndexCondenser — first-sentence + keyword summary per chunk
# ═══════════════════════════════════════════════════════════════════════════════
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_YEAR_RE = re.compile(r'\b(?:1[0-9]{3}|20[0-9]{2})\b')
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3}\b')
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

_KW_STOP = frozenset({
    'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 'A', 'An',
    'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'It', 'He', 'She', 'They',
    'We', 'I', 'You', 'His', 'Her', 'Their', 'Our', 'Its',
    'However', 'Although', 'Because', 'While', 'Since', 'When', 'Where',
    'After', 'Before', 'During', 'Through', 'Without',
})

_INDEX_PROTECTED_KINDS: frozenset = frozenset({'tool_call', 'tool_response'})
_INDEX_PROTECTED_TYPES: frozenset = frozenset({'image', 'video', 'audio'})


class PassageIndexCondenser(Condenser):
    """Replaces each chunk with its first sentence + (Related: keywords).

    Designed for agentic QA: produces a lightweight "directory" so the model
    can decide which blocks to expand via tool calls.

    Args:
        max_keywords: Maximum keywords extracted from hidden body per chunk.
        skip_system: Whether to skip system-role chunks.
    """

    def __init__(self, max_keywords: int = 12, skip_system: bool = True) -> None:
        self.max_keywords = max_keywords
        self.skip_system = skip_system

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        return Chunks(chunks=[self._index_chunk(c) for c in chunks.chunks])

    def _index_chunk(self, chunk: Chunk) -> Chunk:
        content = chunk.get('content', '')
        if not isinstance(content, str) or not content.strip():
            return chunk
        if self.skip_system and chunk.get('role') == 'system':
            return chunk
        if chunk.get('role') == 'tool':
            return chunk
        if chunk.get('type') in _INDEX_PROTECTED_TYPES:
            return chunk
        raw = chunk.get('raw')
        if isinstance(raw, dict) and raw.get('kind') in _INDEX_PROTECTED_KINDS:
            return chunk

        first_sent, rest = self._split_first_sentence(content)
        if not rest:
            return chunk

        keywords = self._extract_keywords(first_sent, rest)
        suffix = f' (Related: {", ".join(keywords)})' if keywords else ''

        new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
        new_chunk['content'] = f'{first_sent}{suffix}'
        if isinstance(raw, dict):
            new_chunk['raw'] = {**raw, 'condensed': True}
        else:
            new_chunk['raw'] = {'condensed': True}
        return new_chunk

    @staticmethod
    def _split_first_sentence(text: str) -> Tuple[str, str]:
        text = text.strip()
        if not text:
            return '', ''
        parts = _SENTENCE_END_RE.split(text, maxsplit=1)
        if len(parts) == 1:
            return text, ''
        return parts[0].rstrip(), parts[1].lstrip()

    def _extract_keywords(self, first_sent: str, rest: str) -> List[str]:
        if not rest:
            return []
        first_lower = first_sent.lower()
        seen: List[str] = []
        seen_lower: set = set()

        def _push(tok: str) -> None:
            low = tok.lower()
            if low in first_lower or low in seen_lower or tok in _KW_STOP:
                return
            seen_lower.add(low)
            seen.append(tok)

        for regex in (_YEAR_RE, _PROPER_NOUN_RE, _NUMBER_RE):
            for m in regex.finditer(rest):
                _push(m.group().strip())
                if len(seen) >= self.max_keywords:
                    return seen
        return seen
