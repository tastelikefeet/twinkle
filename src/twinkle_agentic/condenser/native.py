# Copyright (c) ModelScope Contributors. All rights reserved.
"""PassageIndexCondenser — first-sentence + NER triplets + keywords per chunk."""
from __future__ import annotations

import re
from typing import List, Tuple

import spacy

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser

# ═══════════════════════════════════════════════════════════════════════════════
# PassageIndexCondenser — first-sentence + NER triplets + keywords per chunk
# ═══════════════════════════════════════════════════════════════════════════════
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_KW_PATTERNS = (
    re.compile(r'\b(?:1[0-9]{3}|20[0-9]{2})\b'),                          # years
    re.compile(r'\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3}\b'),     # proper nouns
    re.compile(r'\b\d+(?:\.\d+)?\b'),                                      # numbers
)
_KW_STOP = frozenset(
    'The This That These Those There Here A An Is Are Was Were Be Been '
    'It He She They We I You His Her Their Our Its '
    'However Although Because While Since When Where After Before During Through Without'.split()
)
_DEFAULT_SKIP_ROLES = frozenset({'system', 'tool'})
_SKIP_KINDS = frozenset({'tool_call', 'tool_response'})
_SKIP_TYPES = frozenset({'image', 'video', 'audio'})
_NER_LABELS = {'PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT', 'WORK_OF_ART', 'FAC'}
# POS tag groups used by ``_extract_relation``'s fallback chain.
_REL_POS_VERB = ('VERB', 'ADP', 'PART')
_REL_POS_BROAD = ('VERB', 'ADP', 'PART', 'NOUN', 'ADJ', 'ADV')
_REL_BROAD_STOP = frozenset({'the', 'a', 'an'})

_NLP = spacy.load('en_core_web_sm', disable=['lemmatizer', 'textcat'])


class PassageIndexCondenser(Condenser):
    """First-sentence + NER triplets + keywords index per chunk.

    Args:
        max_keywords: Max regex-based keywords per chunk.
        max_triplets: Max NER triplets per chunk.
    """

    def __init__(
        self,
        max_keywords: int = 12,
        max_triplets: int = 3,
        skip_roles=None,
    ) -> None:
        self.max_keywords = max_keywords
        self.max_triplets = max_triplets
        self.skip_roles = (frozenset(skip_roles) if skip_roles is not None
                           else _DEFAULT_SKIP_ROLES)

    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        return Chunks(chunks=[self._index(c) for c in chunks.chunks])

    def _should_skip(self, chunk: Chunk) -> bool:
        if chunk.get('role') in self.skip_roles:
            return True
        if chunk.get('type') in _SKIP_TYPES:
            return True
        raw = chunk.get('raw')
        return isinstance(raw, dict) and raw.get('kind') in _SKIP_KINDS

    def _index(self, chunk: Chunk) -> Chunk:
        content = chunk.get('content', '')
        if not isinstance(content, str) or not content.strip() or self._should_skip(chunk):
            return chunk

        first, rest = self._split_first(content)
        if not rest:
            return chunk

        parts = []
        triplets = self._triplets(rest)
        if triplets:
            parts.append('Facts: ' + '; '.join(triplets))
        kws = self._keywords(first, rest)
        if kws:
            parts.append('Related: ' + ', '.join(kws))

        new = dict(chunk)
        new['content'] = f'{first} ({" | ".join(parts)})' if parts else first
        raw = chunk.get('raw')
        new['raw'] = {**(raw if isinstance(raw, dict) else {}), 'condensed': True}
        return new  # type: ignore[return-value]

    @staticmethod
    def _split_first(text: str) -> Tuple[str, str]:
        parts = _SENT_RE.split(text.strip(), maxsplit=1)
        return (parts[0].rstrip(), parts[1].lstrip()) if len(parts) > 1 else (text.strip(), '')

    def _triplets(self, text: str) -> List[str]:
        doc = _NLP(text[:2000])
        results: List[str] = []
        seen: set = set()
        for sent in doc.sents:
            ents = [e for e in sent.ents if e.label_ in _NER_LABELS]
            if len(ents) < 2:
                continue
            s, o = (ents[0], ents[1]) if ents[0].start < ents[1].start else (ents[1], ents[0])
            key = (s.text.lower(), o.text.lower())
            if key in seen:
                continue
            seen.add(key)
            rel = self._extract_relation(doc, s, o)
            if not rel:
                continue  # skip triplets with no meaningful relation
            results.append(f'{s.text} [{rel}] {o.text}')
            if len(results) >= self.max_triplets:
                break
        return results

    @staticmethod
    def _extract_relation(doc, subj, obj) -> str:
        """Extract a relation phrase between two entities.

        Tries three strategies in order, returning the first non-empty
        match (each truncated to 30 chars):

        1. Verbs + prepositions in the span between the two entities.
        2. Same span but broader POS set (adds NOUN/ADJ/ADV) when no
           verb is present.
        3. Sentence root verb as a last resort.

        Returns ``''`` when nothing meaningful is found, signalling the
        caller to drop the triplet entirely.
        """
        span = doc[subj.end:obj.start]

        def _join(tokens):
            return ' '.join(t.text for t in tokens)[:30] if tokens else ''

        # Strategy 1: verbs + prepositions only.
        verb_prep = [t for t in span if t.pos_ in _REL_POS_VERB and not t.is_stop]
        if verb_prep:
            return _join(verb_prep)

        # Strategy 2: broaden to nouns / adjectives / adverbs.
        broad = [t for t in span
                 if t.pos_ in _REL_POS_BROAD
                 and not t.is_stop
                 and t.text.lower() not in _REL_BROAD_STOP]
        if broad:
            return _join(broad)

        # Strategy 3: dependency root verb of the sentence.
        sent = getattr(subj, 'sent', None)
        if sent is not None:
            for t in sent:
                if t.dep_ == 'ROOT' and t.pos_ == 'VERB':
                    return t.text
        return ''

    def _keywords(self, first: str, rest: str) -> List[str]:
        first_low = first.lower()
        seen: List[str] = []
        seen_low: set = set()
        for pat in _KW_PATTERNS:
            for m in pat.finditer(rest):
                tok = m.group().strip()
                low = tok.lower()
                if low in first_low or low in seen_low or tok in _KW_STOP:
                    continue
                seen_low.add(low)
                seen.append(tok)
                if len(seen) >= self.max_keywords:
                    return seen
        return seen
