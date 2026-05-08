# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, Iterable, List, Optional, Pattern

from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser


_DEFAULT_SKIP_ROLES = frozenset({'system', 'tool', 'assistant'})
_SKIP_KINDS = frozenset({'tool_call', 'tool_response', 'reasoning_content'})
_SKIP_TYPES = frozenset({'image', 'video', 'audio'})


def _build_special_token_re(template: Optional[Template]) -> Pattern:
    """Build a strip regex from the template's tokenizer special tokens.

    Uses ``tokenizer.all_special_tokens`` so the condenser auto-adapts
    to any model family (Qwen / Llama / …) without a hand-maintained
    token list. Falls back to :data:`_FALLBACK_SPECIAL_TOKEN_RE` when
    no template is available or the tokenizer exposes no special tokens.
    """
    _FALLBACK_SPECIAL_TOKEN_RE = re.compile(
        r'<\|im_end\|>'
        r'|<\|im_start\|>\s*(?:assistant|user|system|tool)?\s*'
        r'|<\|endoftext\|>'
        r'|<\|eot_id\|>'
        r'|<\|start_header_id\|>[^<]*<\|end_header_id\|>'
    )
    if template is None:
        return _FALLBACK_SPECIAL_TOKEN_RE
    try:
        specials = list(getattr(template.tokenizer, 'all_special_tokens', []) or [])
    except Exception:
        return _FALLBACK_SPECIAL_TOKEN_RE
    # Longest first so ``<|im_start|>`` isn't shadowed by ``<|im_end|>`` etc.
    specials = [t for t in specials if isinstance(t, str) and t]
    if not specials:
        return _FALLBACK_SPECIAL_TOKEN_RE
    specials.sort(key=len, reverse=True)
    return re.compile('|'.join(re.escape(t) for t in specials))

_DEFAULT_SYSTEM_PROMPT = """You are a passage compressor. Given a single paragraph, output a compact Markdown summary using EXACTLY these three sections:

**Summary**: One sentence — state the main subject, topic, and scope.
**Key Facts**: Some facts covering the most critical facts: entities, relations, numbers, and dates.
**More**: Comma-separated keywords for secondary details not captured above (minor entities, extra attributes, alternate names, peripheral dates).

## Rules
- Keep the summary substantially shorter than the original passage.
- Do NOT answer any question. Do NOT add any preamble or closing remark.
- Output only the three Markdown sections, nothing else.

## Example

Input:
"Christopher Nolan (born 30 July 1970) is a British-American film director, \
producer and screenwriter. His film Inception (2010), a science-fiction heist \
movie starring Leonardo DiCaprio, grossed over $829 million worldwide and \
received eight Academy Award nominations, winning four. Nolan also directed \
The Dark Knight trilogy and Interstellar (2014)."

Output:
**Summary**: Christopher Nolan is a British-American filmmaker best known for \
directing Inception (2010) and several other major films.
**Key Facts**:
- Born 30 July 1970; British-American director.
- Inception (2010): sci-fi heist film starring Leonardo DiCaprio.
- Inception grossed $829M worldwide; won 4 of 8 Oscar nominations.
- Also directed The Dark Knight trilogy and Interstellar (2014).
**More**: producer, screenwriter, Academy Award full name, "heist movie" wording."""


class LLMPassageCondenser(Condenser):
    """LLM-driven passage condenser."""

    def __init__(
        self,
        sampler: 'vLLMSampler',
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        min_chars: int = 200,
        skip_roles: Optional[Iterable[str]] = None,
        template: Optional[Template] = None,
    ) -> None:
        self.sampler = sampler
        self.sampling_params = sampling_params or SamplingParams(
            max_tokens=128, num_samples=1, temperature=0.3, top_p=0.9)
        self.system_prompt = system_prompt
        self.min_chars = min_chars
        self.skip_roles = (frozenset(skip_roles) if skip_roles is not None
                           else _DEFAULT_SKIP_ROLES)
        self._special_token_re: Pattern = _build_special_token_re(template)


    def condense(self, chunks: Chunks, **kwargs) -> Chunks:
        """Compress eligible chunks via a single parallel sampler call."""
        targets: List[int] = [
            i for i, c in enumerate(chunks.chunks) if self._should_compress(c)
        ]
        if not targets:
            return Chunks(chunks=list(chunks.chunks))

        trajectories: List[Trajectory] = [
            self._build_prompt(chunks.chunks[i].get('content', '') or '')
            for i in targets
        ]

        responses = self.sampler.sample(
            trajectories, self.sampling_params, use_base_model=True)

        out: List[Chunk] = list(chunks.chunks)
        for idx, resp in zip(targets, responses):
            summary = self._extract_text(resp)
            if not summary:
                continue
            out[idx] = self._apply_summary(out[idx], summary)
        return Chunks(chunks=out)

    def _should_compress(self, chunk: Chunk) -> bool:
        if chunk.get('type') != 'text':
            return False
        if chunk.get('role') in self.skip_roles:
            return False
        if chunk.get('type') in _SKIP_TYPES:
            return False
        raw = chunk.get('raw')
        if isinstance(raw, dict):
            if raw.get('kind') in _SKIP_KINDS:
                return False
            if raw.get('condensed'):
                return False  # already compressed
        content = chunk.get('content')
        if not isinstance(content, str) or len(content) < self.min_chars:
            return False
        return True

    def _build_prompt(self, passage: str) -> Trajectory:
        return Trajectory(messages=[
            Message(role='system', content=self.system_prompt),
            Message(role='user', content=f'Passage:\n\n{passage}'),
        ])

    @staticmethod
    def _apply_summary(chunk: Chunk, summary: str) -> Chunk:
        new: Dict[str, Any] = dict(chunk)
        new['content'] = summary
        raw = chunk.get('raw')
        new['raw'] = {**(raw if isinstance(raw, dict) else {}), 'condensed': True}
        return new  # type: ignore[return-value]

    def _extract_text(self, response: Any) -> str:
        seqs = getattr(response, 'sequences', None)
        if not seqs:
            return ''
        decoded = seqs[0].decoded or ''
        decoded = self._special_token_re.sub('', decoded)
        return decoded.strip()
