# Copyright (c) ModelScope Contributors. All rights reserved.
"""LLM-backed passage condenser.

Delegates per-chunk summarisation to a provided ``vLLMSampler`` instance.
The sampler is expected to run the *base* LLM (no LoRA) so compression
stays decoupled from the policy being trained.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.sampler import vLLMSampler

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format.chunk import Chunk

from .base import Condenser

# ═══════════════════════════════════════════════════════════════════════════════
# Skip rules (mirror PassageIndexCondenser semantics)
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULT_SKIP_ROLES = frozenset({'system', 'tool'})
_SKIP_KINDS = frozenset({'tool_call', 'tool_response', 'reasoning_content'})
_SKIP_TYPES = frozenset({'image', 'video', 'audio'})

# Chat-template special tokens that can leak into the decoded summary when
# the sampler's ``template.decode`` is invoked with ``skip_special_tokens``
# disabled (default in Qwen3.5 Template).  If these tokens end up as the
# ``content`` of a condensed chunk, they will be re-serialised into the
# NEXT turn's prompt — breaking the chat structure (e.g. an early
# ``<|im_end|>`` closes the user turn prematurely) and wasting tokens.
# Strip them defensively right after decoding, before the summary is
# handed back to the chunk.  Covers Qwen3.5, Qwen2, Llama-3 and generic
# ``<|endoftext|>``.
_SPECIAL_TOKEN_RE = re.compile(
    r'<\|im_end\|>'
    r'|<\|im_start\|>\s*(?:assistant|user|system|tool)?\s*'
    r'|<\|endoftext\|>'
    r'|<\|eot_id\|>'
    r'|<\|start_header_id\|>[^<]*<\|end_header_id\|>'
)

_DEFAULT_SYSTEM_PROMPT = (
    'You are a passage compressor for a multi-hop QA pipeline. '
    'Given a single passage, produce a compact Markdown summary using '
    'EXACTLY these three sections and nothing else:\n\n'
    '**Summary**: one sentence describing what the passage is about '
    '(the subject entity, topic, and scope).\n'
    '**Key**: a short bullet list (max 5 items, each under 15 words) of '
    'the most salient facts — entities, relations, numbers, dates.\n'
    '**More**: comma-separated keywords/phrases hinting at the additional '
    'information that would be recovered by expanding the passage (names '
    'of secondary entities, minor dates, extra attributes).\n\n'
    'Rules: stay under 120 tokens total. Do NOT answer any question. Do '
    'NOT add any preamble or closing. Output the three Markdown sections '
    'directly, nothing else.'
)


class LLMPassageCondenser(Condenser):
    """LLM-driven passage condenser.

    Delegates per-chunk summarisation to a ``vLLMSampler`` running the
    base model (typically without LoRA). All eligible chunks in a
    :class:`Chunks` object are batched into a single ``sampler.sample``
    call, which vLLM executes in parallel (async gather + DP slicing),
    so total compression latency scales sub-linearly with the number of
    chunks.

    Args:
        sampler: A live ``vLLMSampler`` with ``set_template`` already
            called. The sampler's base weights are used for compression
            regardless of any RL-training LoRA that may be mounted —
            the condenser issues its requests with ``use_base_model=True``
            to bypass the auto-synced LoRA fallback, so the policy
            sampler can be reused as-is.
        sampling_params: Decoding parameters for compression. If ``None``,
            a conservative default is used (max 128 tokens, temperature 0.3).
        system_prompt: Override for the compression system prompt. The
            default asks for a Markdown summary with Summary/Key/More.
        min_chars: Chunks whose text is shorter than this are passed
            through untouched — compressing short passages usually makes
            them longer.
        skip_roles: Message roles whose chunks are never compressed.
            Defaults to ``('system', 'tool')``. For agentic RL pipelines
            where the policy's own reasoning must NOT be summarised
            between turns (otherwise the model sees its own CoT through
            a Summary/Key/More lens), pass ``('system', 'tool',
            'assistant')``.
    """

    def __init__(
        self,
        sampler: 'vLLMSampler',
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        min_chars: int = 200,
        skip_roles: Optional[Iterable[str]] = None,
    ) -> None:
        self.sampler = sampler
        self.sampling_params = sampling_params or SamplingParams(
            max_tokens=128, num_samples=1, temperature=0.3, top_p=0.9)
        self.system_prompt = system_prompt
        self.min_chars = min_chars
        self.skip_roles = (frozenset(skip_roles) if skip_roles is not None
                           else _DEFAULT_SKIP_ROLES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
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

        # One batched call — vLLMSampler issues async requests concurrently
        # (``asyncio.gather`` inside ``sample``) and fans out across DP ranks
        # when ``device_mesh.dp_size > 1``. ``use_base_model=True`` forces
        # the engine to bypass the auto-synced LoRA so compression always
        # runs on the underlying base weights, decoupled from policy updates.
        responses = self.sampler.sample(
            trajectories, self.sampling_params, use_base_model=True)

        out: List[Chunk] = list(chunks.chunks)
        for idx, resp in zip(targets, responses):
            summary = self._extract_text(resp)
            if not summary:
                continue
            out[idx] = self._apply_summary(out[idx], summary)
        return Chunks(chunks=out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
    def _extract_text(response: Any) -> str:
        seqs = getattr(response, 'sequences', None)
        if not seqs:
            return ''
        decoded = seqs[0].decoded or ''
        decoded = _SPECIAL_TOKEN_RE.sub('', decoded)
        return decoded.strip()

    @staticmethod
    def _apply_summary(chunk: Chunk, summary: str) -> Chunk:
        new: Dict[str, Any] = dict(chunk)
        new['content'] = summary
        raw = chunk.get('raw')
        new['raw'] = {**(raw if isinstance(raw, dict) else {}), 'condensed': True}
        return new  # type: ignore[return-value]
