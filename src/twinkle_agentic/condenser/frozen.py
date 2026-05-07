# Copyright (c) ModelScope Contributors. All rights reserved.
"""Frozen-context incremental compression cache.

This module owns all **compression-layer** bookkeeping used by
multi-turn agentic rollouts:

* :class:`FrozenContext` — per-rollout cache of (full, compressed)
  chunks that grows incrementally as new assistant / tool turns arrive.
* :func:`batch_freeze_delta_pairs` — chunks each pending delta locally
  and issues ONE batched ``condenser.condense`` call across all active
  rollouts, avoiding the ``O(n_active)`` serial sampler overhead a
  per-rollout loop would otherwise incur.
* :func:`strip_passage_prefix`, :func:`ensure_context_header`,
  :func:`strip_block_echoes` — helpers tied to the compressor's
  ``<block_N>`` display format.

Kept separate from the rollout loop so the orchestration code can
remain compression-agnostic and both concerns can evolve independently.
"""
import re
from typing import Any, Dict, List, Tuple

from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format.chunk import Chunk, Chunks

# HotpotQA-style ``[N] Title:`` passage prefix. Once chunks are wrapped
# as ``<block_N>`` by the condenser, the upstream ``[N]`` competes with
# our own 1-based numbering, confusing the compressor and wasting
# tokens. ``strip_passage_prefix`` removes it from each chunk's content
# while leaving ``raw`` untouched.
_PASSAGE_PREFIX_RE = re.compile(r'^\s*\[\d+\]\s+')

# Detect the first ``<block_N>`` marker in a rendered user message so we
# can make sure the ``Context:`` header survives the chunk → condense →
# groupby round-trip.
_FIRST_BLOCK_RE = re.compile(r'<block_\d+>')

# Block-tag sanitisers for stripping echoed tags from model outputs.
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_ORPHAN_BLOCK_RE = re.compile(r'</?block_\d+>')

_MEDIA_KEYS = ('images', 'videos', 'audios')


def strip_passage_prefix(chunk: Chunk) -> Chunk:
    """Return a shallow copy of ``chunk`` with any leading ``[N] `` removed.

    Non-text chunks or chunks without the prefix pass through unchanged.
    ``raw`` is always preserved so downstream round-tripping keeps the
    original payload.
    """
    if chunk.get('type') != 'text':
        return chunk
    content = chunk.get('content')
    if not isinstance(content, str):
        return chunk
    stripped = _PASSAGE_PREFIX_RE.sub('', content, count=1)
    if stripped == content:
        return chunk
    return {**chunk, 'content': stripped}


def ensure_context_header(text: str) -> str:
    """Insert a ``Context:`` header before the first ``<block_N>`` if missing.

    Idempotent: if ``Context:`` already appears in the pre-block prefix
    (or there is no block marker at all), returns the text unchanged.
    """
    m = _FIRST_BLOCK_RE.search(text)
    if not m:
        return text
    prefix = text[:m.start()]
    if 'Context:' in prefix:
        return text
    preamble = prefix.rstrip()
    head = preamble + ('\n\n' if preamble else '')
    return f'{head}Context:\n\n{text[m.start():]}'


def strip_block_echoes(text: str) -> str:
    """Remove echoed ``<block_N>...</block_N>`` tags from model output.

    Handles both well-formed pairs (keeps the inner body) and orphan
    open/close tags that leak when ``max_new_tokens`` truncates the
    output mid-block.
    """
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text or '')
    text = _ORPHAN_BLOCK_RE.sub('', text)
    return text


class FrozenContext:
    """Incremental chunk+index cache. Only processes NEW messages each turn.

    Per-batch, the initial prompt (system + user + passages) is identical
    across the ``num_generations`` rollouts of a single prompt. Compress
    that initial slice once and clone the result into each rollout, to
    avoid re-compressing the same paragraphs N times.

    The heavy lifting (chunk + condense) is delegated to
    :func:`batch_freeze_delta_pairs`, which flattens all pending deltas
    into one condenser call.
    """
    __slots__ = ('frozen_msg_count', 'full_chunks', 'compressed_chunks', 'media_frozen')

    def __init__(self) -> None:
        self.frozen_msg_count: int = 0
        self.full_chunks: List[Chunk] = []
        self.compressed_chunks: List[Chunk] = []
        self.media_frozen: bool = False

    def clone(self) -> 'FrozenContext':
        """Shallow-copy safe for independent mutation."""
        cp = FrozenContext()
        cp.frozen_msg_count = self.frozen_msg_count
        cp.full_chunks = list(self.full_chunks)
        cp.compressed_chunks = list(self.compressed_chunks)
        cp.media_frozen = self.media_frozen
        return cp

    def freeze_delta(
        self,
        trajectory: Dict[str, Any],
        chunker: NativeChunker,
        condenser: Condenser,
    ) -> None:
        """Single-pair wrapper around :func:`batch_freeze_delta_pairs`."""
        batch_freeze_delta_pairs([(self, trajectory)], chunker, condenser)

    def render_display(self) -> Dict[str, Any]:
        """Return a trajectory-shaped view of the compressed chunks."""
        traj = Chunks(chunks=list(self.compressed_chunks)).to_trajectory()
        # Safety net: the chunker / condenser min_chars cutoff may merge
        # the tiny ``Question: ...\n\nContext:`` preamble chunk into its
        # neighbours, which strips the visual cue that the blocks ARE
        # the context paragraphs. Re-inject a one-line header.
        for msg in traj.get('messages', []):
            if msg.get('role') != 'user':
                continue
            content = msg.get('content')
            if isinstance(content, str):
                msg['content'] = ensure_context_header(content)
        return traj

    def render_full(self) -> Chunks:
        return Chunks(chunks=list(self.full_chunks))

    def displayed_to_full(self) -> Dict[int, int]:
        """Map ``displayed_block_number → full_chunk_idx``.

        Built off the compressed view so it matches what
        :meth:`render_display` actually wraps as ``<block_N>``.
        """
        return Chunks(chunks=list(self.compressed_chunks)).displayed_block_mapping()


def batch_freeze_delta_pairs(
    pairs: List[Tuple[FrozenContext, Dict[str, Any]]],
    chunker: NativeChunker,
    condenser: Condenser,
) -> None:
    """Batched ``freeze_delta`` across many ``(frozen, trajectory)`` pairs.

    Workflow:

    1. **Per-pair chunking** (CPU-only, fast). Compute each pair's
       message+media delta and chunk it locally.
    2. **One batched condense call** over ALL pairs' stripped chunks —
       avoids the ``O(n_active)`` serial ``sampler.sample`` overhead
       the per-turn loop would otherwise incur.
    3. **Scatter results back** to each pair's cache, preserving order.

    Defensive behaviour:

    * Drops any synthetic ``system`` chunk injected by
      ``template.format_trajectory`` when the delta had no system
      message, so block numbering stays aligned with model-visible turns.
    * Flips ``media_frozen`` only after the chunk is committed in
      phase 3, not optimistically in phase 1.
    """
    if not pairs:
        return

    # ---------- Phase 1: per-pair chunking (CPU-only, fast) ----------
    pending: List[Tuple[FrozenContext, Dict[str, Any], List[Chunk], bool]] = []
    for frozen, trajectory in pairs:
        total_msgs = trajectory['messages']
        new_msgs = total_msgs[frozen.frozen_msg_count:]
        needs_media = (
            not frozen.media_frozen
            and any(trajectory.get(k) for k in _MEDIA_KEYS))
        if not (new_msgs or needs_media):
            continue

        delta: Dict[str, Any] = {
            'messages': list(new_msgs),
            'user_data': trajectory.get('user_data', []),
        }
        if needs_media:
            for k in _MEDIA_KEYS:
                if trajectory.get(k):
                    delta[k] = trajectory[k]

        raw_full = chunker.chunk(delta)

        input_roles = {m.get('role') for m in new_msgs if isinstance(m, dict)}
        if 'system' not in input_roles:
            filtered = [c for c in raw_full.chunks if c.get('role') != 'system']
        else:
            filtered = list(raw_full.chunks)
        stripped = [strip_passage_prefix(c) for c in filtered]
        pending.append((frozen, trajectory, stripped, needs_media))

    if not pending:
        return

    # ---------- Phase 2: ONE batched condense call across all pairs ---
    flat: List[Chunk] = []
    boundaries: List[int] = [0]
    for _, _, stripped, _ in pending:
        flat.extend(stripped)
        boundaries.append(len(flat))
    condensed = condenser.condense(Chunks(chunks=flat)).chunks

    # ---------- Phase 3: scatter results back to each cache -----------
    for (frozen, trajectory, stripped, needs_media), start, end in zip(
            pending, boundaries, boundaries[1:]):
        frozen.full_chunks.extend(stripped)
        frozen.compressed_chunks.extend(condensed[start:end])
        frozen.frozen_msg_count = len(trajectory['messages'])
        if needs_media:
            frozen.media_frozen = True
