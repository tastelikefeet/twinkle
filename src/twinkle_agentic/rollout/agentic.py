# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic multi-turn agentic rollout primitives.

Public API:
    parse_tool_calls(text)       → List[{'tool_name', 'arguments'}]
    clean_assistant_output(text) → str
    strip_passage_prefix(chunk)  → Chunk
    ensure_context_header(text)  → str
    FrozenContext                → per-rollout incremental chunk+condense cache
    batch_freeze_delta_pairs(pairs, chunker, condenser) → None
    Rollout                      → per-rollout state holder
    run_agentic_rollouts(...)    → full chunk → condense → sample → tool loop

The module is intentionally free of any task-specific logic (reward,
dataset, prompts). Callers inject condenser + tool factory so the same
loop works for HotpotQA, synthetic math, open-web search, etc.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.data_format.chunk import Chunk, Chunks
from twinkle_agentic.tools.tool_manager import ToolManager

# ═══════════════════════════════════════════════════════════════════════════════
# Regex constants (tool-call parsing + context sanitization)
# ═══════════════════════════════════════════════════════════════════════════════
_TOOL_CALL_BLOCK_RE = re.compile(r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
_FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
_PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
_TOOL_CALL_STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_ORPHAN_BLOCK_RE = re.compile(r'</?block_\d+>')
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

_MEDIA_KEYS = ('images', 'videos', 'audios')


# ═══════════════════════════════════════════════════════════════════════════════
# Tool-call parsing & text sanitisation
# ═══════════════════════════════════════════════════════════════════════════════
def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from assistant text (Qwen3.5 XML + JSON fallback)."""
    calls: List[Dict[str, Any]] = []
    for block_m in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = block_m.group(1)
        func_m = _FUNCTION_RE.search(block)
        if func_m:
            args: Dict[str, Any] = {}
            for pm in _PARAMETER_RE.finditer(func_m.group(2)):
                key = pm.group(1).strip()
                val = pm.group(2).strip()
                try:
                    args[key] = json.loads(val)
                except (json.JSONDecodeError, ValueError):
                    args[key] = val
            calls.append({'tool_name': func_m.group(1).strip(), 'arguments': args})
            continue
        # JSON fallback
        try:
            data = json.loads(block)
        except json.JSONDecodeError:
            continue
        name = data.get('name') or data.get('tool_name', '')
        if not name:
            continue
        args = data.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        calls.append({
            'tool_name': name,
            'arguments': args if isinstance(args, dict) else {},
        })
    return calls


def clean_assistant_output(text: str) -> str:
    """Strip ``<tool_call>`` and echoed ``<block_N>`` tags from assistant text."""
    text = _TOOL_CALL_STRIP_RE.sub('', text or '')
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text)
    # Mop up orphan ``<block_N>`` / ``</block_N>`` tags that can leak
    # when MAX_NEW_TOKENS truncates the output mid-block.
    text = _ORPHAN_BLOCK_RE.sub('', text)
    return text.rstrip()


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


# ═══════════════════════════════════════════════════════════════════════════════
# FrozenContext — per-rollout incremental chunk+condense cache
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout — per-rollout state holder
# ═══════════════════════════════════════════════════════════════════════════════
class Rollout:
    """Per-rollout bookkeeping: trajectory, turn sequences, frozen cache."""
    __slots__ = (
        'trajectory', 'final_sequence', 'turn_sequences', 'turns', 'done', 'frozen')

    def __init__(
        self,
        prompt_trajectory: Dict[str, Any],
        initial_frozen: Optional[FrozenContext] = None,
    ) -> None:
        self.trajectory: Dict[str, Any] = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        for k in _MEDIA_KEYS:
            if prompt_trajectory.get(k):
                self.trajectory[k] = list(prompt_trajectory[k])
        self.final_sequence = None
        # All per-turn (prompt, generation) training features. Keeping
        # every turn (not just the final one) lets GRPO train on the
        # tool-call decision turns too, so the policy can learn WHEN to
        # expand a <block_N>. Rollout-level reward is replicated onto
        # every turn's advantage at optimiser feed time.
        self.turn_sequences: List[Any] = []
        self.turns = 0
        self.done = False
        # Inherit the already-compressed initial prompt when provided,
        # else fall back to an empty cache (legacy single-rollout).
        self.frozen = (
            initial_frozen.clone() if initial_frozen is not None else FrozenContext())


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level rollout loop
# ═══════════════════════════════════════════════════════════════════════════════
TurnHook = Callable[[int, List[Rollout], List[Dict[str, Any]], List[Any]], None]
ToolFactory = Callable[[Rollout], ToolManager]


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    chunker: NativeChunker,
    condenser: Condenser,
    tool_factory: ToolFactory,
    *,
    max_turns: int,
    min_batch_size: int = 1,
    initial_frozens: Optional[List[Optional[FrozenContext]]] = None,
    on_turn: Optional[TurnHook] = None,
) -> List[Rollout]:
    """Run a multi-turn agentic rollout loop over ``prompts``.

    Args:
        prompts: Prompt trajectories, one per rollout.
        sampler: Policy sampler (vLLM).
        sampling_params: Decoding params for the policy sampler.
        chunker: Chunker used to split each turn's delta.
        condenser: Condenser used to compress chunks between turns.
        tool_factory: Factory ``(rollout) -> ToolManager`` invoked every
            turn for every active rollout. Typically returns a manager
            wired with the rollout's current full/compressed chunks.
        max_turns: Cap on turns per rollout.
        min_batch_size: Pad the sample call to at least this batch size
            (prevents small-batch under-utilisation of vLLM).
        initial_frozens: Optional per-prompt ``FrozenContext`` to clone
            into the rollout's cache — lets callers share the initial
            compression across ``num_generations`` rollouts of the same
            prompt.
        on_turn: Optional hook called after each turn with
            ``(turn, active_rollouts, displays, responses)`` — used by
            trace logging. Exceptions inside the hook are swallowed so
            tracing cannot break training.

    Returns:
        List of completed :class:`Rollout` objects (same order as prompts).
    """
    if initial_frozens is not None:
        assert len(initial_frozens) == len(prompts), (
            f'initial_frozens length {len(initial_frozens)} != prompts {len(prompts)}')
        rollouts = [Rollout(p, ifz) for p, ifz in zip(prompts, initial_frozens)]
    else:
        rollouts = [Rollout(p) for p in prompts]

    for turn in range(max_turns):
        active = [r for r in rollouts if not r.done]
        if not active:
            break

        # Batch chunk + condense for all active rollouts in a SINGLE
        # condenser call.
        batch_freeze_delta_pairs(
            [(r.frozen, r.trajectory) for r in active], chunker, condenser)

        displays: List[Dict[str, Any]] = [r.frozen.render_display() for r in active]
        tool_mgrs: List[ToolManager] = [tool_factory(r) for r in active]

        n_active = len(displays)
        if n_active < min_batch_size:
            displays = displays + [displays[0]] * (min_batch_size - n_active)

        responses = sampler.sample(displays, sampling_params)
        responses = responses[:n_active]

        for r, resp, tool_mgr in zip(active, responses, tool_mgrs):
            _advance_rollout(r, resp, tool_mgr, turn)

        if on_turn is not None:
            try:
                on_turn(turn, active, displays, responses)
            except Exception:  # pragma: no cover — tracing must never break training
                pass

    for r in rollouts:
        r.done = True
    return rollouts


def _advance_rollout(
    rollout: Rollout,
    response: Any,
    tool_mgr: ToolManager,
    turn: int,
) -> None:
    """Apply a single sampler response to one rollout (mutates in place)."""
    seq = response.sequences[0]
    rollout.final_sequence = seq
    rollout.turn_sequences.append(seq)
    rollout.turns += 1
    decoded = seq.decoded or ''
    tool_calls = parse_tool_calls(decoded)
    cleaned = clean_assistant_output(decoded)

    if not tool_calls:
        rollout.trajectory['messages'].append(
            {'role': 'assistant', 'content': cleaned})
        rollout.done = True
        return

    rollout.trajectory['messages'].append({
        'role': 'assistant',
        'content': cleaned,
        'tool_calls': [
            {
                'tool_name': tc['tool_name'],
                'arguments': (
                    tc['arguments'] if isinstance(tc['arguments'], str)
                    else json.dumps(tc['arguments'], ensure_ascii=False)),
            }
            for tc in tool_calls
        ],
    })
    for i, tc in enumerate(tool_calls):
        rollout.trajectory['messages'].append({
            'role': 'tool',
            'content': tool_mgr.dispatch(tc),
            'tool_call_id': f'call_t{turn}_i{i}',
        })
