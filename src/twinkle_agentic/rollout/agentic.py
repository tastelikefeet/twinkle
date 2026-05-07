# Copyright (c) ModelScope Contributors. All rights reserved.
"""Generic multi-turn agentic rollout orchestration.

This module is deliberately format-agnostic:

* The **tool-call wire format** is injected via a
  :class:`~twinkle_agentic.tools.protocol.ToolCallProtocol` instance
  (Qwen3.5 by default).
* The **compression cache** lives entirely in
  :mod:`twinkle_agentic.condenser.frozen`; the rollout only holds a
  reference to each rollout's :class:`FrozenContext` and calls the
  batched freeze helper once per turn.
* Callers can inject **extra output sanitisers**
  (e.g. :func:`strip_block_echoes`) to clean format-specific echoes
  from assistant text before it is committed to the trajectory.
"""
import json
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser.base import Condenser
from twinkle_agentic.condenser.frozen import (
    FrozenContext,
    batch_freeze_delta_pairs,
)
from twinkle_agentic.tools.protocol import (
    Qwen35ToolCallProtocol,
    ToolCallProtocol,
)
from twinkle_agentic.tools.tool_manager import ToolManager

_MEDIA_KEYS = ('images', 'videos', 'audios')

OutputSanitizer = Callable[[str], str]
TurnHook = Callable[[int, List['Rollout'], List[Dict[str, Any]], List[Any]], None]
ToolFactory = Callable[['Rollout'], ToolManager]


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


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    chunker: NativeChunker,
    condenser: Condenser,
    tool_factory: ToolFactory,
    *,
    max_turns: int,
    tool_protocol: Optional[ToolCallProtocol] = None,
    output_sanitizers: Optional[List[OutputSanitizer]] = None,
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
            turn for every active rollout.
        max_turns: Cap on turns per rollout.
        tool_protocol: Tool-call parser + cleaner pair. Defaults to
            :class:`Qwen35ToolCallProtocol`. Swap in a different
            :class:`ToolCallProtocol` subclass to train a non-Qwen
            model without touching this file.
        output_sanitizers: Optional extra cleaners applied AFTER
            ``tool_protocol.clean`` (e.g.
            :func:`~twinkle_agentic.condenser.frozen.strip_block_echoes`
            to remove echoed ``<block_N>`` tags).
        min_batch_size: Pad the sample call to at least this batch size
            (prevents small-batch under-utilisation of vLLM).
        initial_frozens: Optional per-prompt ``FrozenContext`` to clone
            into the rollout's cache — lets callers share the initial
            compression across ``num_generations`` rollouts of the same
            prompt.
        on_turn: Optional hook called after each turn with
            ``(turn, active_rollouts, displays, responses)``. Exceptions
            inside the hook are swallowed so tracing cannot break training.

    Returns:
        List of completed :class:`Rollout` objects (same order as prompts).
    """
    if initial_frozens is not None:
        assert len(initial_frozens) == len(prompts), (
            f'initial_frozens length {len(initial_frozens)} != prompts {len(prompts)}')
        rollouts = [Rollout(p, ifz) for p, ifz in zip(prompts, initial_frozens)]
    else:
        rollouts = [Rollout(p) for p in prompts]

    protocol: ToolCallProtocol = tool_protocol or Qwen35ToolCallProtocol()
    extra_sanitizers: List[OutputSanitizer] = list(output_sanitizers or ())

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
            _advance_rollout(r, resp, tool_mgr, turn, protocol, extra_sanitizers)

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
    protocol: ToolCallProtocol,
    extra_sanitizers: List[OutputSanitizer],
) -> None:
    """Apply a single sampler response to one rollout (mutates in place)."""
    seq = response.sequences[0]
    rollout.final_sequence = seq
    rollout.turn_sequences.append(seq)
    rollout.turns += 1
    decoded = seq.decoded or ''
    tool_calls = protocol.parse(decoded)
    cleaned = _sanitize_output(decoded, protocol, extra_sanitizers)

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


def _sanitize_output(
    text: str,
    protocol: ToolCallProtocol,
    extra_sanitizers: List[OutputSanitizer],
) -> str:
    text = protocol.clean(text)
    for fn in extra_sanitizers:
        text = fn(text)
    return (text or '').rstrip()
