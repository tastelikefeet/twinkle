# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

from twinkle_agentic.tools.tool_manager import ToolManager

_MEDIA_KEYS = ('images', 'videos', 'audios')

OutputSanitizer = Callable[[str], str]
TurnHook = Callable[[int, List['Rollout'], List[Dict[str, Any]], List[Any]], None]
ToolFactory = Callable[['Rollout'], ToolManager]
DisplayBuilder = Callable[[List['Rollout']], List[Dict[str, Any]]]


@dataclass(slots=True)
class Rollout:
    """Per-rollout bookkeeping: trajectory, turn sequences, caller state.

    ``state`` is an opaque per-rollout scratchpad owned by the CALLER.
    The rollout itself never reads it. Callers that need per-rollout
    machinery (e.g. a compression cache) stash it here and recover it
    from their ``display_builder`` / ``tool_factory`` closures.

    Notes on ``turn_sequences``: all per-turn (prompt, generation) training
    features. Keeping every turn (not just the final one) lets GRPO train
    on the tool-call decision turns too, so the policy can learn WHEN to
    expand a ``<block_N>``. Rollout-level reward is replicated onto every
    turn's advantage at optimiser feed time.
    """

    prompt_trajectory: InitVar[Dict[str, Any]]
    state: Any = None
    trajectory: Dict[str, Any] = field(init=False)
    final_sequence: Any = field(init=False, default=None)
    turn_sequences: List[Any] = field(init=False, default_factory=list)
    turns: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)

    def __post_init__(self, prompt_trajectory: Dict[str, Any]) -> None:
        self.trajectory = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        for k in _MEDIA_KEYS:
            if prompt_trajectory.get(k):
                self.trajectory[k] = list(prompt_trajectory[k])


def _default_display_builder(active: List[Rollout]) -> List[Dict[str, Any]]:
    """Default: feed each rollout's trajectory straight to the sampler."""
    return [r.trajectory for r in active]


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    tool_factory: ToolFactory,
    template: Template,
    *,
    max_turns: int,
    display_builder: Optional[DisplayBuilder] = None,
    initial_states: Optional[List[Any]] = None,
    output_sanitizers: Optional[List[OutputSanitizer]] = None,
    min_batch_size: int = 1,
    on_turn: Optional[TurnHook] = None,
) -> List[Rollout]:
    """Run a multi-turn agentic rollout loop over ``prompts``.

    Args:
        prompts: Prompt trajectories, one per rollout.
        sampler: Policy sampler (vLLM). Typically a Ray actor, so its
            remote ``.template`` is unreachable from the driver — pass
            a local ``Template`` via the ``template`` argument instead.
        sampling_params: Decoding params for the policy sampler.
        tool_factory: Factory ``(rollout) -> ToolManager`` invoked every
            turn for every active rollout.
        template: Local :class:`~twinkle.template.Template` used purely
            for ``parse_tool_call`` / ``clean_tool_call`` family
            dispatch; construct once outside the training loop.
        max_turns: Cap on turns per rollout.
        display_builder: Callback ``(active_rollouts) -> List[display]``
            that produces the input fed to the sampler each turn. This
            is the hook callers use to plug in compression / context
            editing WITHOUT the rollout knowing about it. Defaults to
            returning each rollout's trajectory verbatim (no
            compression).
        initial_states: Optional per-prompt opaque state, one entry per
            prompt, stashed on ``Rollout.state``. Callers use this to
            seed per-rollout compression caches (or anything else) —
            the rollout loop never reads these values.
        output_sanitizers: Optional extra cleaners applied AFTER
            ``template.clean_tool_call`` (e.g. ``strip_block_echoes`` to
            remove condenser ``<block_N>`` markup from assistant text).
        min_batch_size: Pad the sample call to at least this batch size
            (prevents small-batch under-utilisation of vLLM).
        on_turn: Optional hook called after each turn with
            ``(turn, active_rollouts, displays, responses)``. Exceptions
            inside the hook are swallowed so tracing cannot break training.

    Returns:
        List of completed :class:`Rollout` objects (same order as prompts).
    """
    assert template is not None, (
        'run_agentic_rollouts requires a local Template for tool-call '
        'parsing / cleaning; build one via ``Template(MODEL_ID)``.')

    if initial_states is not None:
        assert len(initial_states) == len(prompts), (
            f'initial_states length {len(initial_states)} != prompts {len(prompts)}')
        rollouts = [Rollout(p, s) for p, s in zip(prompts, initial_states)]
    else:
        rollouts = [Rollout(p) for p in prompts]

    build_displays = display_builder or _default_display_builder
    extra_sanitizers: List[OutputSanitizer] = list(output_sanitizers or ())

    for turn in range(max_turns):
        active = [r for r in rollouts if not r.done]
        if not active:
            break

        # Caller-owned display construction (compression plugs in here).
        displays: List[Dict[str, Any]] = build_displays(active)
        assert len(displays) == len(active), (
            f'display_builder returned {len(displays)} items for '
            f'{len(active)} active rollouts')
        tool_mgrs: List[ToolManager] = [tool_factory(r) for r in active]

        n_active = len(displays)
        if n_active < min_batch_size:
            displays = displays + [displays[0]] * (min_batch_size - n_active)

        responses = sampler.sample(displays, sampling_params)
        responses = responses[:n_active]

        for r, resp, tool_mgr in zip(active, responses, tool_mgrs):
            _advance_rollout(r, resp, tool_mgr, turn, template, extra_sanitizers)

        if on_turn is not None:
            try:
                on_turn(turn, active, displays, responses)
            except Exception:  # pragma: no cover — tracing must never break training
                pass

    for r in rollouts:
        r.done = True

    if max_turns > 0:
        empty = [i for i, r in enumerate(rollouts) if not r.turn_sequences]
        assert not empty, (
            f'rollouts {empty} have empty turn_sequences after {max_turns} '
            f'turns; likely a sampler or min_batch_size bug.')

    return rollouts


def _advance_rollout(
    rollout: Rollout,
    response: Any,
    tool_mgr: ToolManager,
    turn: int,
    template: Template,
    extra_sanitizers: List[OutputSanitizer],
) -> None:
    """Apply a single sampler response to one rollout (mutates in place)."""
    seq = response.sequences[0]
    rollout.final_sequence = seq
    rollout.turn_sequences.append(seq)
    rollout.turns += 1
    decoded = seq.decoded or ''
    tool_calls = template.parse_tool_call(decoded)
    cleaned = _sanitize_output(decoded, template, extra_sanitizers)

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
    template: Template,
    extra_sanitizers: List[OutputSanitizer],
) -> str:
    text = template.clean_tool_call(text)
    for fn in extra_sanitizers:
        text = fn(text)
    return (text or '').rstrip()
