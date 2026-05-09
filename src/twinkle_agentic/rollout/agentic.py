# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from twinkle.data_format import SamplingParams, ToolCall
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

from twinkle_agentic.tools.tool_manager import ToolManager

OutputSanitizer = Callable[[str], str]
TurnHook = Callable[[int, List['Rollout'], List[Dict[str, Any]], List[Any]], None]
TrajectoryBuilder = Callable[[List['Rollout']], List[Dict[str, Any]]]


@dataclass(slots=True)
class Rollout:
    trajectory: Dict[str, Any] = None
    state: Any = None
    sequence: Any = field(init=False, default=None)
    turns: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        src = self.trajectory
        self.trajectory = {
            k: list(v) if isinstance(v, list) else v for k, v in src.items()
        }


def _default_trajectory_builder(active: List[Rollout]) -> List[Dict[str, Any]]:
    """Default: feed each rollout's trajectory straight to the sampler."""
    return [r.trajectory for r in active]


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    tool_factory: Callable[['Rollout'], Callable[[Dict[str, Any]], str]],
    template: Template,
    *,
    max_turns: int,
    trajectory_builder: Optional[TrajectoryBuilder] = None,
    output_sanitizers: Optional[List[OutputSanitizer]] = None,
    min_batch_size: int = 1,
    on_turn: Optional[TurnHook] = None,
    initial_states: Optional[List[Any]] = None,
) -> List[Rollout]:
    assert template is not None, (
        'run_agentic_rollouts requires a local Template for tool-call '
        'parsing / cleaning; build one via ``Template(MODEL_ID)``.')

    if initial_states is not None:
        assert len(initial_states) == len(prompts), (
            f'initial_states length {len(initial_states)} != prompts {len(prompts)}')
        rollouts = [Rollout(p, s) for p, s in zip(prompts, initial_states)]
    else:
        rollouts = [Rollout(p) for p in prompts]

    build_trajectories = trajectory_builder or _default_trajectory_builder
    extra_sanitizers: List[OutputSanitizer] = list(output_sanitizers or ())

    for turn in range(max_turns):
        active = [r for r in rollouts if not r.done]
        if not active:
            break

        # Caller-owned trajectory construction (compression plugs in here).
        trajectories: List[Dict[str, Any]] = build_trajectories(active)
        assert len(trajectories) == len(active), (
            f'trajectory_builder returned {len(trajectories)} items for '
            f'{len(active)} active rollouts')
        tool_mgrs: List[Callable[[Dict[str, Any]], str]] = [tool_factory(r) for r in active]

        n_active = len(trajectories)
        if n_active < min_batch_size:
            trajectories = trajectories + [trajectories[0]] * (min_batch_size - n_active)

        responses = sampler.sample(trajectories, sampling_params)
        responses = responses[:n_active]

        for r, resp, tool_mgr in zip(active, responses, tool_mgrs):
            _advance_rollout(r, resp, tool_mgr, turn, template, extra_sanitizers)

        if on_turn is not None:
            on_turn(turn, active, trajectories, responses)

    for r in rollouts:
        r.done = True

    if max_turns > 0:
        empty = [i for i, r in enumerate(rollouts) if r.sequence is None]
        assert not empty, (
            f'rollouts {empty} produced no sequence after {max_turns} '
            f'turns; likely a sampler or min_batch_size bug.')

    return rollouts


def _advance_rollout(
    rollout: Rollout,
    response: Any,
    tool_dispatcher: Callable[[Dict[str, Any]], str],
    turn: int,
    template: Template,
    extra_sanitizers: List[OutputSanitizer],
) -> None:
    """Apply a single sampler response to one rollout (mutates in place)."""
    seq = response.sequences[0]
    rollout.sequence = seq
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
            'content': tool_dispatcher(tc),
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
