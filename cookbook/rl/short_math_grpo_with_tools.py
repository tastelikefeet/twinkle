"""GRPO training with context compression + tool-augmented multi-turn rollouts.

Built on top of ``short_math_grpo.py``.  The only difference is the sampling
phase: instead of a single ``sampler.sample()`` producing one response per
prompt, every prompt runs through an **agentic rollout loop**:

    while not done:
        1. Chunk the current multi-turn trajectory (NativeChunker).
        2. Condense each chunk in place (NativeCondenser).
        3. Render it back with ``<block_N>`` markers (``Chunks.to_trajectory``).
        4. Sample one assistant turn from vLLM.
        5. Parse ``<tool_call>...</tool_call>`` blocks from the decoded text.
           - no tool call -> trajectory is terminal, break.
           - tool call(s) -> dispatch via ToolManager, append the
             ``role='tool'`` results back into the trajectory, loop.

The extract tool (``ExtractCompressed``) is re-bound every turn to the *pre-
compression* chunks of that turn, so the model can always recall original
text that step-3 compression pruned away.

Rewards are computed on the *full* multi-turn trajectory.  A new
``ExtractReward`` adds a behavioural regulariser that penalises excessive
tool use (see its docstring for the reverse-sigmoid curve shape).

For simplicity the GRPO training signal uses only the **final (terminal)
turn** of each rollout -- that is the turn producing the answer that got
scored.  Intermediate tool-calling turns are kept in the trajectory so the
reward sees the whole interaction, but they do not each generate their own
training datum; this keeps the group structure (``NUM_GENERATIONS`` per
prompt) identical to the non-agentic baseline.
"""
import json
import os
import re
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.reward import GSM8KAccuracyReward
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler
from twinkle.preprocessor.llm import GSM8KProcessor

from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser.native import NativeCondenser
from twinkle_agentic.reward.extract_reward import ExtractReward
from twinkle_agentic.tools.extract import ExtractCompressed
from twinkle_agentic.tools.tool_manager import ToolManager

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 1000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

# ---- Agentic rollout knobs ----
MAX_TURNS = int(os.environ.get('MAX_TURNS', 4))            # hard cap on tool-call turns
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 512))        # chars per chunk (NativeChunker)
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 0))    # sliding-window overlap
KEEP_RATIO = float(os.environ.get('KEEP_RATIO', 0.5))      # NativeCondenser target ratio

# ========== System Prompt ==========
SYSTEM_PROMPT = (
    'You are a helpful math assistant. Solve the problem with minimal but correct '
    'reasoning and put your final answer within \\boxed{}.\n\n'
    'CONTEXT COMPRESSION: Earlier parts of the conversation may be shown with '
    '<block_N>...</block_N> markers around each chunk. Some chunks have been '
    'shortened to save context. If you need the original (un-shortened) text of '
    'one or more blocks, call the tool below. Otherwise, answer directly.\n\n'
    'TOOL CALL FORMAT: Emit tool calls inside a single fenced block like this, '
    'then stop generating and wait for the tool result:\n'
    '<tool_call>\n'
    '{"name": "extract_compressed", "arguments": {"blocks": [1, 3]}}\n'
    '</tool_call>\n\n'
    'Prefer answering without tool calls whenever the compressed content is '
    'already sufficient -- each tool call reduces your reward.')

# ========== Tool-call parsing (Hermes / Qwen3 style) ==========
_TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, str]]:
    """Extract ``<tool_call>{...}</tool_call>`` blocks from an LLM completion.

    Robust to duplicated blocks, spurious whitespace, and Qwen's ``name`` vs
    the internal ``tool_name`` key.  Malformed JSON blocks are silently
    skipped rather than crashing the rollout.
    """
    calls: List[Dict[str, str]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get('name') or data.get('tool_name')
        args = data.get('arguments', {})
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        elif not isinstance(args, str):
            args = str(args)
        if name:
            calls.append({'tool_name': name, 'arguments': args})
    return calls


# ========== Reward Functions ==========
class GSM8KBrevityReward(Reward):
    """Brevity reward: rewards shorter final answers that contain a valid answer."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            messages = traj.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content') or ''
                    if not isinstance(completion, str):
                        # structured content -> concat text parts
                        completion = '\n'.join(
                            p.get('text', '') for p in completion
                            if isinstance(p, dict) and p.get('type') == 'text')
                    break

            has_answer = bool(
                re.search(r'\\boxed\{[^}]+\}', completion)
                or re.search(r'####\s*[\-\d,\.]+', completion)
            )

            if not has_answer:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= 300:
                    rewards.append(1.0)
                else:
                    rewards.append(max(0.0, 1.0 - (length - 300) / 3000))
        return rewards


# ========== Dataset ==========
def create_gsm8k_dataset():
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta('ms://modelscope/gsm8k', subset_name='main', split='train'))
    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=4096,
        truncation_strategy='delete', enable_thinking=False)
    dataset.map(GSM8KProcessor(system=SYSTEM_PROMPT))
    dataset.encode(add_generation_prompt=True)
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Sum of accuracy + brevity + extract-usage rewards.

    ``ExtractReward`` is the behavioural regulariser paired with
    :class:`ExtractCompressed`; it counts assistant ``tool_calls`` entries
    and applies a reverse-sigmoid decay.
    """
    accuracy = GSM8KAccuracyReward()(trajectories)
    brevity = GSM8KBrevityReward()(trajectories)
    extract = ExtractReward(midpoint=3.0, steepness=1.5)(trajectories)
    total = [a + b + e for a, b, e in zip(accuracy, brevity, extract)]
    return total, brevity, accuracy, extract


# ========== Agentic rollout ==========
class _Rollout:
    """Mutable bookkeeping for one prompt's multi-turn unroll."""
    __slots__ = ('trajectory', 'final_sequence', 'turns', 'done')

    def __init__(self, prompt_trajectory: Dict[str, Any]) -> None:
        self.trajectory: Dict[str, Any] = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        self.final_sequence = None  # SampledSequence of the terminal turn
        self.turns = 0
        self.done = False


def _append_terminal(r: _Rollout, decoded: str) -> None:
    r.trajectory['messages'].append({'role': 'assistant', 'content': decoded})
    r.done = True


def _append_tool_turn(
    r: _Rollout,
    decoded: str,
    tool_calls: List[Dict[str, str]],
    tool_mgr: ToolManager,
    turn_idx: int,
) -> None:
    r.trajectory['messages'].append({
        'role': 'assistant',
        'content': decoded,
        'tool_calls': tool_calls,
    })
    for i, tc in enumerate(tool_calls):
        result = tool_mgr.dispatch(tc)
        r.trajectory['messages'].append({
            'role': 'tool',
            'content': result,
            'tool_call_id': f'call_t{turn_idx}_i{i}',
        })


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    chunker: NativeChunker,
    condenser: NativeCondenser,
    max_turns: int,
) -> List[_Rollout]:
    """Batched multi-turn rollout with chunk-compress-tool loop.

    At each iteration we process only the rollouts that are still active,
    shrinking the batch as trajectories finish (either via a terminal
    response or by hitting ``max_turns``).
    """
    rollouts = [_Rollout(p) for p in prompts]

    for turn in range(max_turns):
        active = [r for r in rollouts if not r.done]
        if not active:
            break

        displays: List[Dict[str, Any]] = []
        tool_mgrs: List[ToolManager] = []
        for r in active:
            full_chunks = chunker.chunk(r.trajectory)
            compressed = condenser.condense(full_chunks)
            displays.append(compressed.to_trajectory())
            # Re-bind extract tool to *this* turn's pre-compression chunks so
            # block numbers the model sees in the prompt resolve correctly.
            tool_mgrs.append(ToolManager([ExtractCompressed(full_chunks)]))

        responses = sampler.sample(displays, sampling_params)

        for r, resp, tool_mgr in zip(active, responses, tool_mgrs):
            seq = resp.sequences[0]
            r.final_sequence = seq
            r.turns += 1
            decoded = seq.decoded or ''

            tool_calls = parse_tool_calls(decoded)
            if not tool_calls:
                _append_terminal(r, decoded)
            else:
                _append_tool_turn(r, decoded, tool_calls, tool_mgr, turn)

    # Anything still not done hit max_turns: accept the last completion as
    # its terminal answer so reward can still be computed.
    for r in rollouts:
        if not r.done and r.final_sequence is not None:
            _append_terminal(r, r.final_sequence.decoded or '')

    return rollouts


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    lora_config = LoraConfig(
        target_modules='all-linear',
        r=LORA_RANK,
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.05,
    )

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    else:
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=model_mesh,
            remote_group='model',
        )

    model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 8192,
            'max_lora_rank': 32,
            'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # Chunker / condenser live on the driver (pure-Python, no GPU).
    chunker = NativeChunker(
        model_id=MODEL_ID, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    condenser = NativeCondenser(keep_ratio=KEEP_RATIO)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=create_gsm8k_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95,
        # Stop after a tool_call so we can dispatch before the model keeps rambling.
        stop=['</tool_call>'],
    )

    optim_step = 0
    logger.info('Starting GSM8K GRPO training (agentic + context compression)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()
        expand_prompts: List[Dict[str, Any]] = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # ---- Multi-turn rollout (the agentic heart of this script) ----
        rollouts = run_agentic_rollouts(
            expand_prompts, sampler, sampling_params,
            chunker, condenser, max_turns=MAX_TURNS)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        all_trajectories: List[Dict[str, Any]] = []
        n_turns_per_rollout: List[int] = []
        for r in rollouts:
            seq = r.final_sequence
            if seq is None:
                # Should not happen if MAX_TURNS>=1, but guard to keep shapes aligned.
                continue
            all_input_data.append(seq.new_input_feature)
            all_old_logps.append([logprob[0][1] for logprob in (seq.logprobs or [])])
            all_completion_lengths.append(len(seq.tokens))
            all_trajectories.append(r.trajectory)
            n_turns_per_rollout.append(r.turns)

        total_rewards, brevity_rewards, accuracy_rewards, extract_rewards = compute_rewards(
            all_trajectories)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'brevity': brevity_rewards,
                'accuracy': accuracy_rewards,
                'extract': extract_rewards,
            },
        )

        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
            mb_inputs = all_input_data[mb_start:mb_end]
            mb_old_logps = all_old_logps[mb_start:mb_end]
            mb_advantages = advantages[mb_start:mb_end]

            model.forward_backward(
                inputs=mb_inputs,
                old_logps=mb_old_logps,
                advantages=mb_advantages,
                micro_batch_size=MICRO_BATCH_SIZE,
            )
            model.clip_grad_and_step()
            optim_step += 1

            if optim_step >= MAX_STEPS:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'math-grpo-tools-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        # Rollout depth is a useful diagnostic: if it collapses to 1 every step
        # the policy has stopped using tools entirely.
        if n_turns_per_rollout:
            log_dict['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('math-grpo-tools-final')


if __name__ == '__main__':
    main()
