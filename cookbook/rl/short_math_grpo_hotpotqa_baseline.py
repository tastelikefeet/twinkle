"""HotpotQA GRPO training **baseline** -- full-context, no compression, no tools.

Companion to ``short_math_grpo_with_tools.py``.  Same task (HotpotQA multi-hop
QA), same reward scaffolding (F1 + format gate + length penalty), same GRPO
outer loop, but strips everything that makes the tool-augmented variant a
moving target:

  * **no chunking / condensing** -- the full 10 paragraphs are rendered
    verbatim into the user turn.  Context window is bounded only by
    ``HOTPOTQA_MAX_LENGTH`` at dataset encode time.
  * **no tool calls / no multi-turn rollout** -- each prompt is sampled
    exactly once; the assistant is expected to emit the boxed answer in
    a single turn.
  * **no ``<block_N>`` markers, no ExtractCompressed** -- the model sees
    the raw paragraphs with their ``[N] Title: body`` numbering only as
    a human-readable index, not as a tool address.
  * **no tool_use / tool_success rewards** -- only F1 (primary),
    format (diagnostic, weight 0), and length_pen (safety net) remain.

This file is intentionally **kept parallel** to the tool-augmented variant:
identical data preprocessor, identical Qwen3.5 template wiring, identical
GRPO hyperparameters.  It exists so that any accuracy delta observed in
the tool-augmented run can be attributed to tool use + compression rather
than to downstream training-infra drift.
"""
import json
import os
import random
import re
import string
import time
from collections import Counter
from typing import Any, Dict, List, Tuple

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import Message, SamplingParams, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.base import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.reward.base import Reward
from twinkle.sampler import vLLMSampler

import swanlab
swanlab.init(
    project='twinkle',
)

logger = get_logger()

# ========== Configuration ==========
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 5e-8))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 5000))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 4))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

# ---- Reward-balancing knobs (mirrors the tool-augmented variant for the
# subset of rewards that survive in the baseline) ----
# F1 carries the whole learning signal here -- there is no tool-use branch
# to reward, so the weight can (and should) be higher than in the tool run.
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
# Kept at 0.0 as a monitoring-only signal; the format bonus is 100% saturated
# once the model learns to emit \boxed{} and therefore contributes pure
# noise to GRPO within-group advantages.
FORMAT_REWARD_WEIGHT = float(os.environ.get('FORMAT_REWARD_WEIGHT', 0.0))
# Safety net against repetition loops; activates only when the terminal
# message exceeds ``FORMAT_MAX_CHARS`` and grows linearly to -1 at the
# ``MAX_NEW_TOKENS * 4`` char ceiling.
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
# Single-turn baseline: legitimate answers fit well under 5000 chars.  Keep
# the threshold aligned with the tool-augmented run so both share the same
# length-shape diagnostic.
FORMAT_MAX_CHARS = int(os.environ.get('FORMAT_MAX_CHARS', 5000))
# Round-5 fix: F1-gate for length_pen.  Only penalise length overflow on
# rollouts whose F1 is above this threshold, so the GRPO within-group
# advantage cannot reward "short-and-wrong" over "long-and-wrong".  Set
# to 0.0 to disable (legacy behaviour).
LENGTH_PEN_F1_GATE = float(os.environ.get('LENGTH_PEN_F1_GATE', 0.3))
# GRPO homogeneous-group filter: mask samples whose per-sample max
# |advantage| is below this threshold.  Prevents KL drift and numerical-
# noise-level pseudo-advantages on zero-variance groups.
GRPO_HOMOGENEOUS_THRESHOLD = float(
    os.environ.get('GRPO_HOMOGENEOUS_THRESHOLD', 1e-4))

# ---- HotpotQA dataset encode knobs ----
HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# ========== System Prompt ==========
# Kept deliberately short and tool-free: the baseline must NOT see any
# ``<tool_call>`` / ``<block_N>`` / ``extract_compressed`` vocabulary, or
# the comparison with the tool-augmented run would be polluted by the model
# hallucinating tool calls into a pipeline that has no dispatcher.
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Give a short factual answer (a name, '
    'entity, date, or "yes"/"no") inside \\boxed{}. Do not include extra '
    'words in the box. Read the paragraphs carefully before answering; '
    'the answer can always be derived from them.')


# ========== Assistant output sanitisation ==========
# Single-turn baseline still benefits from stripping stray ``<block_N>``
# / ``[[#N]]`` tokens that the pretrained model may emit by accident (it
# has seen such patterns during pretraining).  The ``<tool_call>`` stripper
# is retained for defense-in-depth: a tool-call span leaking into training
# data would teach the model a format the baseline pipeline cannot honour.
_TOOL_CALL_STRIP_RE = re.compile(r'<tool_call>.*?(?:</tool_call>|\Z)', re.DOTALL)
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_FAKE_CITE_RE = re.compile(r'\[\[#\d+\]\]')


def _clean_assistant_output(text: str) -> str:
    text = _TOOL_CALL_STRIP_RE.sub('', text or '')
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text)
    text = _FAKE_CITE_RE.sub('', text)
    return text.rstrip()


# ========== Answer extraction ==========
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')


def _extract_final_answer(completion: str) -> str:
    """Pull the predicted answer out of a completion.

    Prefers the last ``\\boxed{...}`` span; falls back to the last non-empty
    line (stripped) so partially-formatted completions still get a graded F1.
    """
    matches = _BOXED_RE.findall(completion or '')
    if matches:
        return matches[-1].strip()
    for line in reversed((completion or '').splitlines()):
        line = line.strip()
        if line:
            return line
    return ''


def _last_assistant_text(traj: Dict[str, Any]) -> str:
    for msg in reversed(traj.get('messages', [])):
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content') or ''
        if isinstance(content, str):
            return content
        return '\n'.join(
            p.get('text', '') for p in content
            if isinstance(p, dict) and p.get('type') == 'text')
    return ''


# ========== Reward Functions ==========
_F1_REWARD = None
_FORMAT_REWARD = None
_LENGTH_PENALTY = None


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lowercase, strip punct/articles/extra ws."""
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def _f1_score(prediction: str, gold: str) -> Tuple[float, float]:
    """Word-level F1 and EM between normalized prediction and gold."""
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        em = float(pred_tokens == gold_tokens)
        return em, em
    em = float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    return 2 * p * r / (p + r), em


class HotpotQAF1Reward(Reward):
    """Token-level F1 on the extracted \\boxed{} answer vs ground truth."""

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            gold = ''
            for key, val in traj.get('user_data', []) or []:
                if key == 'ground_truth':
                    gold = val or ''
                    break
            pred = _extract_final_answer(_last_assistant_text(traj))
            f1, _em = _f1_score(pred, gold)
            rewards.append(f1)
        return rewards


class HotpotQAFormatReward(Reward):
    """+1 if the final assistant turn contains a \\boxed{...} span, else 0.

    Gated by :data:`FORMAT_MAX_CHARS` to prevent the policy from padding
    garbage around a single ``\\boxed{...}`` to satisfy both signals.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            if _BOXED_RE.search(text) and len(text) <= FORMAT_MAX_CHARS:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards


class HotpotQALengthPenalty(Reward):
    """Negative reward proportional to terminal-message length overflow.

    Returns 0 if the final assistant message is within :data:`FORMAT_MAX_CHARS`,
    else a linearly growing penalty that reaches -1 when the message is
    ``MAX_NEW_TOKENS * 4`` chars (empirical 4 chars/token ceiling).
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        budget = max(1, MAX_NEW_TOKENS * 4 - FORMAT_MAX_CHARS)
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            overflow = max(0, len(text) - FORMAT_MAX_CHARS)
            rewards.append(-min(1.0, overflow / budget))
        return rewards


# ========== Dataset ==========
class HotpotQAProcessor(Preprocessor):
    """Render a HotpotQA row into a single-turn prompt Trajectory.

    Identical to the tool-augmented variant's preprocessor *except* this
    version has no downstream chunker/condenser, so the ``[N] Title: body``
    numbering here is purely a human-readable index for the reader's eye --
    there is no ``extract_compressed(N)`` tool that resolves to it.

    ``ground_truth`` (for the F1/EM reward) is stashed in ``user_data``.
    """

    def __init__(self, system: str = SYSTEM_PROMPT):
        self.system = system

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        titles = context.get('title', []) or []
        sentences = context.get('sentences', []) or []
        lines = []
        for i, (title, sents) in enumerate(zip(titles, sentences), start=1):
            body = ''.join(sents) if isinstance(sents, list) else str(sents)
            lines.append(f'[{i}] {title}: {body.strip()}')
        # Blank-line separator between passages.  Not strictly required for
        # the baseline (no chunker to respect paragraph boundaries), but
        # kept aligned with the tool-augmented variant so that encoded
        # token counts -- and therefore any context-length ceilings -- are
        # comparable across the two runs.
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '') or ''
        context_block = self._format_context(row.get('context', {}) or {})

        user_msg = (
            f'Question: {question}\n\n'
            f'Context:\n\n{context_block}')

        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=user_msg),
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', answer.strip())],
        )


def create_hotpotqa_dataset() -> Dataset:
    """Build + encode the HotpotQA dataset in-process.

    Mirrors the tool-augmented variant so fingerprints / cache files line up.
    """
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(
        'hf://hotpotqa/hotpot_qa', subset_name='fullwiki', split='train'))
    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    dataset.map(HotpotQAProcessor(system=SYSTEM_PROMPT))
    dataset.encode(
        add_generation_prompt=True,
        load_from_cache_file=True,
        num_proc=HOTPOTQA_NUM_PROC,
    )
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Weighted sum of F1 + format (diagnostic) + length_pen.

    Return tuple: ``(total, format, f1, length_pen)``.  No extract /
    tool_use / tool_success vectors -- the baseline has no tool pipeline.
    """
    global _F1_REWARD, _FORMAT_REWARD, _LENGTH_PENALTY
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _FORMAT_REWARD = HotpotQAFormatReward()
        _LENGTH_PENALTY = HotpotQALengthPenalty()
    accuracy = _F1_REWARD(trajectories)
    fmt = _FORMAT_REWARD(trajectories)
    length_pen_raw = _LENGTH_PENALTY(trajectories)

    # Round-5 fix: F1-gated length_pen (see tool-augmented variant for the
    # full post-mortem).  Long-and-wrong rollouts no longer attract a
    # negative length_pen, so "short-is-better" can't sneak in as a false
    # advantage signal when F1 is flat.
    if LENGTH_PEN_F1_GATE > 0.0:
        length_pen = [
            lp if acc > LENGTH_PEN_F1_GATE else 0.0
            for lp, acc in zip(length_pen_raw, accuracy)
        ]
    else:
        length_pen = length_pen_raw

    total = [
        F1_REWARD_WEIGHT * a
        + FORMAT_REWARD_WEIGHT * f
        + LENGTH_PENALTY_WEIGHT * lp
        for a, f, lp in zip(accuracy, fmt, length_pen)
    ]
    return total, fmt, accuracy, length_pen


# ========== Rollout trace dump ==========
# Post-mortem JSONL log, one line per (random) rollout.  Lighter than the
# tool-augmented variant's trace (no compressed/full chunk dump, no turn
# progression) but keeps identical per-component reward snapshots so the
# baseline can be analysed by the same ``_trace_analyze.py`` tooling.
_ROLLOUT_TRACE_PATH = os.environ.get(
    'ROLLOUT_TRACE_PATH', 'rollout_trace_baseline.jsonl')


def _dump_random_rollout_trace(
    trajectories: List[Dict[str, Any]],
    responses: List[Any],
) -> None:
    """Append one random trajectory's post-rollout state to JSONL."""
    if not _ROLLOUT_TRACE_PATH or not trajectories:
        return
    idx = random.randrange(len(trajectories))
    traj = trajectories[idx]
    resp = responses[idx] if idx < len(responses) else None

    last_decoded = ''
    if resp is not None and getattr(resp, 'sequences', None):
        last_decoded = resp.sequences[0].decoded or ''

    final_answer = _extract_final_answer(_last_assistant_text(traj))

    reward_snapshot: Dict[str, float] = {}
    try:
        global _F1_REWARD, _FORMAT_REWARD, _LENGTH_PENALTY
        if _F1_REWARD is None:
            _F1_REWARD = HotpotQAF1Reward()
            _FORMAT_REWARD = HotpotQAFormatReward()
            _LENGTH_PENALTY = HotpotQALengthPenalty()
        traj_list = [traj]
        f1_val = _F1_REWARD(traj_list)[0]
        fmt_val = _FORMAT_REWARD(traj_list)[0]
        len_val = _LENGTH_PENALTY(traj_list)[0]
        total_val = (
            F1_REWARD_WEIGHT * f1_val
            + FORMAT_REWARD_WEIGHT * fmt_val
            + LENGTH_PENALTY_WEIGHT * len_val)
        reward_snapshot = {
            'f1': float(f1_val),
            'format': float(fmt_val),
            'length_pen': float(len_val),
            'total': float(total_val),
        }
    except Exception as e:  # pragma: no cover -- tracing must never crash training
        logger.warning('rollout trace reward snapshot failed: %s', e)
        reward_snapshot = {'error': repr(e)}

    record = {
        'ts': time.time(),
        'batch_size': len(trajectories),
        'picked_idx': idx,
        'done': True,
        'rewards': reward_snapshot,
        'last_decoded': last_decoded,
        'final_answer': final_answer,
    }
    try:
        with open(_ROLLOUT_TRACE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:  # pragma: no cover -- tracing must never crash training
        logger.warning('rollout trace dump failed: %s', e)


# ========== Single-turn rollout ==========
def run_single_turn_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
) -> Tuple[List[Any], List[Dict[str, Any]]]:
    """Dispatch all prompts once and pair each response with a trajectory.

    Replaces the agentic multi-turn loop.  Returns ``(responses, trajectories)``
    with index-aligned shapes so downstream bookkeeping (``all_input_data``,
    ``all_old_logps`` etc.) mirrors the tool-augmented variant and the same
    GRPO glue works unchanged.
    """
    # Each prompt already carries ``messages`` + ``user_data``.  The sampler
    # accepts a list of such dicts directly; no chunk+condense preprocessing
    # is needed here because the baseline has no compression stage.
    responses = sampler.sample(list(prompts), sampling_params)

    trajectories: List[Dict[str, Any]] = []
    for prompt, resp in zip(prompts, responses):
        seq = resp.sequences[0]
        decoded = seq.decoded or ''
        trajectory: Dict[str, Any] = {
            'messages': list(prompt.get('messages', [])),
            'user_data': prompt.get('user_data', []),
        }
        trajectory['messages'].append(
            {'role': 'assistant', 'content': _clean_assistant_output(decoded)})
        trajectories.append(trajectory)

    return responses, trajectories


# ========== Main ==========
def main():
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=2, pp_size=2)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    logger.info('Building HotpotQA dataset (num_proc=%d, max_length=%d)',
                HOTPOTQA_NUM_PROC, HOTPOTQA_MAX_LENGTH)
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

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
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    # Baseline sampling -- no ``stop=['</tool_call>']`` because there is no
    # tool-call format in this pipeline; keep repetition_penalty / temperature
    # / top_p aligned with the tool-augmented variant so the sole confound
    # between the two runs is the rollout structure itself.
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.15, top_p=0.9,
        repetition_penalty=1.1,
    )

    optim_step = 0
    logger.info('Starting HotpotQA GRPO baseline (full-context, single-turn, no tools)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        metrics.reset()
        expand_prompts: List[Dict[str, Any]] = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=True)
        sampler.reset_prefix_cache()

        # ---- Single-turn rollout (the baseline's whole sampling phase) ----
        responses, all_trajectories = run_single_turn_rollouts(
            expand_prompts, sampler, sampling_params)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []
        for resp in responses:
            seq = resp.sequences[0]
            all_input_data.append(seq.new_input_feature)
            all_old_logps.append([logprob[0][1] for logprob in (seq.logprobs or [])])
            all_completion_lengths.append(len(seq.tokens))

        total_rewards, brevity_rewards, accuracy_rewards, length_penalties = (
            compute_rewards(all_trajectories))

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': brevity_rewards,
                'f1': accuracy_rewards,
                'length_pen': length_penalties,
            },
        )

        # Trace one random rollout per step (same cadence as the tool variant's
        # per-turn dump).  Useful for decoded-length / non-ASCII / reward
        # progression monitoring during baseline training.
        _dump_random_rollout_trace(all_trajectories, responses)

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
                model.save(f'hotpotqa-grpo-baseline-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict = {key: value for key, value in log_dict.items() if '_std' not in key}
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-baseline-final')


if __name__ == '__main__':
    main()
