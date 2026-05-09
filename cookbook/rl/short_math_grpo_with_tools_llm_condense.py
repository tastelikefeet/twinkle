import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Callable

import swanlab
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import Message, SamplingParams, Trajectory, ToolCall
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.preprocessor.base import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser import (
    FrozenContext,
    LLMPassageCondenser,
    build_initial_rollout_states,
    make_compression_trajectory_builder,
    strip_block_echoes,
)
from twinkle_agentic.rollout import Rollout, run_agentic_rollouts
from twinkle_agentic.reward import HotpotQACoTReward, HotpotQAToolExploreReward, HotpotQAF1Reward
from twinkle_agentic.tools import ExtractCompressed, ToolManager

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 10))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 0))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

MAX_TURNS = int(os.environ.get('MAX_TURNS', 6))
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1024))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 0))

COMPRESS_TEMPERATURE = float(os.environ.get('COMPRESS_TEMPERATURE', 0.4))
COMPRESS_RATIO = float(os.environ.get('COMPRESS_RATIO', 4.0))
COMPRESS_MAX_TOKENS = int(os.environ.get('COMPRESS_MAX_TOKENS', 256))
COMPRESS_MIN_CHARS = int(os.environ.get('COMPRESS_MIN_CHARS', 200))

HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# Reward weights
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.0))
COT_REWARD_WEIGHT = float(os.environ.get('COT_REWARD_WEIGHT', 0.5))
TOOL_BONUS_WEIGHT = float(os.environ.get('TOOL_BONUS_WEIGHT', 0.1))

WRONG_IDS_FILE = os.environ.get('WRONG_IDS_FILE', '')

_ROLLOUT_TRACE_PATH = os.environ.get('ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')

SYSTEM_PROMPT = """You are a careful multi-hop QA assistant.

## Compressed Context
The context you receive is **compressed**. Each paragraph is wrapped in \
<block_N>...</block_N> and displayed as a Markdown summary with three sections:
- **Summary**: one-sentence overview of the block
- **Key**: bulleted salient facts
- **More**: keywords hinting at details hidden in the full text

Because the context is compressed, critical details may not be immediately \
visible. You are strongly encouraged to call the `extract_compressed` tool \
to expand blocks that likely contain the answer.

## Workflow

### Phase 1 — Scan and Decide
Step 1: Read each block's Summary and Key facts to get an overview.
Step 2: Check the More keywords to judge whether hidden details are needed.
Step 3: Decide which blocks to expand, then call `extract_compressed`.

### Phase 2 — Reason and Answer
After the tool returns the full text, continue stepping through the evidence:
Step N:   From block X, I learn that [fact A].
Step N+1: From block Y, I need to call `extract_compressed` to get more information, because this block is related to...
Step N+2: Combining these, the answer is ...
\\boxed{answer}

You may call `extract_compressed` several times to expand more blocks if the information is not enough, only answer the question if you are sure about the facts.

## Tool Call Format
<tool_call>
<function=extract_compressed>
<parameter=blocks>
[1, 3]
</parameter>
</function>
</tool_call>

## Output Format
End your final response with \\boxed{answer}, e.g. \\boxed{Delhi}.
Keep the boxed text short: a name, entity, date, or "yes"/"no".
Answers not inside \\boxed{} will not be scored."""

COMPRESS_SYSTEM_PROMPT = f"""You are a passage compressor. Given a single paragraph, output a compact Markdown summary using EXACTLY these three sections:

**Summary**: One sentence — state the main subject, topic, and scope.
**Key Facts**: Some facts covering the most critical facts: entities, relations, numbers, and dates.
**More**: Comma-separated keywords for secondary details not captured above (minor entities, extra attributes, alternate names, peripheral dates).

## Rules
- Target ~{int(round(100 / COMPRESS_RATIO))}% of the original length (compression ratio ~{COMPRESS_RATIO:g}×).
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


_F1_REWARD: Optional[HotpotQAF1Reward] = HotpotQAF1Reward()
_COT_REWARD: Optional[HotpotQACoTReward] = HotpotQACoTReward()
_TOOL_EXPLORE_REWARD: Optional[HotpotQAToolExploreReward] = HotpotQAToolExploreReward()


def compute_rewards(trajectories: List[Dict[str, Any]]):
    f1 = _F1_REWARD(trajectories)
    cot = _COT_REWARD(trajectories)
    tool_explore = _TOOL_EXPLORE_REWARD(trajectories)
    total = [
        F1_REWARD_WEIGHT * a + COT_REWARD_WEIGHT * c + TOOL_BONUS_WEIGHT * te
        for a, c, te in zip(f1, cot, tool_explore)
    ]
    return total, f1, cot, tool_explore


class HotpotQAProcessor(Preprocessor):
    def __init__(self, system: str = SYSTEM_PROMPT, levels=None):
        self.system = system
        self.levels = levels

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = [r for r in rows if r is not None]
        rows = self.map_row_to_col(rows)
        return rows

    @staticmethod
    def _format_context(context: Dict[str, Any]) -> str:
        titles = context.get('title', []) or []
        sentences = context.get('sentences', []) or []
        lines = []
        for i, (title, sents) in enumerate(zip(titles, sentences), start=1):
            if isinstance(sents, list):
                body = ' '.join(s.strip() for s in sents if s and s.strip())
            else:
                body = str(sents).strip()
            lines.append(f'[{i}] {title}: {body}')
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Trajectory]:
        if self.levels is not None and (row.get('level') or '').strip().lower() not in self.levels:
            return None
        question = row['question']
        answer = row.get('answer', '') or ''
        context_block = self._format_context(row.get('context', {}) or {})
        user_msg = f'Question: {question}\n\nContext:\n\n{context_block}'
        messages = [
            Message(role='system', content=self.system),
            Message(role='user', content=user_msg),
        ]
        return Trajectory(messages=messages, user_data=[('ground_truth', answer.strip())])


def create_hotpotqa_dataset() -> Dataset:
    dataset = Dataset()
    dataset.add_dataset(DatasetMeta(
        'hf://hotpotqa/hotpot_qa', subset_name='fullwiki', split='train'))

    _wrong_ids_path = WRONG_IDS_FILE.strip()
    if _wrong_ids_path:
        with open(_wrong_ids_path, 'r', encoding='utf-8') as fh:
            _ids = frozenset(ln.strip() for ln in fh if ln.strip())
        if _ids:
            _key = next(iter(dataset.datasets.keys()))
            _before = len(dataset.datasets[_key])
            dataset.datasets[_key] = dataset.datasets[_key].filter(
                lambda row: row.get('id') in _ids)
            dataset.dataset = dataset.datasets[_key]
            logger.info(f'[WRONG_IDS_FILE] {_wrong_ids_path}: {_before} -> {len(dataset.dataset)} rows')

    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    _HOTPOTQA_COLS = ['id', 'question', 'answer', 'type', 'level',
                      'supporting_facts', 'context']
    dataset.map(HotpotQAProcessor(system=SYSTEM_PROMPT, levels=['hard']), remove_columns=_HOTPOTQA_COLS)
    return dataset


# Matches a LaTeX ``\boxed{...}`` final-answer marker — used to flag
# rollouts that never committed an answer. Brace-balanced is overkill for
# a logging heuristic; a non-greedy ``[^}]*`` is good enough.
_BOXED_RE = re.compile(r'\\boxed\{[^}]*\}')


def _last_assistant_text(trajectory: Dict[str, Any]) -> Optional[str]:
    """Return the text of the last ``assistant`` message, or ``None``."""
    for m in reversed(trajectory.get('messages', [])):
        if m.get('role') == 'assistant':
            return m.get('content')
    return None


def _make_dump_rollout_trace(path: str):
    """Factory returning a ``TurnHook`` that appends each turn to JSONL.

    The returned callable matches the
    ``Callable[[int, List[Rollout], List[Dict], List[Any]], None]``
    signature expected by :func:`run_agentic_rollouts`. Best-effort:
    any per-record or per-file error is swallowed so tracing cannot
    break training.
    """
    if not path:
        def _noop(*args, **kwargs) -> None:
            return None
        return _noop

    def _hook(turn: int, active: List[Rollout],
              trajectories: List[Dict[str, Any]],
              responses: List[Any]) -> None:
        if not active:
            return
        try:
            records: List[str] = []
            for idx, r in enumerate(active):
                try:
                    resp = responses[idx] if idx < len(responses) else None
                    tcc = sum(
                        len(m.get('tool_calls') or [])
                        for m in r.trajectory.get('messages', [])
                        if m.get('role') == 'assistant')
                    last_decoded = ''
                    if resp and getattr(resp, 'sequences', None):
                        last_decoded = resp.sequences[0].decoded or ''
                    match = _BOXED_RE.search(_last_assistant_text(r.trajectory) or '') if r.done else None
                    final_answer = match.group(0) if match else ''
                    record = {
                        'ts': time.time(), 'turn': turn,
                        'group_size': len(active), 'picked_idx': idx,
                        'rollout_id': id(r), 'tool_call_count': tcc,
                        'done': bool(r.done),
                        'compressed': (trajectories[idx]
                                       if idx < len(trajectories) else None),
                        'last_decoded': last_decoded,
                        'final_answer': final_answer,
                    }
                    records.append(
                        json.dumps(record, ensure_ascii=False, default=str))
                except Exception:
                    pass
            if records:
                with open(path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(records) + '\n')
        except Exception:
            pass

    return _hook


def _compute_rollout_diagnostics(
    rollouts: List[Rollout],
    n_turns_per_rollout: List[int],
    all_trajectories: List[Dict[str, Any]],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if n_turns_per_rollout:
        out['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)

    _max_prompt_tok = 0
    for r in rollouts:
        seq = r.sequence
        if seq is None:
            continue
        feat = getattr(seq, 'new_input_feature', None) or {}
        ids = feat.get('input_ids') if isinstance(feat, dict) else None
        if ids:
            prompt_len = max(0, len(ids) - len(seq.tokens or []))
            if prompt_len > _max_prompt_tok:
                _max_prompt_tok = prompt_len
    out['max_prompt_tokens'] = _max_prompt_tok

    if all_trajectories:
        tool_counts = [
            sum(len(m.get('tool_calls') or [])
                for m in t.get('messages', []) if m.get('role') == 'assistant')
            for t in all_trajectories]
        out['avg_tool_calls'] = sum(tool_counts) / len(tool_counts)
        out['tool_use_rate'] = sum(1 for c in tool_counts if c > 0) / len(tool_counts)
        n_no_boxed = sum(
            0 if _BOXED_RE.search(_last_assistant_text(t) or '') else 1
            for t in all_trajectories)
        out['no_boxed_rate'] = n_no_boxed / len(all_trajectories)
    return out


def main():
    swanlab.init(project='twinkle')

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS,
                       groups=device_groups, lazy_collect=False)

    if _ROLLOUT_TRACE_PATH:
        try:
            open(_ROLLOUT_TRACE_PATH, 'w').close()
        except OSError:
            pass

    logger.info('Building HotpotQA dataset')
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    batches_per_epoch = max(1, len(_prebuilt_dataset) // GLOBAL_BATCH_SIZE)
    EXPECTED_AVG_TURNS = int(os.environ.get('EXPECTED_AVG_TURNS', 3))
    optim_steps_per_batch = max(1, (GLOBAL_BATCH_SIZE * NUM_GENERATIONS * EXPECTED_AVG_TURNS
                                     + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
    steps_per_epoch = batches_per_epoch * optim_steps_per_batch
    derived_total_steps = NUM_EPOCHS * steps_per_epoch
    total_steps = min(MAX_STEPS, derived_total_steps) if MAX_STEPS > 0 else derived_total_steps
    logger.info('Training horizon: %d steps (%d epochs × %d batches × %d steps/batch)',
                total_steps, NUM_EPOCHS, batches_per_epoch, optim_steps_per_batch)

    lora_config = LoraConfig(
        target_modules='all-linear', r=LORA_RANK,
        lora_alpha=LORA_RANK * 2, lora_dropout=0.05)

    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(
            model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model',
            mixed_precision='bf16', variable_seq_lengths=True)
    else:
        model = TransformersModel(
            model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    model.add_adapter_to_model(ADAPTER_NAME, lora_config,
                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=total_steps, max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=total_steps, eta_min=0)

    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor, padding_free=True)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)

    model.add_metric('GRPOMetric', is_training=True)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8, 'max_model_len': 32768,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False, max_length=HOTPOTQA_MAX_LENGTH)
    rollout_template = Qwen3_5Template(MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH)
    condenser = LLMPassageCondenser(
        sampler=sampler,
        sampling_params=SamplingParams(
            max_tokens=COMPRESS_MAX_TOKENS, num_samples=1,
            temperature=COMPRESS_TEMPERATURE, top_p=0.9),
        system_prompt=COMPRESS_SYSTEM_PROMPT,
        min_chars=COMPRESS_MIN_CHARS,
        skip_roles=('system', 'tool', 'assistant'),
        template=rollout_template,
    )

    def _build_tool_manager(r: Rollout) -> Callable[[Dict[str, Any]], str]:
        fc: FrozenContext = r.state['frozen']
        return ToolManager([
            ExtractCompressed(
                fc.get_full_chunks(),
                displayed_to_full=fc.displayed_to_full(),
            )
        ])

    on_turn_hook = _make_dump_rollout_trace(_ROLLOUT_TRACE_PATH)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    chunker = NativeChunker(
        model_id=MODEL_ID, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        passage_boundary_re=r'^\[\d+\]\s+')

    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE, min_batch_size=GLOBAL_BATCH_SIZE)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95, stop=['</tool_call>'])

    optim_step = 0
    logger.info('Starting HotpotQA GRPO training (LLM condenser variant)')

    def _epoch_cycle(dl, n_epochs):
        for ep in range(1, n_epochs + 1):
            logger.info(f'=== Epoch {ep}/{n_epochs} (step={optim_step}/{total_steps}) ===')
            for batch in dl:
                yield batch

    for batch in _epoch_cycle(dataloader, NUM_EPOCHS):
        if optim_step >= total_steps:
            break

        metrics.reset()
        expand_prompts = [p for prompt in batch for p in [prompt] * NUM_GENERATIONS]

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        initial_states = build_initial_rollout_states(
            expand_prompts, chunker, condenser)

        rollouts = run_agentic_rollouts(
            expand_prompts, sampler, sampling_params,
            _build_tool_manager, rollout_template,
            max_turns=MAX_TURNS,
            trajectory_builder=make_compression_trajectory_builder(chunker, condenser),
            initial_states=initial_states,
            output_sanitizers=[strip_block_echoes],
            min_batch_size=GLOBAL_BATCH_SIZE,
            on_turn=on_turn_hook)

        all_trajectories = [r.trajectory for r in rollouts]
        n_turns_per_rollout = [r.turns for r in rollouts]
        per_rollout_completion_length = [
            len(r.sequence.tokens) for r in rollouts]

        total_rewards, f1_rewards, cot_rewards, tool_explore_rewards = \
            compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=per_rollout_completion_length,
            rewards={'total': total_rewards, 'f1': f1_rewards,
                     'cot': cot_rewards, 'tool_explore': tool_explore_rewards})

        rollout_advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        all_input_data: List[Any] = []
        all_old_logps: List[List[float]] = []
        advantages: List[float] = []
        for r, adv in zip(rollouts, rollout_advantages):
            all_input_data.append(r.sequence.new_input_feature)
            all_old_logps.append([lp[0][1] for lp in (r.sequence.logprobs or [])])
            advantages.append(adv)

        total_completions = len(all_input_data)
        aligned_completions = (total_completions // MODEL_GPUS) * MODEL_GPUS
        if aligned_completions < total_completions:
            logger.info(
                '[dp-align] dropping %d tail sample(s): total=%d -> aligned=%d (dp=%d)',
                total_completions - aligned_completions,
                total_completions, aligned_completions, MODEL_GPUS)
        for mb_start in range(0, aligned_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, aligned_completions)
            test = all_input_data[mb_start:mb_end]
            if len(test) < MICRO_BATCH_SIZE:
                print()
            model.forward_backward(
                inputs=all_input_data[mb_start:mb_end],
                old_logps=all_old_logps[mb_start:mb_end],
                advantages=advantages[mb_start:mb_end],
                micro_batch_size=MICRO_BATCH_SIZE)
            model.clip_grad_and_step()
            optim_step += 1
            if optim_step >= total_steps:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'hotpotqa-grpo-tools-llmcondense-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        log_dict.update(_compute_rollout_diagnostics(
            rollouts, n_turns_per_rollout, all_trajectories))
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{total_steps}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-llmcondense-final')


if __name__ == '__main__':
    main()
