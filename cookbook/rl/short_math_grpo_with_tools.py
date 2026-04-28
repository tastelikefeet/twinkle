"""HotpotQA GRPO training with context compression + tool-augmented multi-turn rollouts.

Built on top of ``short_math_grpo.py`` (the math variant).  Task has been
swapped from GSM8K math to **HotpotQA multi-hop QA** because that is where
context compression + extract tool actually earn their reward: each example
ships 10 paragraphs, only 2 are gold, so compressing distractors and
extracting the right blocks gives a real training signal.

The sampling phase runs an **agentic rollout loop** (unchanged from the
math variant):

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
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

from peft import LoraConfig

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
KEEP_RATIO = float(os.environ.get('KEEP_RATIO', 0.3))      # NativeCondenser target ratio

# ---- HotpotQA dataset encode knobs ----
HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# ========== System Prompt ==========
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Give a short factual answer (a name, '
    'entity, date, or "yes"/"no") inside \\boxed{}. Do not include extra '
    'words in the box.\n\n'
    'CONTEXT COMPRESSION: The provided paragraphs may be shown with '
    '<block_N>...</block_N> markers around each chunk. Some chunks have been '
    'shortened to save context. If you need the original (un-shortened) text '
    'of one or more blocks, call the tool below. Otherwise, answer directly.\n\n'
    'TOOL CALL FORMAT: Emit tool calls inside a single fenced block like '
    'this, then stop generating and wait for the tool result:\n'
    '<tool_call>\n'
    '{"name": "extract_compressed", "arguments": {"blocks": [1, 3]}}\n'
    '</tool_call>\n\n'
    'Prefer answering without tool calls whenever the compressed content is '
    'already sufficient -- each tool call reduces your reward.')

# ========== Tool-call parsing (Hermes / Qwen3 style) ==========
# Tolerant of a missing closing tag: vLLM strips the ``stop`` string from the
# decoded output by default, so a tool-calling turn ends at ``<tool_call>{...}``
# with the closing tag gone.  We accept either ``</tool_call>`` or end-of-string
# as the right boundary so the parser still works.
_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*(\{.*?\})\s*(?:</tool_call>|\Z)', re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract ``<tool_call>{...}</tool_call>`` blocks from an LLM completion.

    Robust to a missing trailing ``</tool_call>`` (vLLM stop-string stripping),
    duplicated blocks, spurious whitespace, and Qwen's ``name`` vs the internal
    ``tool_name`` key.  Malformed JSON blocks are silently skipped rather than
    crashing the rollout.

    ``arguments`` is returned as a dict (or empty dict) -- :class:`ToolManager`
    accepts dicts directly, so there is no need to re-serialise to a string.
    """
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        name = data.get('name') or data.get('tool_name')
        if not name:
            continue
        args = data.get('arguments', {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
        elif not isinstance(args, dict):
            args = {}
        calls.append({'tool_name': name, 'arguments': args})
    return calls


# Strip tool-call spans (open tag through close tag OR EOS) from the assistant
# text before we stash it on the trajectory, so next-turn rendering does not
# double-print them (once from the raw text, once from the ``tool_calls`` field).
_TOOL_CALL_STRIP_RE = re.compile(
    r'<tool_call>.*?(?:</tool_call>|\Z)', re.DOTALL)


def _strip_tool_call_spans(text: str) -> str:
    return _TOOL_CALL_STRIP_RE.sub('', text or '').rstrip()


# ========== HotpotQA preprocessor ==========
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
        # Structured content -> concat text parts.
        return '\n'.join(
            p.get('text', '') for p in content
            if isinstance(p, dict) and p.get('type') == 'text')
    return ''


# ========== Reward Functions ==========
# Module-level singletons lazily instantiated by ``compute_rewards`` on first
# call.  Declared here (rather than inside the function) so the ``global``
# statement's ``if _F1_REWARD is None`` read does not ``NameError`` on step 1.
_F1_REWARD = None
_FORMAT_REWARD = None
_EXTRACT_REWARD = None


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
    """Token-level F1 on the extracted \\boxed{} answer vs ground truth.

    Uses F1 rather than strict EM so partial matches still provide a gradient
    during RL.  Returns 0 if no answer was produced.
    """

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

    Keeps the signal that taught GSM8K models to respect the answer schema.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        return [
            1.0 if _BOXED_RE.search(_last_assistant_text(t) or '') else 0.0
            for t in trajectories
        ]


# ========== Dataset ==========
class HotpotQAProcessor(Preprocessor):
    """Render a HotpotQA row into a prompt-only Trajectory.

    HotpotQA schema (after HF ``datasets`` flattening)::

        id:                str
        question:          str
        answer:            str                    (may be 'yes' / 'no')
        type:              'bridge' | 'comparison'
        level:             'easy' | 'medium' | 'hard'
        supporting_facts:  {'title': List[str], 'sent_id': List[int]}
        context:           {'title': List[str], 'sentences': List[List[str]]}

    We ship the ten (title, paragraph) pairs to the model as a numbered list
    so that ``NativeChunker`` naturally slices them into one block per
    paragraph, and the ``extract_compressed`` tool can recall a specific
    paragraph by its block id.

    ``ground_truth`` (for the F1/EM reward) is stashed in ``user_data``.

    Note: ``user_data`` is flattened into an Arrow table downstream, so every
    tuple value in it must have the *same* primitive type across rows --
    heterogeneous types (e.g. mixing ``str`` with ``list[str]``) trigger
    ``ArrowTypeError`` during ``dataset.map``.  That is why we do NOT stash
    ``supporting_facts.title`` here; if a supporting-fact auxiliary reward is
    ever added, JSON-encode the list to a string first.
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
        return '\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '') or ''
        context_block = self._format_context(row.get('context', {}) or {})

        user_msg = (
            f'Question: {question}\n\n'
            f'Context:\n{context_block}')

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

    ``dataset.encode(num_proc=HOTPOTQA_NUM_PROC)`` uses a
    ``multiprocess.Pool`` whose start method has already been forced to
    ``spawn`` by ``twinkle.dataset.base`` at import time.  Spawn workers boot
    fresh interpreters, so they do NOT inherit the parent's CUDA state.
    ``Qwen3_5Template`` also caches its rope-index function at module level
    rather than on the instance, so the template pickles deterministically and
    ``load_from_cache_file=True`` can actually hit on re-runs.
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
    """Sum of F1 + format + extract-usage rewards.

    ``ExtractReward`` is the behavioural regulariser paired with
    :class:`ExtractCompressed`; it counts assistant ``tool_calls`` entries
    and applies a reverse-sigmoid decay.
    """
    global _F1_REWARD, _FORMAT_REWARD, _EXTRACT_REWARD
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _FORMAT_REWARD = HotpotQAFormatReward()
        _EXTRACT_REWARD = ExtractReward(midpoint=3.0, steepness=1.5)
    accuracy = _F1_REWARD(trajectories)
    fmt = _FORMAT_REWARD(trajectories)
    extract = _EXTRACT_REWARD(trajectories)
    total = [a + f + e for a, f, e in zip(accuracy, fmt, extract)]
    return total, fmt, accuracy, extract


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
    tool_calls: List[Dict[str, Any]],
    tool_mgr: ToolManager,
    turn_idx: int,
) -> None:
    # Strip the raw <tool_call> spans from the text; the structured
    # ``tool_calls`` field is the canonical source, and the template re-renders
    # them into the prompt on the next turn.  Leaving both present would
    # duplicate (or malform, when the stop tag was stripped) the tool-call
    # markup.
    r.trajectory['messages'].append({
        'role': 'assistant',
        'content': _strip_tool_call_spans(decoded),
        'tool_calls': [
            {'tool_name': tc['tool_name'],
             'arguments': (tc['arguments']
                           if isinstance(tc['arguments'], str)
                           else json.dumps(tc['arguments'], ensure_ascii=False))}
            for tc in tool_calls
        ],
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
    min_batch_size: int = 1,
) -> List[_Rollout]:
    """Batched multi-turn rollout with chunk-compress-tool loop.

    At each iteration we process only the rollouts that are still active,
    shrinking the batch as trajectories finish (either via a terminal
    response or by hitting ``max_turns``).

    ``min_batch_size`` guards against the late-turn case where ``active``
    shrinks below the sampler's DP world size.  The infra-side
    ``_check_uniform`` raises ``ValueError('Batch too small for N workers,
    some ranks have no data')`` when any sampler rank would otherwise get
    zero items, so we pad ``displays`` up to ``min_batch_size`` with the
    first live display (a safe no-op: we slice the padded responses off
    before using them).
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

        # Pad up to ``min_batch_size`` so every sampler DP rank receives at
        # least one prompt.  The padded entries are throw-away duplicates of
        # the first live display; we slice them off from ``responses`` below.
        n_active = len(displays)
        if n_active < min_batch_size:
            pad = displays[0]
            displays = displays + [pad] * (min_batch_size - n_active)

        responses = sampler.sample(displays, sampling_params)
        # Drop padded responses before pairing with real rollouts.
        responses = responses[:n_active]

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

    # Anything still not done hit max_turns with a trailing tool call.  The
    # assistant message was already appended by ``_append_tool_turn`` in the
    # final iteration, so we MUST NOT append it again -- just mark the rollout
    # as done.  Reward functions look up the last assistant message and will
    # correctly penalise the missing \boxed{} answer.
    for r in rollouts:
        r.done = True

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

    # Build and encode the HotpotQA dataset in-process.  Safe because
    # ``twinkle.dataset.base`` has already forced the ``multiprocess`` start
    # method to ``spawn`` at import time, and ``Qwen3_5Template`` now caches
    # its rope-index function at module level rather than on the instance so
    # the template pickles deterministically for HF datasets fingerprinting.
    logger.info('Building HotpotQA dataset (num_proc=%d, max_length=%d)',
                HOTPOTQA_NUM_PROC, HOTPOTQA_MAX_LENGTH)
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

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
    condenser = NativeCondenser(keep_ratio=(0.1, KEEP_RATIO), skip_system=True)

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
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
    logger.info('Starting HotpotQA GRPO training (agentic + context compression)')
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
            chunker, condenser, max_turns=MAX_TURNS,
            min_batch_size=GLOBAL_BATCH_SIZE)

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
                'format': brevity_rewards,
                'f1': accuracy_rewards,
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
                model.save(f'hotpotqa-grpo-tools-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        # Rollout depth is a useful diagnostic: if it collapses to 1 every step
        # the policy has stopped using tools entirely.
        if n_turns_per_rollout:
            log_dict['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-final')


if __name__ == '__main__':
    main()
