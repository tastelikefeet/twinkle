"""HotpotQA GRPO training with context compression + tool-augmented multi-turn rollouts."""
import json
import os
import re
import string
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

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
from twinkle_agentic.condenser import PassageIndexCondenser
from twinkle_agentic.data_format.chunk import Chunk, Chunks
from twinkle_agentic.tools.extract import ExtractCompressed
from twinkle_agentic.tools.tool_manager import ToolManager

import swanlab

from nltk.stem import PorterStemmer as _PorterStemmer
logger = get_logger()
_STEMMER = _PorterStemmer()

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '1')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 5))
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

HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# Reward weights
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
ANSWER_COMMIT_PENALTY_WEIGHT = float(os.environ.get('ANSWER_COMMIT_PENALTY_WEIGHT', 1.0))
ANSWER_TOO_LONG_CHARS = int(os.environ.get('ANSWER_TOO_LONG_CHARS', 5000))

_ROLLOUT_TRACE_PATH = os.environ.get('ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Put your FINAL answer inside \\boxed{} '
    '(e.g. ``\\boxed{Delhi}``).  Answers not inside \\boxed{} will not be '
    'scored.  Keep the boxed text short: a name, entity, date, or '
    '"yes"/"no". Do not include extra words in the box.\n\n'
    'CONTEXT FORMAT: The provided paragraphs are wrapped as '
    '<block_N>...</block_N>. For long paragraphs, only the first sentence '
    'is shown, followed by "(Facts: entity1 [rel] entity2; ... | Related: '
    'keyword1, keyword2, ...)" listing key relationships and additional '
    'terms present in the hidden remainder.\n\n'
    'Use the first sentence plus the Related-list to decide whether a '
    'block probably contains the fact you need. When it does, call the '
    '``extract_compressed`` tool with the relevant block numbers to recall '
    'the full original text. When the visible text is already sufficient, '
    'answer immediately without calling the tool.\n\n'
    'TOOL CALL FORMAT:\n'
    '<tool_call>\n'
    '<function=extract_compressed>\n'
    '<parameter=blocks>\n'
    '[1, 3]\n'
    '</parameter>\n'
    '</function>\n'
    '</tool_call>\n\n'
    'You may call ``extract_compressed`` again in a later turn if the '
    'first recall did not contain the answer.')

# ═══════════════════════════════════════════════════════════════════════════════
# Tool-call parsing & text sanitization
# ═══════════════════════════════════════════════════════════════════════════════
# Qwen3.5 native XML format: <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>
_TOOL_CALL_BLOCK_RE = re.compile(r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
_FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
_PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
# Legacy JSON format fallback
_TOOL_CALL_STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls (Qwen3.5 XML format + JSON fallback)."""
    calls: List[Dict[str, Any]] = []
    for block_m in _TOOL_CALL_BLOCK_RE.finditer(text):
        block = block_m.group(1)
        func_m = _FUNCTION_RE.search(block)
        if func_m:
            args = {}
            for pm in _PARAMETER_RE.finditer(func_m.group(2)):
                try:
                    args[pm.group(1).strip()] = json.loads(pm.group(2).strip())
                except (json.JSONDecodeError, ValueError):
                    args[pm.group(1).strip()] = pm.group(2).strip()
            calls.append({'tool_name': func_m.group(1).strip(), 'arguments': args})
        else:
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
            calls.append({'tool_name': name, 'arguments': args if isinstance(args, dict) else {}})
    return calls


def _clean_assistant_output(text: str) -> str:
    """Remove tool_call tags and echoed block tags from assistant text."""
    text = _TOOL_CALL_STRIP_RE.sub('', text or '')
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text)
    return text.rstrip()


# ═══════════════════════════════════════════════════════════════════════════════
# Passage Index Generation (delegates to PassageIndexCondenser)
# ═══════════════════════════════════════════════════════════════════════════════
_passage_index_condenser = PassageIndexCondenser(max_keywords=12)


def _generate_structured_index(chunks: Chunks) -> Chunks:
    return _passage_index_condenser.condense(chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# F1 Reward & Helpers
# ═══════════════════════════════════════════════════════════════════════════════
_FILLER_TOKENS: frozenset = frozenset([
    'long', 'tall', 'high', 'wide', 'deep', 'heavy', 'old', 'large',
    'small', 'big', 'short', 'away', 'ago', 'approximately', 'about',
    'around', 'over', 'under', 'below', 'above', 'total', 'roughly',
    'nearly', 'almost', 'exactly',
])


def _stem(tok: str) -> str:
    return _STEMMER.stem(tok) if len(tok) >= 4 and tok.isalpha() else tok


def _normalize_answer(s: str) -> str:
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(_stem(t) for t in s.split())


def _f1_score(prediction: str, gold: str) -> Tuple[float, float]:
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
    f1 = 2 * p * r / (p + r)

    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    if gold_set < pred_set:
        extras = pred_set - gold_set
        if all(t.isdigit() or t in _FILLER_TOKENS for t in extras):
            return 1.0, em
    if pred_set < gold_set:
        missing = gold_set - pred_set
        if all(t in _FILLER_TOKENS for t in missing):
            return 1.0, em
    return f1, em


def _extract_final_answer(completion: str) -> str:
    matches = _BOXED_RE.findall(completion or '')
    return matches[-1].strip() if matches else ''


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


# ═══════════════════════════════════════════════════════════════════════════════
# Reward Classes
# ═══════════════════════════════════════════════════════════════════════════════
class HotpotQAF1Reward(Reward):
    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            gold = ''
            for key, val in traj.get('user_data', []) or []:
                if key == 'ground_truth':
                    gold = val or ''
                    break
            pred = _extract_final_answer(_last_assistant_text(traj))
            f1, _ = _f1_score(pred, gold)
            rewards.append(f1)
        return rewards


class HotpotQALengthPenalty(Reward):
    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        budget = max(1, MAX_NEW_TOKENS * 4 - ANSWER_TOO_LONG_CHARS)
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            overflow = max(0, len(text) - ANSWER_TOO_LONG_CHARS)
            rewards.append(-min(1.0, overflow / budget))
        return rewards


class HotpotQAAnswerCommitPenalty(Reward):
    _COMPRESSION_TAG_RE = re.compile(
        r'\(\s*(?:Facts|Related)\s*:[^)]*\)\s*</?block_\d+>?\s*$', re.IGNORECASE)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            final = (_last_assistant_text(t) or '').strip()
            if len(final) < 5:
                rewards.append(-1.0)
            elif self._COMPRESSION_TAG_RE.search(final):
                rewards.append(-0.5)
            else:
                rewards.append(0.0)
        return rewards


_F1_REWARD: Optional[HotpotQAF1Reward] = None
_LENGTH_PENALTY: Optional[HotpotQALengthPenalty] = None
_ANSWER_COMMIT_PENALTY: Optional[HotpotQAAnswerCommitPenalty] = None


def compute_rewards(trajectories: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    global _F1_REWARD, _LENGTH_PENALTY, _ANSWER_COMMIT_PENALTY
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _LENGTH_PENALTY = HotpotQALengthPenalty()
        _ANSWER_COMMIT_PENALTY = HotpotQAAnswerCommitPenalty()
    f1 = _F1_REWARD(trajectories)
    length_pen = _LENGTH_PENALTY(trajectories)
    answer_commit = _ANSWER_COMMIT_PENALTY(trajectories)
    total = [
        F1_REWARD_WEIGHT * a + LENGTH_PENALTY_WEIGHT * lp + ANSWER_COMMIT_PENALTY_WEIGHT * ac
        for a, lp, ac in zip(f1, length_pen, answer_commit)
    ]
    return total, f1, length_pen, answer_commit


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════
class HotpotQAProcessor(Preprocessor):
    def __init__(self, system: str = SYSTEM_PROMPT):
        self.system = system

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
        if (row.get('level') or '').strip().lower() != 'hard':
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

    # Optional id whitelist
    _wrong_ids_path = os.environ.get('WRONG_IDS_FILE', '').strip()
    if _wrong_ids_path:
        with open(_wrong_ids_path, 'r', encoding='utf-8') as fh:
            _ids = frozenset(ln.strip() for ln in fh if ln.strip())
        if _ids:
            _key = next(iter(dataset.datasets.keys()))
            _before = len(dataset.datasets[_key])
            dataset.datasets[_key] = dataset.datasets[_key].filter(
                lambda row: row.get('id') in _ids)
            dataset.dataset = dataset.datasets[_key]
            print(f'[WRONG_IDS_FILE] {_wrong_ids_path}: {_before} -> {len(dataset.dataset)} rows')

    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    _HOTPOTQA_COLS = ['id', 'question', 'answer', 'type', 'level',
                      'supporting_facts', 'context']
    dataset.map(HotpotQAProcessor(system=SYSTEM_PROMPT), remove_columns=_HOTPOTQA_COLS)
    dataset.encode(add_generation_prompt=True, load_from_cache_file=True, num_proc=HOTPOTQA_NUM_PROC)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic Rollout
# ═══════════════════════════════════════════════════════════════════════════════
_MEDIA_KEYS = ('images', 'videos', 'audios')


class _FrozenContext:
    """Incremental chunk+index cache. Only processes NEW messages each turn."""
    __slots__ = ('frozen_msg_count', 'full_chunks', 'compressed_chunks', 'media_frozen')

    def __init__(self) -> None:
        self.frozen_msg_count: int = 0
        self.full_chunks: List[Chunk] = []
        self.compressed_chunks: List[Chunk] = []
        self.media_frozen: bool = False

    def freeze_delta(self, trajectory: Dict[str, Any], chunker: NativeChunker) -> None:
        total_msgs = trajectory['messages']
        new_msgs = total_msgs[self.frozen_msg_count:]
        needs_media = not self.media_frozen and any(trajectory.get(k) for k in _MEDIA_KEYS)
        if not (new_msgs or needs_media):
            return

        delta: Dict[str, Any] = {
            'messages': list(new_msgs),
            'user_data': trajectory.get('user_data', []),
        }
        if needs_media:
            for k in _MEDIA_KEYS:
                if trajectory.get(k):
                    delta[k] = trajectory[k]
            self.media_frozen = True

        new_full = chunker.chunk(delta)
        new_compressed = _generate_structured_index(new_full)
        self.full_chunks.extend(new_full.chunks)
        self.compressed_chunks.extend(new_compressed.chunks)
        self.frozen_msg_count = len(total_msgs)

    def render_display(self) -> Dict[str, Any]:
        return Chunks(chunks=list(self.compressed_chunks)).to_trajectory()

    def render_full(self) -> Chunks:
        return Chunks(chunks=list(self.full_chunks))


class _Rollout:
    __slots__ = ('trajectory', 'final_sequence', 'turns', 'done', 'frozen')

    def __init__(self, prompt_trajectory: Dict[str, Any]) -> None:
        self.trajectory: Dict[str, Any] = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        for _k in _MEDIA_KEYS:
            if prompt_trajectory.get(_k):
                self.trajectory[_k] = list(prompt_trajectory[_k])
        self.final_sequence = None
        self.turns = 0
        self.done = False
        self.frozen = _FrozenContext()


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    chunker: NativeChunker,
    max_turns: int,
    min_batch_size: int = 1,
) -> List[_Rollout]:
    rollouts = [_Rollout(p) for p in prompts]

    for turn in range(max_turns):
        active = [r for r in rollouts if not r.done]
        if not active:
            break

        displays: List[Dict[str, Any]] = []
        tool_mgrs: List[ToolManager] = []
        for r in active:
            r.frozen.freeze_delta(r.trajectory, chunker)
            displays.append(r.frozen.render_display())
            tool_mgrs.append(ToolManager([ExtractCompressed(r.frozen.render_full())]))

        # Pad to min_batch_size for DP
        n_active = len(displays)
        if n_active < min_batch_size:
            displays = displays + [displays[0]] * (min_batch_size - n_active)

        responses = sampler.sample(displays, sampling_params)
        responses = responses[:n_active]

        for r, resp, tool_mgr in zip(active, responses, tool_mgrs):
            seq = resp.sequences[0]
            r.final_sequence = seq
            r.turns += 1
            decoded = seq.decoded or ''
            tool_calls = parse_tool_calls(decoded)

            if not tool_calls:
                # Terminal turn
                r.trajectory['messages'].append(
                    {'role': 'assistant', 'content': _clean_assistant_output(decoded)})
                r.done = True
            else:
                # Tool turn
                r.trajectory['messages'].append({
                    'role': 'assistant',
                    'content': _clean_assistant_output(decoded),
                    'tool_calls': [
                        {'tool_name': tc['tool_name'],
                         'arguments': (tc['arguments'] if isinstance(tc['arguments'], str)
                                       else json.dumps(tc['arguments'], ensure_ascii=False))}
                        for tc in tool_calls
                    ],
                })
                for i, tc in enumerate(tool_calls):
                    r.trajectory['messages'].append({
                        'role': 'tool',
                        'content': tool_mgr.dispatch(tc),
                        'tool_call_id': f'call_t{turn}_i{i}',
                    })

        # Dump trace
        _dump_rollout_trace(turn, active, displays, responses)

    # Mark remaining as done
    for r in rollouts:
        r.done = True
    return rollouts


def _dump_rollout_trace(turn, active, displays, responses):
    """Append all active rollouts' state to trace JSONL (best-effort)."""
    if not _ROLLOUT_TRACE_PATH or not active:
        return
    try:
        records: List[str] = []
        for idx, r in enumerate(active):
            try:
                resp = responses[idx] if idx < len(responses) else None
                tcc = sum(len(m.get('tool_calls') or [])
                          for m in r.trajectory.get('messages', [])
                          if m.get('role') == 'assistant')
                last_decoded = ''
                if resp and getattr(resp, 'sequences', None):
                    last_decoded = resp.sequences[0].decoded or ''
                final_answer = _extract_final_answer(
                    _last_assistant_text(r.trajectory)) if r.done else ''

                record = {
                    'ts': time.time(), 'turn': turn,
                    'group_size': len(active), 'picked_idx': idx,
                    'rollout_id': id(r), 'tool_call_count': tcc,
                    'done': bool(r.done),
                    'compressed': displays[idx] if idx < len(displays) else None,
                    'last_decoded': last_decoded, 'final_answer': final_answer,
                }
                records.append(json.dumps(record, ensure_ascii=False, default=str))
            except Exception:
                pass
        if records:
            with open(_ROLLOUT_TRACE_PATH, 'a', encoding='utf-8') as f:
                f.write('\n'.join(records) + '\n')
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    swanlab.init(project='twinkle')

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # Truncate trace file for clean run
    if _ROLLOUT_TRACE_PATH:
        try:
            open(_ROLLOUT_TRACE_PATH, 'w').close()
        except OSError:
            pass

    # Build dataset
    logger.info('Building HotpotQA dataset')
    _prebuilt_dataset = create_hotpotqa_dataset()
    logger.info('Dataset ready: %d rows', len(_prebuilt_dataset))

    # Compute training horizon (accounts for multiple optim steps per batch)
    _global_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    batches_per_epoch = max(1, len(_prebuilt_dataset) // _global_batch_size)
    optim_steps_per_batch = max(1, (BATCH_SIZE * NUM_GENERATIONS + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
    steps_per_epoch = batches_per_epoch * optim_steps_per_batch
    derived_total_steps = NUM_EPOCHS * steps_per_epoch
    total_steps = min(MAX_STEPS, derived_total_steps) if MAX_STEPS > 0 else derived_total_steps
    logger.info('Training horizon: %d steps (%d epochs × %d batches × %d steps/batch)',
                total_steps, NUM_EPOCHS, batches_per_epoch, optim_steps_per_batch)

    # Model setup
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
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8, 'max_model_len': 8192,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
    chunker = NativeChunker(
        model_id=MODEL_ID, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        passage_boundary_re=r'^\[\d+\]\s+')

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE, min_batch_size=GLOBAL_BATCH_SIZE)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95, stop=['</tool_call>'])

    optim_step = 0
    logger.info('Starting HotpotQA GRPO training')

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

        # Multi-turn rollout
        rollouts = run_agentic_rollouts(
            expand_prompts, sampler, sampling_params, chunker,
            max_turns=MAX_TURNS, min_batch_size=GLOBAL_BATCH_SIZE)

        all_input_data, all_old_logps, all_completion_lengths = [], [], []
        all_trajectories, n_turns_per_rollout = [], []
        for r in rollouts:
            seq = r.final_sequence
            if seq is None:
                continue
            all_input_data.append(seq.new_input_feature)
            all_old_logps.append([lp[0][1] for lp in (seq.logprobs or [])])
            all_completion_lengths.append(len(seq.tokens))
            all_trajectories.append(r.trajectory)
            n_turns_per_rollout.append(r.turns)

        total_rewards, f1_rewards, length_pen_rewards, answer_commit_rewards = \
            compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={'total': total_rewards, 'f1': f1_rewards,
                     'length_pen': length_pen_rewards, 'answer_commit': answer_commit_rewards})

        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Mini-batch training loop
        total_completions = len(all_input_data)
        for mb_start in range(0, total_completions, MINI_BATCH_SIZE):
            mb_end = min(mb_start + MINI_BATCH_SIZE, total_completions)
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
                model.save(f'hotpotqa-grpo-tools-checkpoint-{optim_step}')

        # Logging
        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        if n_turns_per_rollout:
            log_dict['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)
        if all_trajectories:
            tool_counts = [
                sum(len(m.get('tool_calls') or [])
                    for m in t.get('messages', []) if m.get('role') == 'assistant')
                for t in all_trajectories]
            log_dict['avg_tool_calls'] = sum(tool_counts) / len(tool_counts)
            log_dict['tool_use_rate'] = sum(1 for c in tool_counts if c > 0) / len(tool_counts)
            n_no_boxed = sum(
                0 if _BOXED_RE.search(_last_assistant_text(t) or '') else 1
                for t in all_trajectories)
            log_dict['no_boxed_rate'] = n_no_boxed / len(all_trajectories)
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{total_steps}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-final')


if __name__ == '__main__':
    main()
