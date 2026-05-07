"""HotpotQA GRPO training with LLM-based context compression + tool-augmented multi-turn rollouts.

Variant of ``short_math_grpo_with_tools.py`` where the context compressor
is an LLM (the *base* Qwen3.5-4B without any LoRA) rather than the
rule-based ``PassageIndexCondenser``. The compressor reuses the same
``vLLMSampler`` as the policy rollouts — because the sampler is launched
with ``enable_lora=True`` and the LoRA is mounted as a request-level
adapter, we can force the base weights at compression time by passing
``use_base_model=True`` to ``sampler.sample``. No extra GPU group needed.
"""
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
from twinkle_agentic.condenser import LLMPassageCondenser
from twinkle_agentic.rollout import (
    FrozenContext,
    Rollout,
    batch_freeze_delta_pairs,
    run_agentic_rollouts,
)
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

# Compressor (base-LLM) knobs
COMPRESS_MAX_TOKENS = int(os.environ.get('COMPRESS_MAX_TOKENS', 160))
COMPRESS_TEMPERATURE = float(os.environ.get('COMPRESS_TEMPERATURE', 0.4))
COMPRESS_MIN_CHARS = int(os.environ.get('COMPRESS_MIN_CHARS', 200))
# Target compression ratio (input_len / output_len). 4.0 ≈ keep ~25% of
# the original passage length.
COMPRESS_RATIO = float(os.environ.get('COMPRESS_RATIO', 4.0))

HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# Reward weights
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
ANSWER_COMMIT_PENALTY_WEIGHT = float(os.environ.get('ANSWER_COMMIT_PENALTY_WEIGHT', 1.0))
COT_PENALTY_WEIGHT = float(os.environ.get('COT_PENALTY_WEIGHT', 0.5))
ANSWER_TOO_LONG_CHARS = int(os.environ.get('ANSWER_TOO_LONG_CHARS', 5000))

# ─── Tool-use shaping ───
# TOOL_EXPLORE_BONUS (default 0.1, Phase 1 ON): flat per-rollout bonus
# if the rollout called ANY tool. Meant to kickstart tool use at
# cold-start — turn OFF (=0.0) once ``tool_use_rate`` is healthy
# (>~50%) or if ``avg_tool_calls`` starts climbing without matching F1
# gains (tool spam).
# TOOL_COT_BONUS (default 0.0, Phase 2 OFF): replaces the current 0 in
# ``HotpotQACoTReward`` for the "tool used + enough Step-lines" branch,
# so a well-reasoned tool-calling rollout scores STRICTLY above a
# no-tool rollout. Safer than EXPLORE because it is already gated by
# ``_MIN_STEPS``. Turn ON (=0.3) once EXPLORE is phased out.
# Recommended ramp-up:
#   Phase 0 (baseline):      TOOL_EXPLORE_BONUS=0.0, TOOL_COT_BONUS=0.0
#   Phase 1 (cold-start ← HERE): TOOL_EXPLORE_BONUS=0.1, TOOL_COT_BONUS=0.0
#   Phase 2 (rate > 0.5):    TOOL_EXPLORE_BONUS=0.0, TOOL_COT_BONUS=0.3
TOOL_EXPLORE_BONUS = float(os.environ.get('TOOL_EXPLORE_BONUS', 0.1))
TOOL_COT_BONUS = float(os.environ.get('TOOL_COT_BONUS', 0.0))

_ROLLOUT_TRACE_PATH = os.environ.get('ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt (policy-side — consumed by the LoRA model)
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Put your FINAL answer inside \\boxed{} '
    '(e.g. ``\\boxed{Delhi}``).  Answers not inside \\boxed{} will not be '
    'scored.  Keep the boxed text short: a name, entity, date, or '
    '"yes"/"no". Do not include extra words in the box.\n\n'
    'CONTEXT FORMAT: The provided paragraphs are wrapped as '
    '<block_N>...</block_N>. For long paragraphs, only a Markdown summary '
    'is shown with three sections — **Summary** (one-sentence overview), '
    '**Key** (bulleted salient facts), and **More** (keywords hinting at '
    'additional information available if the block is expanded).\n\n'
    'Use the Summary + Key lines plus the More-list to decide whether a '
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
    'first recall did not contain the answer.\n\n'
    'REASONING FORMAT: After extracting blocks, you MUST reason step by '
    'step before answering. Reference the specific blocks you used:\n'
    'Step 1: From block X, I learn that [fact A].\n'
    'Step 2: From block Y, I learn that [fact B].\n'
    'Step 3: Combining these, the answer is ...\n'
    '\\boxed{answer}\n'
    'Do NOT skip reasoning — always explain which blocks gave you '
    'which facts before writing \\boxed{}.')

# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt for the compressor (base Qwen3.5-4B)
# ═══════════════════════════════════════════════════════════════════════════════
COMPRESS_SYSTEM_PROMPT = (
    'You are a passage compressor for a multi-hop QA pipeline. Given a '
    'single paragraph, produce a compact Markdown summary using EXACTLY '
    'these three sections and nothing else:\n\n'
    '**Summary**: one sentence describing what the passage is about '
    '(subject entity, topic, scope).\n'
    '**Key**: a short bullet list (max 5 items, each under 15 words) of '
    'the most salient facts — entities, relations, numbers, dates.\n'
    '**More**: comma-separated keywords/phrases hinting at additional '
    'information that would be recovered by expanding the passage '
    '(secondary entities, minor dates, extra attributes).\n\n'
    f'Rules: target ~{int(round(100 / COMPRESS_RATIO))}% of the '
    f'original length (compression ratio ~{COMPRESS_RATIO:g}x), and in '
    f'any case stay under {COMPRESS_MAX_TOKENS} tokens. Do NOT answer '
    'any question. Do NOT add any preamble or closing. Output the three '
    'Markdown sections directly, nothing else.\n\n'
    'Example\n'
    'Input paragraph:\n'
    '"Christopher Nolan (born 30 July 1970) is a British-American film '
    'director, producer and screenwriter. His film Inception (2010), a '
    'science-fiction heist movie starring Leonardo DiCaprio, grossed over '
    '$829 million worldwide and received eight Academy Award nominations, '
    'winning four. Nolan also directed The Dark Knight trilogy and '
    'Interstellar (2014)."\n\n'
    'Output:\n'
    '**Summary**: Profile of filmmaker Christopher Nolan and his film '
    'Inception.\n'
    '**Key**:\n'
    '- Christopher Nolan: British-American director, born 30 July 1970.\n'
    '- Inception released 2010, sci-fi heist film.\n'
    '- Stars Leonardo DiCaprio.\n'
    '- Grossed over $829 million worldwide.\n'
    '- 8 Oscar nominations, won 4.\n'
    '**More**: expand to recover — Nolan\'s other roles (producer, '
    'screenwriter), his other directed works (The Dark Knight trilogy, '
    'Interstellar 2014), and the genre label ("heist movie" wording, '
    'Academy Award full name).')

# ═══════════════════════════════════════════════════════════════════════════════
# HotpotQA-specific regex (final-answer extraction)
# ═══════════════════════════════════════════════════════════════════════════════
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')


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
        r'\*\*(?:Summary|Key|More)\*\*\s*:?\s*(?:</?block_\d+>?)?\s*$', re.IGNORECASE)

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


class HotpotQACoTReward(Reward):
    """Reward chain-of-thought reasoning for tool-callers.

    Requires at least ``_MIN_REFS=2`` DISTINCT block references in the
    reasoning preceding ``\\boxed{}``. A block reference is one of:

    * ``From block N`` (strong — names the source)
    * ``block[N`` (bracket shorthand)
    * ``block N <verb>`` where verb ∈ {shows, mentions, states,
      indicates, tells}

    The earlier ``_STEP_RE`` conflated ``Step N:`` headers with block
    refs and DOUBLE-COUNTED them on a single line like
    ``"Step 1: From block 2, ..."`` → 2 regex matches → passed
    ``_MIN_STEPS=2`` threshold with a SINGLE reasoning line. This
    tighter regex demands two DIFFERENT block references across the
    reasoning, so a genuine multi-hop chain is needed before
    ``TOOL_COT_BONUS`` is granted.

    Only penalises rollouts that used tools; direct answerers unaffected.
    """
    _BLOCK_REF_RE = re.compile(
        r'(?:'
        r'from\s+block[\s_]*\d+'
        r'|block[\s_]*\[\d+'
        r'|block[\s_]*\d+[\w\s,.;:\-]{0,20}?(?:shows?|mentions?|states?|indicates?|tells?)'
        r')',
        re.IGNORECASE)
    _MIN_REFS = 2

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            msgs = t.get('messages', [])
            has_tool_call = any(
                m.get('role') == 'assistant' and m.get('tool_calls')
                for m in msgs)
            if not has_tool_call:
                rewards.append(0.0)
                continue

            final = (_last_assistant_text(t) or '').strip()
            boxed_pos = final.rfind('\\boxed{')
            reasoning = final[:boxed_pos] if boxed_pos > 0 else final
            n_refs = len(self._BLOCK_REF_RE.findall(reasoning))

            if n_refs >= self._MIN_REFS:
                # With a non-zero ``TOOL_COT_BONUS`` this branch
                # becomes strictly positive, making "tool + proper
                # multi-hop reasoning" beat "no tool" (which is always
                # 0) in GRPO group-relative advantage.
                rewards.append(TOOL_COT_BONUS)
            elif n_refs == 1:
                rewards.append(-0.3)
            else:
                rewards.append(-0.5)
        return rewards


class HotpotQAToolExploreReward(Reward):
    """Flat bonus for rollouts that called at least one tool.

    Purely exploration-focused: fires on ANY ``tool_calls`` entry in the
    assistant messages regardless of downstream correctness. Intended to
    be turned on ONLY when ``tool_use_rate`` collapses at cold-start,
    and turned back OFF once healthy. See ``TOOL_EXPLORE_BONUS`` above.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards: List[float] = []
        for t in trajectories:
            has_tool_call = any(
                m.get('role') == 'assistant' and m.get('tool_calls')
                for m in t.get('messages', []))
            rewards.append(TOOL_EXPLORE_BONUS if has_tool_call else 0.0)
        return rewards


_F1_REWARD: Optional[HotpotQAF1Reward] = None
_LENGTH_PENALTY: Optional[HotpotQALengthPenalty] = None
_ANSWER_COMMIT_PENALTY: Optional[HotpotQAAnswerCommitPenalty] = None
_COT_REWARD: Optional[HotpotQACoTReward] = None
_TOOL_EXPLORE_REWARD: Optional[HotpotQAToolExploreReward] = None


def compute_rewards(trajectories: List[Dict[str, Any]]) -> Tuple[
        List[float], List[float], List[float], List[float], List[float], List[float]]:
    global _F1_REWARD, _LENGTH_PENALTY, _ANSWER_COMMIT_PENALTY, _COT_REWARD, _TOOL_EXPLORE_REWARD
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _LENGTH_PENALTY = HotpotQALengthPenalty()
        _ANSWER_COMMIT_PENALTY = HotpotQAAnswerCommitPenalty()
        _COT_REWARD = HotpotQACoTReward()
        _TOOL_EXPLORE_REWARD = HotpotQAToolExploreReward()
    f1 = _F1_REWARD(trajectories)
    length_pen = _LENGTH_PENALTY(trajectories)
    answer_commit = _ANSWER_COMMIT_PENALTY(trajectories)
    cot = _COT_REWARD(trajectories)
    tool_explore = _TOOL_EXPLORE_REWARD(trajectories)
    # tool_explore and TOOL_COT_BONUS are already env-gated (default 0),
    # so they stay harmless when disabled. No extra weight knob.
    total = [
        F1_REWARD_WEIGHT * a + LENGTH_PENALTY_WEIGHT * lp
        + ANSWER_COMMIT_PENALTY_WEIGHT * ac + COT_PENALTY_WEIGHT * c + te
        for a, lp, ac, c, te in zip(f1, length_pen, answer_commit, cot, tool_explore)
    ]
    return total, f1, length_pen, answer_commit, cot, tool_explore


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
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Rollout-trace hook (HotpotQA-specific — uses _extract_final_answer)
# ═══════════════════════════════════════════════════════════════════════════════
def _make_dump_rollout_trace(trace_path: str):
    """Build an ``on_turn`` hook that appends per-turn state to a JSONL file.

    Returns ``None`` when ``trace_path`` is empty so the caller can pass
    the result directly into :func:`run_agentic_rollouts`.
    """
    if not trace_path:
        return None

    def _hook(turn, active, displays, responses):
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
                with open(trace_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(records) + '\n')
        except Exception:
            pass

    return _hook


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
    # With the per-turn training feed (see ``Rollout.turn_sequences``)
    # the optimiser does ``sum(rollout.turns)`` mini-batch steps per
    # batch, not just ``rollouts / MINI_BATCH_SIZE``. We therefore
    # multiply by a conservative ``EXPECTED_AVG_TURNS`` so the LR
    # scheduler horizon and the ``optim_step >= total_steps`` early-exit
    # reflect the REAL per-turn step count, not the per-rollout one.
    # If the observed ``avg_turns`` diverges noticeably from the
    # constant below, set ``EXPECTED_AVG_TURNS`` in the environment so
    # training does not stop after a fraction of the planned epochs.
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
    model.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    # Policy sampler (LoRA-enabled, receives weight syncs). The same
    # sampler is reused by the condenser at compression time; the
    # condenser calls ``sampler.sample(..., use_base_model=True)`` so
    # vLLM serves those requests with the base weights (no LoRA),
    # independent of the policy LoRA currently mounted.
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            # Bumped from 8192 -> 32768. The multi-turn rollout keeps
            # APPENDING previously-expanded <block_N> passages (full
            # original text) on every extract_compressed call, so the
            # visible prompt grows monotonically across turns. With
            # MAX_TURNS=6 and MAX_NEW_TOKENS=4096, the previous 8192
            # budget would be exhausted after 1-2 tool expansions and
            # vLLM would start truncating / failing; 32768 gives enough
            # headroom for the full MAX_TURNS rollout plus the final
            # generation even when several passages are recalled.
            'gpu_memory_utilization': 0.8, 'max_model_len': 32768,
            'max_lora_rank': 32, 'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh, remote_group='sampler')
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    condenser = LLMPassageCondenser(
        sampler=sampler,
        sampling_params=SamplingParams(
            max_tokens=COMPRESS_MAX_TOKENS, num_samples=1,
            temperature=COMPRESS_TEMPERATURE, top_p=0.9),
        system_prompt=COMPRESS_SYSTEM_PROMPT,
        min_chars=COMPRESS_MIN_CHARS,
        # Skip the policy's own assistant turns in addition to the
        # default system/tool skip. Otherwise any turn whose cleaned
        # assistant content exceeds ``min_chars`` gets summarised into
        # Summary/Key/More on the NEXT turn, and the policy would see
        # its own chain-of-thought through a compression lens —
        # directly fighting the ``HotpotQACoTReward`` signal.
        skip_roles=('system', 'tool', 'assistant'),
    )

    # Per-rollout tool factory: the ExtractCompressed tool needs the
    # rollout's CURRENT full+compressed chunks (which evolve every turn),
    # so the factory closes over ``r.frozen`` instead of being built
    # once up front.
    def _build_tool_manager(r: Rollout) -> ToolManager:
        return ToolManager([
            ExtractCompressed(
                r.frozen.render_full(),
                displayed_to_full=r.frozen.displayed_to_full(),
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

        # Compress the initial prompt (system + user + passages) ONCE per
        # unique prompt in the batch, then share the result across its
        # ``NUM_GENERATIONS`` rollouts via ``FrozenContext.clone()``.
        # Without this, each rollout would independently re-run the
        # base-LLM compressor on the same passages, multiplying
        # compression cost by NUM_GENERATIONS and introducing sampling
        # noise across rollouts of the same prompt (different
        # **Summary**/**Key**/**More** for the same input paragraph).
        # Furthermore, unique-prompt compressions are batched together
        # into a single condenser call, so the whole batch's initial
        # compression costs exactly one ``sampler.sample`` trip.
        shared_frozens: Dict[int, FrozenContext] = {}
        initial_pairs: List[Tuple[FrozenContext, Dict[str, Any]]] = []
        for prompt in batch:
            key = id(prompt)
            if key in shared_frozens:
                continue
            fc = FrozenContext()
            shared_frozens[key] = fc
            initial_pairs.append((fc, prompt))
        batch_freeze_delta_pairs(initial_pairs, chunker, condenser)
        initial_frozens: List[Optional[FrozenContext]] = [
            shared_frozens[id(p)] for p in expand_prompts]

        rollouts = run_agentic_rollouts(
            expand_prompts, sampler, sampling_params, chunker, condenser,
            _build_tool_manager,
            max_turns=MAX_TURNS, min_batch_size=GLOBAL_BATCH_SIZE,
            initial_frozens=initial_frozens, on_turn=on_turn_hook)

        # ---- Per-rollout aggregates (reward + metrics) ------------------
        # Reward is computed on the whole rollout trajectory (final F1 +
        # length / answer-commit / cot shaping). Completion length logged
        # per rollout is the SUM of every turn's generated token count so
        # we see the true generation budget usage, not just the last turn.
        # NOTE: do NOT silently filter out rollouts with empty
        # ``turn_sequences`` — the rollout list is ordered as consecutive
        # groups of NUM_GENERATIONS per prompt, and ``GRPOAdvantage`` with
        # ``scale='group'`` relies on that grouping exactly. Dropping one
        # rollout would shift every subsequent group by one and corrupt
        # the group-relative advantage computation. Crash loudly instead.
        empty_rollouts = [i for i, r in enumerate(rollouts) if not r.turn_sequences]
        assert not empty_rollouts, (
            f'rollouts {empty_rollouts} have empty turn_sequences; this would '
            'break GRPO group alignment. Likely a sampler or min_batch_size bug.')
        all_trajectories = [r.trajectory for r in rollouts]
        n_turns_per_rollout = [r.turns for r in rollouts]
        per_rollout_completion_length = [
            sum(len(s.tokens) for s in r.turn_sequences) for r in rollouts]

        total_rewards, f1_rewards, length_pen_rewards, answer_commit_rewards, cot_rewards, tool_explore_rewards = \
            compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=per_rollout_completion_length,
            rewards={'total': total_rewards, 'f1': f1_rewards,
                     'length_pen': length_pen_rewards, 'answer_commit': answer_commit_rewards,
                     'cot': cot_rewards, 'tool_explore': tool_explore_rewards})

        # GRPO advantages are computed at rollout level (group-scaled over
        # NUM_GENERATIONS same-prompt rollouts), then REPLICATED to every
        # turn of that rollout. This way the same advantage signal is
        # back-propagated through (i) the tool-call decision turns and
        # (ii) the final answer turn, so the policy gets direct gradient
        # on WHEN to expand a block, not only on what to say after it
        # was already expanded.
        rollout_advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # ---- Per-turn training feed ------------------------------------
        all_input_data: List[Any] = []
        all_old_logps: List[List[float]] = []
        advantages: List[float] = []
        for r, adv in zip(rollouts, rollout_advantages):
            for seq in r.turn_sequences:
                all_input_data.append(seq.new_input_feature)
                all_old_logps.append([lp[0][1] for lp in (seq.logprobs or [])])
                advantages.append(adv)

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
                model.save(f'hotpotqa-grpo-tools-llmcondense-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        if n_turns_per_rollout:
            log_dict['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)
        # Track the maximum prompt length across ALL turn-level training
        # samples in this batch. With max_model_len=32768, the worst-case
        # multi-turn rollout (6 turns × 4K MAX_NEW_TOKENS + 6 expanded
        # blocks of original text) can approach 33K tokens and silently
        # overflow vLLM. If this metric climbs above ~30K, lower
        # ``max_blocks_per_call`` on ExtractCompressed or bump
        # ``max_model_len``.
        _max_prompt_tok = 0
        for r in rollouts:
            for seq in r.turn_sequences:
                feat = getattr(seq, 'new_input_feature', None) or {}
                ids = feat.get('input_ids') if isinstance(feat, dict) else None
                if ids:
                    prompt_len = max(0, len(ids) - len(seq.tokens or []))
                    if prompt_len > _max_prompt_tok:
                        _max_prompt_tok = prompt_len
        log_dict['max_prompt_tokens'] = _max_prompt_tok
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
    model.save('hotpotqa-grpo-tools-llmcondense-final')


if __name__ == '__main__':
    main()
