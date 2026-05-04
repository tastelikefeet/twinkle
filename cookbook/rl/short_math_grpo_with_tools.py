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
import random
import re
import string
import time
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
from twinkle_agentic.data_format.chunk import Chunk, Chunks
from twinkle_agentic.reward.extract_reward import ExtractReward
from twinkle_agentic.tools.extract import ExtractCompressed
from twinkle_agentic.tools.tool_manager import ToolManager

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

# ---- Agentic rollout knobs ----
MAX_TURNS = int(os.environ.get('MAX_TURNS', 5))            # hard cap on tool-call turns
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1024))        # chars per chunk (NativeChunker)
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 0))    # sliding-window overlap
KEEP_RATIO = float(os.environ.get('KEEP_RATIO', 0.3))      # NativeCondenser target ratio

# ---- Reward-balancing knobs (tuned from 2nd crash post-mortem: F1 stagnant) ----
# Round 1 post-mortem: policy collapsed into repetition loops (fixed via
# repetition_penalty + length_pen + format gate).
# Round 2 post-mortem (current): policy stopped collapsing but F1 stagnated
# at ~0.5 below the base-model baseline of 0.62, with tool-use dropping to
# 0% across 437/445 rollouts.  Root cause: ``format`` saturated at 1.0 and
# ``extract`` saturated at 0.99 across ALL rollouts, so GRPO's within-group
# normalisation zeroed their advantages; meanwhile ``ExtractReward`` is a
# *penalty* on tool calls, giving the policy no incentive to use tools at
# all.  Optimal policy: emit a random ``\\boxed{X}`` and collect constants.
#
# Fixes applied here:
#   1. Replace ``ExtractReward`` (penalty) with ``HotpotQAToolUseReward``
#      (+0.2 bonus for using at least one extract_compressed call).  Creates
#      within-group variance on ``tool_use`` → GRPO can push policy toward
#      using tools instead of away from them.
#   2. Drop ``FORMAT_REWARD_WEIGHT`` to 0.0 -- signal is 100% saturated,
#      pure noise contribution to GRPO advantages.  Model has already
#      learned to emit ``\\boxed{}``; no gradient is needed there.
#   3. Bump ``F1_REWARD_WEIGHT`` 1.0 → 1.5 so the real learning signal
#      dominates the weighted sum.
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
FORMAT_REWARD_WEIGHT = float(os.environ.get('FORMAT_REWARD_WEIGHT', 0.0))
# Round-3 bump: 0.2 was too small -- with ~3% tool-call base rate, 76%+
# of GRPO groups have all-zero tool_use vectors, zeroing advantages.
# Raising to 1.0 gives the (rare) tool-using rollouts a much larger
# positive advantage relative to their non-using peers inside each group
# that *does* have mixed tool-use, so the policy can actually learn from
# the (still infrequent) exploration.  Pair with temperature bump on
# SamplingParams for wider exploration.
TOOL_USE_REWARD_WEIGHT = float(os.environ.get('TOOL_USE_REWARD_WEIGHT', 1.0))
# Round-4: bonus given ONLY when a rollout BOTH used a tool AND answered
# correctly (F1 >= threshold).  Needed because Round-3 showed tool-path
# F1 was ~50% and length_pen could eat the raw tool_use bonus, making
# "guess a short answer" dominate "explore with tools" again in GRPO.
# This shifts the equilibrium to: {correct + tool} > {correct, no tool} >
# {wrong + tool} ~ {wrong, no tool}.
TOOL_SUCCESS_REWARD_WEIGHT = float(os.environ.get('TOOL_SUCCESS_REWARD_WEIGHT', 0.5))
TOOL_SUCCESS_F1_THRESHOLD = float(os.environ.get('TOOL_SUCCESS_F1_THRESHOLD', 0.5))
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
# Kept for backward compatibility -- no longer consumed by compute_rewards
# (replaced by TOOL_USE_REWARD_WEIGHT) but still logged as a diagnostic.
EXTRACT_REWARD_WEIGHT = float(os.environ.get('EXTRACT_REWARD_WEIGHT', 0.0))
# Format reward is gated: an assistant message longer than this does NOT
# get the +1 for emitting \boxed{} -- stops the policy from padding garbage
# around a single \boxed{...} to satisfy both signals.
# Round-4: 1500 was too strict -- a legitimate "retrieve + reason + answer"
# rollout takes 2000-4000 chars, getting unfairly penalised and eating the
# tool-use bonus.  Raised to 5000 so only real repetition loops (>10k chars)
# still hit the penalty.
FORMAT_MAX_CHARS = int(os.environ.get('FORMAT_MAX_CHARS', 5000))
# F1-gate threshold for length_pen (Round-5 fix).  When F1 on a rollout is
# below this threshold, its length_pen is forced to 0 so long-and-wrong
# rollouts don't get a free "short-is-better" gradient via GRPO within-group
# variance.  Set to 0.0 to disable the gate (legacy behaviour: length_pen
# always active).  See post-mortem in Round-5 diagnosis for why ungated
# length_pen inflates thinking length and triggers truncation collapse.
LENGTH_PEN_F1_GATE = float(os.environ.get('LENGTH_PEN_F1_GATE', 0.3))
# GRPO homogeneous-group filter threshold (Round-5 fix).  Samples whose
# per-sample max |advantage| is below this are masked out of the loss;
# prevents KL drift and zero-variance-noise advantage amplification on
# groups with no reward signal.  Passed to ``GRPOLoss(..., homogeneous_threshold=..)``.
GRPO_HOMOGENEOUS_THRESHOLD = float(
    os.environ.get('GRPO_HOMOGENEOUS_THRESHOLD', 1e-4))
# Curriculum: disable tool-use bonus for the first fraction of MAX_STEPS so
# the policy first learns to answer (F1-driven) before tool-use pressure.
# Set to 0.0 to apply the tool-use bonus from step 0 (recommended given
# the 2nd-round symptom was tool-use already at 0%).
EXTRACT_WARMUP_FRAC = float(os.environ.get('EXTRACT_WARMUP_FRAC', 0.0))

# ---- HotpotQA dataset encode knobs ----
HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# ========== System Prompt ==========
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Give a short factual answer (a name, '
    'entity, date, or "yes"/"no") inside \\boxed{}. Do not include extra '
    'words in the box.\n\n'
    'CONTEXT COMPRESSION: The provided paragraphs are shown with '
    '<block_N>...</block_N> markers around each chunk. Some chunks have been '
    'shortened to save context. When a block looks relevant but its content '
    'appears truncated or ambiguous, call ``extract_compressed`` to recover '
    'the full original passage before answering -- accuracy on multi-hop '
    'questions typically improves when you retrieve the exact evidence.\n\n'
    'TOOL CALL FORMAT: Emit tool calls inside a single fenced block like '
    'this, then stop generating and wait for the tool result:\n'
    '<tool_call>\n'
    '{"name": "extract_compressed", "arguments": {"blocks": [1, 3]}}\n'
    '</tool_call>\n\n'
    'If you are confident the compressed content is already sufficient, you '
    'may answer directly; otherwise prefer retrieving the evidence first.')

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


# ── Assistant output sanitisation ──────────────────────────────────────────
# Three transforms run unconditionally before any assistant-generated text is
# committed to the trajectory.  Applying them together is safe and idempotent
# on clean text, and collapsing the three historically-separate helpers into
# one entry point removes the risk of forgetting one at a new call site.
#
#   1. Drop raw ``<tool_call>...</tool_call>`` spans.  The structured
#      ``tool_calls`` field is the canonical source; leaving both present
#      would duplicate (or malform, when the stop tag was stripped) the
#      markup on the next turn's re-render.
#   2. Unwrap ``<block_N>...</block_N>`` markers that the model sometimes
#      echoes back from the user prompt into its own reasoning.  The inner
#      text is preserved; only the wrappers are dropped.  Keeping them would
#      (a) pollute the training signal by teaching the model to emit these
#      tags, and (b) leave stale references in later turns where the re-
#      chunker would have assigned ``block_N`` to something different.
#   3. Drop ``[[#N]]`` pseudo-citations, which no tool or prompt in this
#      pipeline defines -- pure hallucinations that should not survive into
#      the training data.
_TOOL_CALL_STRIP_RE = re.compile(r'<tool_call>.*?(?:</tool_call>|\Z)', re.DOTALL)
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_FAKE_CITE_RE = re.compile(r'\[\[#\d+\]\]')


def _clean_assistant_output(text: str) -> str:
    text = _TOOL_CALL_STRIP_RE.sub('', text or '')
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text)
    text = _FAKE_CITE_RE.sub('', text)
    return text.rstrip()


# ``HotpotQAProcessor`` renders the user turn as
# ``'Question: {q}\n\nContext:\n\n{passages}'``.  Query-aware condensation
# only needs the question half: feeding the full turn (which includes every
# passage) would give near-every token a query-match, neutralising the bonus.
# We therefore slice off everything from ``\n\nContext:`` onward, and drop
# the ``Question:`` / ``Q:`` / ``问题：`` / ... prefix so the framing word
# itself does not become a boost-inducing token on unrelated sentences.
_QUESTION_PREFIX_RE = re.compile(
    r'^\s*(?:question|query|q|问题|提问|问)\s*[:：]\s*',
    re.IGNORECASE)
# Common delimiters separating a short question from a longer attached
# context / document block.  Checked in order; first hit wins.
_CONTEXT_DELIMITERS: Tuple[str, ...] = (
    '\n\nContext:', '\n\ncontext:', '\n\nCONTEXT:',
    '\n\nPassages:', '\n\nDocuments:', '\n\nReference:',
    '\n\n上下文：', '\n\n文档：', '\n\n材料：',
)
# Cap the hint length so long-form datasets (entire-document-as-question)
# cannot explode token overlap and neutralise query-aware scoring.
_QUERY_HINT_MAX_CHARS = 512


def _extract_query_hint(messages: List[Dict[str, Any]]) -> str:
    """Return the question text from the first user message, if any.

    Designed to degrade gracefully across datasets rather than lock to the
    HotpotQA ``Question: ... \n\nContext: ...`` template:

      1. Flatten the user message content.  It may arrive either as a
         plain ``str`` or as a multimodal list of ``{'type': 'text',
         'text': ...}`` / ``{'type': 'image', ...}`` dicts; we concatenate
         every ``text`` fragment.
      2. Split off any trailing context / document block.  Multiple
         markers are tried (``Context:`` / ``Passages:`` / ``上下文：`` /
         ...) so other preprocessors that follow the same pattern with a
         different label still work out of the box.
      3. If no explicit delimiter is found, fall back to the first
         blank-line block -- covers prompts where the question is the
         opening paragraph and supporting text follows.
      4. Strip a leading ``Question:`` / ``Q:`` / ``问题：`` prefix so the
         framing word does not bias query-token matching.
      5. Cap the result at ``_QUERY_HINT_MAX_CHARS``.  Without this cap,
         datasets whose user message *is* a long document would yield a
         "hint" that matches almost every passage, destroying the point
         of query-awareness.

    Returns ``''`` when no user message carries text content, which
    disables query-aware scoring downstream.
    """
    for msg in messages:
        if msg.get('role') != 'user':
            continue
        raw = msg.get('content')
        if isinstance(raw, str):
            content = raw
        elif isinstance(raw, list):
            parts: List[str] = []
            for item in raw:
                if isinstance(item, dict) and item.get('type') == 'text':
                    t = item.get('text')
                    if isinstance(t, str) and t:
                        parts.append(t)
            content = '\n'.join(parts)
        else:
            continue
        if not content or not content.strip():
            continue
        head = content
        for delim in _CONTEXT_DELIMITERS:
            if delim in head:
                head = head.split(delim, 1)[0]
                break
        else:
            # No explicit context delimiter -- use first blank-line block.
            head = head.split('\n\n', 1)[0]
        head = _QUESTION_PREFIX_RE.sub('', head).strip()
        if len(head) > _QUERY_HINT_MAX_CHARS:
            head = head[:_QUERY_HINT_MAX_CHARS].rstrip()
        return head
    return ''


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
_LENGTH_PENALTY = None
_TOOL_USE_REWARD = None
_TOOL_SUCCESS_REWARD = None


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

    Gated by :data:`FORMAT_MAX_CHARS`: an assistant message whose cleaned
    text exceeds the cap does NOT get the bonus even if it contains a
    ``\\boxed{...}`` somewhere inside.  Without this gate the policy can
    (and did, in the previous collapse run) learn to ramble for 20k chars
    and slip a single ``\\boxed{garbage}`` at the end, collecting the
    format bonus while emitting pure repetition loops.
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
    ``MAX_NEW_TOKENS * 4`` chars (empirical 4 chars/token ceiling).  This
    directly counteracts the "fill the budget with repetition" exploit
    observed in the crash post-mortem.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        # Budget in characters -- picked so the penalty saturates exactly
        # when the model fills the full MAX_NEW_TOKENS window with ASCII.
        budget = max(1, MAX_NEW_TOKENS * 4 - FORMAT_MAX_CHARS)
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            overflow = max(0, len(text) - FORMAT_MAX_CHARS)
            rewards.append(-min(1.0, overflow / budget))
        return rewards


class HotpotQAToolUseReward(Reward):
    """Positive reward for using ``extract_compressed`` at least once.

    Replaces :class:`ExtractReward` (which was a *penalty* that collapsed to
    a 0.99 constant when the policy learned to stop calling tools).  This
    variant is binary: +1.0 if any assistant message carries ``tool_calls``,
    else 0.0.  Multiplied by :data:`TOOL_USE_REWARD_WEIGHT` downstream.

    Binary-ness is intentional: it gives GRPO a *within-group* variance
    signal (some rollouts used tools, some didn't) that directly rewards
    the tool-calling branch, unlike a logistic decay whose 0.95–0.99 range
    is indistinguishable under group-normalised advantages.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            used = False
            for m in t.get('messages', []) or []:
                if m.get('role') == 'assistant' and (m.get('tool_calls') or []):
                    used = True
                    break
            rewards.append(1.0 if used else 0.0)
        return rewards


class HotpotQAToolSuccessReward(Reward):
    """Double-incentive bonus: +1 iff the rollout used a tool AND got F1 >= threshold.

    Motivation (Round-4 post-mortem): with only :class:`HotpotQAToolUseReward`
    active, tool-calling rollouts paid a ``length_pen`` hit (up to -0.92)
    because compressed evidence + tool result + reasoning pushes the
    terminal message past :data:`FORMAT_MAX_CHARS`, which roughly cancels
    the raw +1 tool-use bonus.  Meanwhile the "guess a short answer"
    branch collects zero length_pen and keeps the F1 component on average.
    The GRPO advantage therefore flattens between the two branches and
    the policy regresses back toward guessing.

    This reward shifts the equilibrium by paying an *extra* bonus only
    when the tool path actually paid off, i.e. answer F1 >= threshold.
    That way:

        {tool + correct}  > {no-tool + correct}
                          > {no-tool + wrong}
                          ~ {tool + wrong}

    so the policy is pushed to explore tools only when they help, not as
    a blind behavioural habit.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            used = False
            for m in traj.get('messages', []) or []:
                if m.get('role') == 'assistant' and (m.get('tool_calls') or []):
                    used = True
                    break
            if not used:
                rewards.append(0.0)
                continue
            gold = ''
            for key, val in traj.get('user_data', []) or []:
                if key == 'ground_truth':
                    gold = val or ''
                    break
            pred = _extract_final_answer(_last_assistant_text(traj))
            f1, _em = _f1_score(pred, gold)
            rewards.append(1.0 if f1 >= TOOL_SUCCESS_F1_THRESHOLD else 0.0)
        return rewards


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
        # Join passages with a BLANK line (``\n\n``) rather than a single
        # ``\n``.  :class:`NativeChunker._split_preserving_code` uses
        # ``_PARAGRAPH_RE = r'\n\s*\n'`` as the paragraph boundary, so only a
        # blank line prevents adjacent passages from being fused into one
        # "mega-paragraph" that then gets sentence-hard-split across passage
        # boundaries.  The visible artefact of that bug: a single chunk whose
        # head belongs to ``[N]`` and whose tail belongs to ``[N+1]``, with
        # only the tail's anchor preserved by the condenser -- the head
        # passage becomes invisible to ``extract_compressed``.
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
    optim_step: int = 0,
) -> Tuple[List[float], List[float], List[float], List[float], List[float], List[float], List[float]]:
    """Weighted sum of F1 + format + tool-use + length-penalty (+ extract, diagnostic only).

    Round-2 rebalance:
      * ``F1`` weight raised 1.0 → 1.5 -- the only reward with real signal
        variance, must dominate the sum.
      * ``format`` weight dropped to 0.0 -- 100% saturated in the last run,
        contributes pure noise to GRPO within-group advantages.  Kept in
        the return tuple (and in metrics) as a monitoring-only signal.
      * ``extract`` (logistic penalty on tool-call count) replaced by
        ``tool_use`` (binary +1 if any tool call).  The old ``extract``
        value is still computed and returned for dashboard continuity but
        no longer enters the weighted sum.
      * ``tool_use`` multiplied by :data:`TOOL_USE_REWARD_WEIGHT`; zeroed
        during the first ``EXTRACT_WARMUP_FRAC * MAX_STEPS`` steps so the
        policy can first learn to answer before being pressured to invoke
        tools.  Default ``EXTRACT_WARMUP_FRAC=0.0`` disables the warmup.
      * ``length_pen`` unchanged -- safety net against repetition loops.

    Return tuple order preserved for caller shape stability; the new
    ``tool_use`` vector is appended at the end.
    """
    global _F1_REWARD, _FORMAT_REWARD, _EXTRACT_REWARD, _LENGTH_PENALTY, _TOOL_USE_REWARD, _TOOL_SUCCESS_REWARD
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _FORMAT_REWARD = HotpotQAFormatReward()
        _EXTRACT_REWARD = ExtractReward(midpoint=3.0, steepness=1.5)
        _LENGTH_PENALTY = HotpotQALengthPenalty()
        _TOOL_USE_REWARD = HotpotQAToolUseReward()
        _TOOL_SUCCESS_REWARD = HotpotQAToolSuccessReward()
    accuracy = _F1_REWARD(trajectories)
    fmt = _FORMAT_REWARD(trajectories)
    extract = _EXTRACT_REWARD(trajectories)   # diagnostic only
    length_pen_raw = _LENGTH_PENALTY(trajectories)
    tool_use = _TOOL_USE_REWARD(trajectories)
    tool_success = _TOOL_SUCCESS_REWARD(trajectories)

    # Round-5 fix: F1-gated length_pen.  Without this gate, long-and-wrong
    # rollouts (F1=0, len>>cap, length_pen=-0.8) and short-and-wrong ones
    # (F1=0, len<cap, length_pen=0) create a within-group advantage that
    # rewards "short nonsense" over "long nonsense" -- which is exactly the
    # pathology driving the step-115 collapse to ``Okay<|endoftext|>``.
    # Gating length_pen by F1 breaks this false signal: only rollouts that
    # actually answered correctly can incur a length penalty.  The length
    # penalty therefore becomes a tiebreaker *among correct answers*
    # (preferring concise) rather than a global "short is better" bias.
    if LENGTH_PEN_F1_GATE > 0.0:
        length_pen = [
            lp if acc > LENGTH_PEN_F1_GATE else 0.0
            for lp, acc in zip(length_pen_raw, accuracy)
        ]
    else:
        length_pen = length_pen_raw

    # Curriculum: mute tool-use bonus during warmup so F1 alone teaches the
    # policy to answer before the tool-call incentive kicks in.
    warmup_steps = int(EXTRACT_WARMUP_FRAC * MAX_STEPS)
    tool_weight = 0.0 if optim_step < warmup_steps else TOOL_USE_REWARD_WEIGHT
    tool_success_weight = (
        0.0 if optim_step < warmup_steps else TOOL_SUCCESS_REWARD_WEIGHT)

    total = [
        F1_REWARD_WEIGHT * a
        + FORMAT_REWARD_WEIGHT * f
        + tool_weight * tu
        + tool_success_weight * ts
        + LENGTH_PENALTY_WEIGHT * lp
        for a, f, tu, ts, lp in zip(accuracy, fmt, tool_use, tool_success, length_pen)
    ]
    return total, fmt, accuracy, extract, length_pen, tool_use, tool_success


# ========== Agentic rollout ==========
# Trajectory-level media fields that may ride on a prompt.  Defined once so
# ``_Rollout.__init__`` and ``_FrozenContext.freeze_delta`` agree on the set.
_MEDIA_KEYS = ('images', 'videos', 'audios')


class _FrozenContext:
    """Per-rollout monotone-append cache for chunked+condensed context.

    Multi-turn rollouts re-render the *entire* trajectory at each turn, which
    naively would run the chunker + condenser over the whole history every
    time -- compressing already-compressed content, silently drifting
    ``<block_N>`` numbering, and potentially shredding the
    ``<block_N>...</block_N>`` tag pair across sentence splits.

    Solution: **freeze-and-append**.  Each call to :meth:`freeze_delta` only
    chunks and condenses the messages NEW since the last freeze, appending
    the results to :attr:`full_chunks` and :attr:`compressed_chunks`.
    Because the trajectory is append-only (see ``_append_terminal`` /
    ``_append_tool_turn``), the newly chunked range never overlaps the
    frozen range, so ``block_N`` stays bound to the same underlying text
    across all rounds: a model tool call to ``extract_compressed(N)`` emitted
    in round 3 still resolves to the exact same original passage that
    ``<block_N>`` wrapped in round 1.

    Trajectory-level media are processed **only on the first freeze**;
    subsequent freezes route the partial trajectory WITHOUT those fields to
    avoid duplicate media chunks.
    """
    __slots__ = ('frozen_msg_count', 'full_chunks',
                 'compressed_chunks', 'media_frozen', 'query_hint')

    def __init__(self) -> None:
        self.frozen_msg_count: int = 0
        self.full_chunks: List[Chunk] = []
        self.compressed_chunks: List[Chunk] = []
        self.media_frozen: bool = False
        # Cached on the first freeze (the question is part of the prompt's
        # user message and never changes across rounds); subsequent freezes
        # reuse it to keep biasing TF-IDF on tool-output chunks toward
        # answer-critical facts.
        self.query_hint: str = ''

    def freeze_delta(self, trajectory: Dict[str, Any],
                      chunker: NativeChunker,
                      condenser: NativeCondenser) -> None:
        """Chunk + condense only unfrozen messages/media; append to cache.

        The first freeze applies the full ``keep_ratio`` gradient (early
        chunks compressed hard, late chunks gently).  Subsequent freezes
        contain only the latest assistant/tool turns, which semantically sit
        at the tail of the gradient -- we pin them to the gentler
        ``max_keep_ratio`` rather than re-applying a local gradient that
        would mistakenly over-compress their first sub-chunk.

        The question extracted from the initial user message is threaded
        through as ``query_hint`` so answer-critical sentences (e.g. the
        ``started in 1989`` line in ``First for Women``) win out over
        filler sentences that happen to have higher raw TF-IDF mass.
        """
        total_msgs = trajectory['messages']
        new_msgs = total_msgs[self.frozen_msg_count:]
        needs_media = (not self.media_frozen and
                        any(trajectory.get(k) for k in _MEDIA_KEYS))
        if not (new_msgs or needs_media):
            return

        if not self.query_hint:
            self.query_hint = _extract_query_hint(total_msgs)

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
        condense_kwargs: Dict[str, Any] = {'query_hint': self.query_hint}
        if self.frozen_msg_count > 0:
            condense_kwargs['keep_ratio'] = condenser.max_keep_ratio
        new_compressed = condenser.condense(new_full, **condense_kwargs)

        self.full_chunks.extend(new_full.chunks)
        self.compressed_chunks.extend(new_compressed.chunks)
        self.frozen_msg_count = len(total_msgs)

    def render_display(self) -> Dict[str, Any]:
        """Emit the accumulated compressed state as a sampler-ready dict."""
        return Chunks(chunks=list(self.compressed_chunks)).to_trajectory()

    def render_full(self) -> Chunks:
        """Emit the accumulated pre-compression chunks for ExtractCompressed."""
        return Chunks(chunks=list(self.full_chunks))


class _Rollout:
    """Mutable bookkeeping for one prompt's multi-turn unroll."""
    __slots__ = ('trajectory', 'final_sequence', 'turns', 'done', 'frozen')

    def __init__(self, prompt_trajectory: Dict[str, Any]) -> None:
        self.trajectory: Dict[str, Any] = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        # Preserve trajectory-level media so the first freeze can chunk them;
        # later freezes are skipped via ``frozen.media_frozen``.
        for _k in _MEDIA_KEYS:
            if prompt_trajectory.get(_k):
                self.trajectory[_k] = list(prompt_trajectory[_k])
        self.final_sequence = None  # SampledSequence of the terminal turn
        self.turns = 0
        self.done = False
        self.frozen: _FrozenContext = _FrozenContext()


def _append_terminal(r: _Rollout, decoded: str) -> None:
    r.trajectory['messages'].append(
        {'role': 'assistant', 'content': _clean_assistant_output(decoded)})
    r.done = True


def _append_tool_turn(
    r: _Rollout,
    decoded: str,
    tool_calls: List[Dict[str, Any]],
    tool_mgr: ToolManager,
    turn_idx: int,
) -> None:
    r.trajectory['messages'].append({
        'role': 'assistant',
        'content': _clean_assistant_output(decoded),
        'tool_calls': [
            {'tool_name': tc['tool_name'],
             'arguments': (tc['arguments']
                           if isinstance(tc['arguments'], str)
                           else json.dumps(tc['arguments'], ensure_ascii=False))}
            for tc in tool_calls
        ],
    })
    for i, tc in enumerate(tool_calls):
        r.trajectory['messages'].append({
            'role': 'tool',
            'content': tool_mgr.dispatch(tc),
            'tool_call_id': f'call_t{turn_idx}_i{i}',
        })


# ---- Rollout trace dump (post-mortem crash analysis) ---------------------
# Append-only JSONL file.  One line per ``run_agentic_rollouts`` turn, each
# line describing ONE randomly picked active rollout's full state after that
# turn's sampling.  Designed to be grepped / replayed after a training crash:
# every line is self-contained JSON with the compressed prompt actually fed
# to the sampler, the pre-compression chunk list, the cumulative tool-call
# count and (once the rollout terminates) the final boxed answer.
#
# Path is configurable via ``ROLLOUT_TRACE_PATH`` so multiple runs can
# coexist without clobbering each other; set to empty string to disable.
_ROLLOUT_TRACE_PATH = os.environ.get(
    'ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')


def _dump_random_rollout_trace(
    turn: int,
    active: List[_Rollout],
    displays: List[Dict[str, Any]],
    responses: List[Any],
) -> None:
    """Append one random active rollout's post-turn state to JSONL.

    Fields captured:
      * ``compressed``    -- the sampler-ready display (what the model saw
        this turn after chunk + condense + ``<block_N>`` rendering).
      * ``full``          -- the pre-compression chunk list (role + content
        preview); lets you see whether the condenser dropped an anchor or
        tore a ``<block_N>`` pair.
      * ``tool_call_count`` -- cumulative ``tool_calls`` entries across all
        assistant messages so far; a collapse to 0 across many turns is the
        "policy stopped exploring" signature.
      * ``rewards``       -- per-component reward snapshot (``f1``, ``format``,
        ``extract``, ``length_pen``, ``total``) computed on the trajectory
        as of this turn.  Non-terminal turns naturally have ``f1=0`` /
        ``format=0``; the progression from turn N to turn N+1 is what
        matters, not the absolute value.
      * ``last_decoded``  -- raw decoded text this turn (before cleaning /
        tool-call stripping).
      * ``final_answer``  -- ``\\boxed{...}`` extraction if the rollout
        terminated on this turn, else ``''``.
    """
    if not _ROLLOUT_TRACE_PATH or not active:
        return
    idx = random.randrange(len(active))
    r = active[idx]
    resp = responses[idx] if idx < len(responses) else None

    tool_call_count = sum(
        len(m.get('tool_calls') or [])
        for m in r.trajectory.get('messages', [])
        if m.get('role') == 'assistant')

    full_chunks_preview = [
        {'role': c.get('role'),
         'type': c.get('type'),
         'content': (c.get('content') if isinstance(c.get('content'), str)
                     else repr(c.get('content'))[:500])}
        for c in r.frozen.full_chunks
    ]

    last_decoded = ''
    if resp is not None and getattr(resp, 'sequences', None):
        last_decoded = resp.sequences[0].decoded or ''

    final_answer = ''
    if r.done:
        final_answer = _extract_final_answer(_last_assistant_text(r.trajectory))

    # Per-component reward snapshot for the picked rollout.  Computed here
    # (rather than waiting for the post-batch ``compute_rewards`` call) so
    # mid-trajectory turns also carry a reward signal, letting you line up
    # "compression quality" against "reward collected so far" in the trace.
    # Non-terminal turns will typically have f1=0 / format=0 (no \boxed yet)
    # and extract=logistic(tool_call_count); that's intended -- we want to
    # see the reward build up, not only the final value.
    reward_snapshot: Dict[str, float] = {}
    try:
        global _F1_REWARD, _FORMAT_REWARD, _EXTRACT_REWARD, _LENGTH_PENALTY, _TOOL_USE_REWARD, _TOOL_SUCCESS_REWARD
        if _F1_REWARD is None:
            _F1_REWARD = HotpotQAF1Reward()
            _FORMAT_REWARD = HotpotQAFormatReward()
            _EXTRACT_REWARD = ExtractReward(midpoint=3.0, steepness=1.5)
            _LENGTH_PENALTY = HotpotQALengthPenalty()
            _TOOL_USE_REWARD = HotpotQAToolUseReward()
            _TOOL_SUCCESS_REWARD = HotpotQAToolSuccessReward()
        traj_list = [r.trajectory]
        f1_val = _F1_REWARD(traj_list)[0]
        fmt_val = _FORMAT_REWARD(traj_list)[0]
        ext_val = _EXTRACT_REWARD(traj_list)[0]
        len_val = _LENGTH_PENALTY(traj_list)[0]
        tu_val = _TOOL_USE_REWARD(traj_list)[0]
        ts_val = _TOOL_SUCCESS_REWARD(traj_list)[0]
        # Weighted total mirrors ``compute_rewards`` -- minus the curriculum
        # mute on ``tool_use`` since we do not have ``optim_step`` at rollout
        # time; the raw components are also emitted for post-hoc inspection.
        total_val = (
            F1_REWARD_WEIGHT * f1_val
            + FORMAT_REWARD_WEIGHT * fmt_val
            + TOOL_USE_REWARD_WEIGHT * tu_val
            + TOOL_SUCCESS_REWARD_WEIGHT * ts_val
            + LENGTH_PENALTY_WEIGHT * len_val)
        reward_snapshot = {
            'f1': float(f1_val),
            'format': float(fmt_val),
            'extract': float(ext_val),
            'length_pen': float(len_val),
            'tool_use': float(tu_val),
            'tool_success': float(ts_val),
            'total': float(total_val),
        }
    except Exception as e:  # pragma: no cover -- tracing must never crash training
        logger.warning('rollout trace reward snapshot failed: %s', e)
        reward_snapshot = {'error': repr(e)}

    record = {
        'ts': time.time(),
        'turn': turn,
        'active_size': len(active),
        'picked_idx': idx,
        'rollout_id': id(r),
        'tool_call_count': tool_call_count,
        'done': bool(r.done),
        'rewards': reward_snapshot,
        'compressed': displays[idx],
        'full_chunks': full_chunks_preview,
        'last_decoded': last_decoded,
        'final_answer': final_answer,
    }
    try:
        with open(_ROLLOUT_TRACE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:  # pragma: no cover -- tracing must never crash training
        logger.warning('rollout trace dump failed: %s', e)


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
            # Incrementally chunk+condense only the messages new since last
            # freeze; the accumulated state is then rendered as a sampler
            # display + bound to a fresh ExtractCompressed so ``block_N``
            # resolves identically across rounds.
            r.frozen.freeze_delta(r.trajectory, chunker, condenser)
            displays.append(r.frozen.render_display())
            tool_mgrs.append(ToolManager(
                [ExtractCompressed(r.frozen.render_full())]))

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

        # Post-mortem trace: dump ONE random active rollout's compressed
        # display + full chunks + tool-call count + final answer to JSONL.
        # Kept after the response-processing loop so ``r.done`` / final
        # answer already reflect this turn's outcome.
        _dump_random_rollout_trace(turn, active, displays, responses)

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

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=2, pp_size=2)
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

    # lora_config = LoraConfig(
    #     target_modules='all-linear',
    #     r=LORA_RANK,
    #     lora_alpha=LORA_RANK * 2,
    #     lora_dropout=0.05,
    # )

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

    # model.add_adapter_to_model(ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)
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
            #'enable_lora': True,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    # Chunker / condenser live on the driver (pure-Python, no GPU).
    # ``passage_boundary_re=r'^\[\d+\]\s+'`` makes each numbered HotpotQA
    # passage its own chunk, so ``<block_N>`` index == passage index ``[N]``
    # and ``ExtractCompressed(N)`` returns exactly that passage's original
    # text (not a mixed blob of 3-4 passages that happened to greedy-pack
    # into the same window).
    chunker = NativeChunker(
        model_id=MODEL_ID, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        passage_boundary_re=r'^\[\d+\]\s+')
    condenser = NativeCondenser(keep_ratio=(0.2, KEEP_RATIO), skip_system=True)

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
        # Round-3 exploration: 1.0 was too conservative (3% tool-call rate),
        # but 1.3 produced garbage tokens (e.g. Cyrillic chars mixed into
        # English answers).  1.15 + top_p=0.9 keeps exploration while
        # tightening the sampling tail.
        temperature=1.15, top_p=0.9,
        # Crash post-mortem showed the policy degenerated into 200-char
        # blocks repeated 90+ times until MAX_NEW_TOKENS.  ``repetition_penalty``
        # directly suppresses that loop at decode time; pair with the
        # reward-side length-penalty for defense-in-depth.
        repetition_penalty=1.1,
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

        ckpt_manager.sync_weights(merge_and_sync=True)
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

        (total_rewards, brevity_rewards, accuracy_rewards, extract_rewards,
         length_penalties, tool_use_rewards, tool_success_rewards) = compute_rewards(
            all_trajectories, optim_step=optim_step)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': brevity_rewards,
                'f1': accuracy_rewards,
                'extract': extract_rewards,
                'length_pen': length_penalties,
                'tool_use': tool_use_rewards,
                'tool_success': tool_success_rewards,
            },
        )

        advantages = advantage_fn(
            total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Round-5 fix: homogeneous-group filter (script-level).
        # Zero out advantages for samples from groups where all rewards
        # are identical (max |adv| < threshold).  Prevents the policy
        # from drifting on zero-variance groups where the GRPO advantage
        # is pure numerical noise.
        if GRPO_HOMOGENEOUS_THRESHOLD > 0.0:
            n_active = 0
            for g_start in range(0, len(advantages), NUM_GENERATIONS):
                g_end = min(g_start + NUM_GENERATIONS, len(advantages))
                group = advantages[g_start:g_end]
                max_abs = max(abs(a) for a in group)
                if max_abs > GRPO_HOMOGENEOUS_THRESHOLD:
                    n_active += (g_end - g_start)
                else:
                    for i in range(g_start, g_end):
                        advantages[i] = 0.0
            active_ratio = n_active / len(advantages) if advantages else 1.0
            logger.info('[Homogeneous filter] active_sample_ratio=%.3f', active_ratio)
        else:
            active_ratio = 1.0

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
        log_dict['active_sample_ratio'] = active_ratio
        log_dict = {key: value for key, value in log_dict.items() if '_std' not in key}
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-final')


if __name__ == '__main__':
    main()
