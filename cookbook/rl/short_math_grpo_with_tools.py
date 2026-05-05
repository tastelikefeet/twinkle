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
        2. Replace each chunk with a structured summary index
           ``<first sentence> (Related: kw1, kw2, ...)``.
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
from twinkle_agentic.data_format.chunk import Chunk, Chunks
from twinkle_agentic.reward.extract_reward import ExtractReward
from twinkle_agentic.tools.extract import ExtractCompressed
from twinkle_agentic.tools.tool_manager import ToolManager

import swanlab

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
# Round-6: bumped 4 -> 6 so the policy has headroom for multi-extract
# patterns ("peek block A, insufficient, peek blocks B+C, then answer").
# Prior run collapsed to a 1-tool-call attractor (89% of done rollouts
# had tool_call_count=1), which capped F1 at ~0.75 because a single
# guessed block often misses the second hop for HotpotQA multi-hop
# questions.  A larger budget does NOT force more calls; it only lifts
# the ceiling so gradients from ``HotpotQAReasoningReward`` +
# ``HotpotQAToolSuccessReward`` can shape a richer retrieve pattern.
MAX_TURNS = int(os.environ.get('MAX_TURNS', 6))            # hard cap on tool-call turns
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1024))        # chars per chunk (NativeChunker)
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 0))    # sliding-window overlap

# ---- HotpotQA dataset encode knobs ----
HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# ---- Reward weights (Round-6 shape; see reward-class docstrings) ----
# Prior Round-4/5 attribution analysis on the rollout_trace showed
# ``tool_use`` contributed +216% of the total_reward gain while F1
# stagnated at 0.75 (tool_use variance collapsed to 0 after step 40, so
# its huge weight produced NO gradient anyway).  Round-6 re-allocates:
#   - TOOL_USE: 1.0 -> 0.4   (lower but non-zero: keep as soft prior)
#   - TOOL_SUCCESS: 0.5 -> 0.8 (push F1-conditioned tool-use harder)
#   - REASONING: new 0.5   (pull back the pre-tool reasoning chain that
#     collapsed to 0 chars in Round-4/5, forcing multi-hop block choice
#     to be a single forward-pass keyword match)
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
FORMAT_REWARD_WEIGHT = float(os.environ.get('FORMAT_REWARD_WEIGHT', 0.0))
TOOL_USE_REWARD_WEIGHT = float(os.environ.get('TOOL_USE_REWARD_WEIGHT', 0.4))
TOOL_SUCCESS_REWARD_WEIGHT = float(os.environ.get(
    'TOOL_SUCCESS_REWARD_WEIGHT', 0.8))
TOOL_SUCCESS_F1_THRESHOLD = float(os.environ.get(
    'TOOL_SUCCESS_F1_THRESHOLD', 0.5))
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
EXTRACT_REWARD_WEIGHT = float(os.environ.get('EXTRACT_REWARD_WEIGHT', 0.0))
REASONING_REWARD_WEIGHT = float(os.environ.get('REASONING_REWARD_WEIGHT', 0.5))
# ``HotpotQAReasoningReward`` saturates at REASONING_MIN_CHARS chars of
# reasoning BEFORE the first ``<tool_call>`` tag.  Short preambles
# (<REASONING_FLOOR_CHARS) receive 0; longer text ramps linearly to 1.0
# at REASONING_MIN_CHARS and plateaus beyond.  Defaults roughly match the
# EARLY-phase rollout preamble length seen in rollout_trace.jsonl
# (~500-900 chars with full reasoning, ~85 chars at the collapsed state).
REASONING_FLOOR_CHARS = int(os.environ.get('REASONING_FLOOR_CHARS', 50))
REASONING_MIN_CHARS = int(os.environ.get('REASONING_MIN_CHARS', 200))

# ---- Cold-start curriculum (Plan B only: tool_use weight; Plan A disabled) ----
# History: initial A+B (temp 1.3 + weight 1.5) over-shot badly --
# temperature 1.3 poisoned turn-2 generation with multi-lingual garbage
# tokens and forced premature EOS; weight 1.5 taught the policy "call
# tool -> collect 1.5 reward -> stop" (i.e. empty post-tool assistant
# turns).  Both branches' F1 collapsed from 0.71 to 0.35 within 18 steps.
# Resolution: kill Plan A entirely (all temperatures pinned at 1.0) and
# soften Plan B so tool_use weight stays below the F1 weight (1.5).
# This keeps the F1 channel as the dominant signal while still giving
# the tool branch a modest head-start during cold start.
CURRICULUM_COLD_STEPS = int(os.environ.get('CURRICULUM_COLD_STEPS', 15))
CURRICULUM_TRANSITION_STEPS = int(
    os.environ.get('CURRICULUM_TRANSITION_STEPS', 40))
# Plan A disabled: all phases use the same (stable) sampling temperature.
# Kept as env knobs so they can be re-enabled without a code edit, but
# their defaults are pinned to the stable value.
SAMPLING_TEMP_COLD = float(os.environ.get('SAMPLING_TEMP_COLD', 1.0))
SAMPLING_TEMP_TRANSITION = float(os.environ.get('SAMPLING_TEMP_TRANSITION', 1.0))
SAMPLING_TEMP_STABLE = float(os.environ.get('SAMPLING_TEMP_STABLE', 1.0))
# Plan B softened: weight stays strictly below F1_REWARD_WEIGHT (1.5) so
# "call tool AND answer correctly" dominates "call tool AND skip answer".
TOOL_USE_WEIGHT_COLD = float(os.environ.get('TOOL_USE_WEIGHT_COLD', 0.7))
TOOL_USE_WEIGHT_TRANSITION = float(
    os.environ.get('TOOL_USE_WEIGHT_TRANSITION', 0.55))
# ``format_reward`` is +1 only if the boxed-answer turn is shorter than
# this.  Raised to 5000 in Round-4: a legitimate retrieve+reason+answer
# rollout takes 2000-4000 chars, so tighter gates unfairly punished the
# tool-use branch.  Only real repetition loops (>5k) now hit the penalty.
FORMAT_MAX_CHARS = int(os.environ.get('FORMAT_MAX_CHARS', 5000))

# ---- Rollout trace dump (post-mortem diagnosis) ----
# Append-only JSONL: one line per turn per run, carrying the compressed
# prompt, pre-compression chunks, decoded completion, cumulative tool
# count, and per-component reward snapshot for ONE randomly picked active
# rollout.  Empty string disables.  File is truncated at main() start.
_ROLLOUT_TRACE_PATH = os.environ.get(
    'ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')

# ========== System Prompt ==========
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Give a short factual answer (a name, '
    'entity, date, or "yes"/"no") inside \\boxed{}. Do not include extra '
    'words in the box.\n\n'
    'CONTEXT FORMAT: The provided paragraphs are wrapped as '
    '<block_N>...</block_N>. For long paragraphs, only the first sentence '
    'is shown, followed by a parenthetical "(Related: keyword1, keyword2, '
    '...)" listing additional concrete terms (years, proper nouns, numbers) '
    'present in the hidden remainder. Short paragraphs are shown in full '
    'with no such parenthetical.\n\n'
    'Use the first sentence plus the Related-list to decide whether a '
    'block probably contains the fact you need. When it does, call the '
    '``extract_compressed`` tool with the relevant block numbers to recall '
    'the full original text. When the visible text is already sufficient '
    '(e.g. the answer is a year mentioned in Related, or the first '
    'sentence directly states the fact), answer immediately without '
    'calling the tool.\n\n'
    'TOOL CALL FORMAT: Emit tool calls inside a single fenced block like '
    'this, then stop generating and wait for the tool result:\n'
    '<tool_call>\n'
    '{"name": "extract_compressed", "arguments": {"blocks": [1, 3]}}\n'
    '</tool_call>\n\n'
    'BEFORE emitting a tool call, briefly think out loud (2-4 sentences): '
    'name the entities you are tracing, say which blocks look relevant '
    'based on their first sentence and Related list, and explain why. '
    'Naked tool calls with no preamble waste signal; a short reasoning '
    'chain before the ``<tool_call>`` tag is rewarded. You may also call '
    '``extract_compressed`` again in a later turn if the first recall did '
    'not contain the answer -- targeted follow-up retrievals are '
    'rewarded, blind or redundant ones are not.')

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


# ========== HotpotQA preprocessor ==========
_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')


# ========== Plan B: Structured Summary Index (replaces NativeCondenser) ==========
# For every text chunk we emit a "summary index" instead of TF-IDF-selected
# sentences: the full first sentence is preserved, followed by an inline
# ``(Related: k1, k2, ...)`` parenthetical listing the most informative
# CONCRETE keywords extracted from the hidden body (years, proper nouns,
# numbers).  The model then decides per-block whether to call
# ``extract_compressed(N)`` to recall the full passage.
#
# Format chosen to stay close to the base (no-SFT) Qwen3.5-4B's pretraining
# distribution: natural English prose with parenthetical aside, NOT
# machine-log-style bracket metadata.  The ``<block_N>`` XML wrapper is
# kept because ``ExtractCompressed`` uses it to locate chunks for recall.
#
# Non-compressible chunks (system messages, tool-call structural payloads,
# multi-modal media) are passed through unchanged.
_PROTECTED_KINDS = frozenset({'tool_call', 'tool_response'})
_PROTECTED_TYPES = frozenset({'image', 'video', 'audio'})

_YEAR_RE = re.compile(r'\b(?:1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b')
# Proper noun: a capitalised word (optionally hyphenated) optionally followed
# by up to three more capitalised words.  Good enough to catch "Timothy Shay
# Arthur", "New Jersey", "Bauer Media" without pulling in a full NER model.
_PROPER_NOUN_RE = re.compile(r'\b[A-Z][A-Za-z\-]+(?:\s+[A-Z][A-Za-z\-]+){0,3}\b')
_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')
# Sentence-end followed by whitespace and a capital letter / digit.
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

# Words that pass the ``_PROPER_NOUN_RE`` capitalisation test but carry no
# retrieval signal (sentence-initial articles, month/weekday names, common
# abstract terms).  Filtered from the hidden-keyword list.
_KEYWORD_STOP = frozenset({
    'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 'A', 'An',
    'Is', 'Are', 'Was', 'Were', 'Be', 'Been', 'It', 'He', 'She', 'They',
    'We', 'I', 'You', 'His', 'Her', 'Their', 'Our', 'Its',
    'However', 'Although', 'Because', 'While', 'Since', 'When', 'Where',
    'After', 'Before', 'During', 'Through', 'Without',
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December',
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Saturday', 'Sunday',
})


def _split_first_sentence(text: str) -> Tuple[str, str]:
    """Return ``(first_sentence, rest)`` with conservative boundary detection.

    Falls back to returning the whole text as ``first_sentence`` (with
    empty ``rest``) when no ``. [A-Z]`` boundary exists, so single-sentence
    chunks degrade gracefully.
    """
    text = (text or '').strip()
    if not text:
        return '', ''
    parts = _SENTENCE_END_RE.split(text, maxsplit=1)
    if len(parts) == 1:
        return text, ''
    return parts[0].rstrip(), parts[1].lstrip()


def _split_all_sentences(text: str) -> List[str]:
    """Split ``text`` on sentence boundaries, preserving each sentence.

    Used by :func:`_pick_relational_sentence` to score every sentence in
    a passage (not just the first) by entity density so the compressor
    can keep one "relation-bearing" sentence in addition to the first.
    """
    text = (text or '').strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END_RE.split(text) if s.strip()]


def _entity_density(sentence: str) -> int:
    """Count year + proper-noun + number matches in ``sentence``.

    Used as a cheap proxy for "this sentence carries multi-hop bridging
    content".  Relational sentences in HotpotQA ("X was directed by Y in
    Z in 1962") typically pack 3-5 entities, while filler ("The film was
    a commercial success.") packs 0-1.  Higher density = more useful for
    multi-hop retrieval.
    """
    if not sentence:
        return 0
    c = 0
    for m in _YEAR_RE.finditer(sentence):
        y = m.group()
        if 1000 <= int(y) <= 2999:
            c += 1
    c += len(_PROPER_NOUN_RE.findall(sentence))
    c += len(_NUMBER_RE.findall(sentence))
    return c


def _pick_relational_sentence(sentences: List[str], first_sent: str) -> str:
    """Return the non-first sentence with highest entity density, or ''.

    Skips the first sentence (already kept verbatim).  Skips sentences
    that share >=70% word overlap with the first (redundant).  Requires
    at least 2 entities to be considered "relational enough"; otherwise
    returns empty string and the compressor falls back to first-only.
    """
    if len(sentences) <= 1:
        return ''
    first_words = set(first_sent.lower().split())
    best, best_score = '', 1  # require density >= 2 to trigger
    for sent in sentences[1:]:
        sw = set(sent.lower().split())
        if first_words and sw:
            overlap = len(first_words & sw) / max(1, len(sw))
            if overlap >= 0.7:
                continue
        score = _entity_density(sent)
        if score > best_score:
            best_score, best = score, sent
    return best


def _extract_hidden_keywords(
    first_sent: str, rest: str, max_keywords: int = 12,
) -> List[str]:
    """Extract concrete retrieval-useful keywords from the hidden body.

    Priority order (higher-signal first):
      1. 4-digit years in [1000, 2999]
      2. Multi-word proper nouns (minus ``_KEYWORD_STOP``)
      3. Other numbers

    Deduplicates against ``first_sent`` tokens so the hidden list only
    carries information NOT already visible.  Caps at ``max_keywords``.

    ``max_keywords`` default bumped 8 -> 12 so long (>600 char) passages
    still squeeze a recognisable entity footprint into the summary without
    relying on a tool call.  Compression ratio on a ~1000-char passage
    with 12 keywords + first sentence lands around 20-25%.
    """
    if not rest:
        return []
    first_lower = first_sent.lower()
    seen: List[str] = []
    seen_lower: set = set()

    def _push(tok: str) -> bool:
        low = tok.lower()
        if low in first_lower or low in seen_lower:
            return False
        if tok in _KEYWORD_STOP:
            return False
        seen_lower.add(low)
        seen.append(tok)
        return True

    for m in _YEAR_RE.finditer(rest):
        year = m.group()
        if 1000 <= int(year) <= 2999:
            _push(year)
        if len(seen) >= max_keywords:
            return seen

    for m in _PROPER_NOUN_RE.finditer(rest):
        _push(m.group().strip())
        if len(seen) >= max_keywords:
            return seen

    for m in _NUMBER_RE.finditer(rest):
        num = m.group()
        if num in seen_lower:
            continue
        _push(num)
        if len(seen) >= max_keywords:
            return seen

    return seen


def _generate_passage_index(chunk: Chunk) -> Chunk:
    """Build a summary-index chunk from a text passage chunk.

    Output content layout (appears INSIDE ``<block_N>...</block_N>`` after
    :meth:`Chunks.to_trajectory` wrapping, which the ``ExtractCompressed``
    tool still depends on for recall)::

        <first sentence>[ <relational sentence>] (Related: k1, k2, ...)

    Round-6 upgrade: additionally keep ONE "relational" sentence when
    available -- the non-first sentence with the highest entity density
    (years + proper nouns + numbers).  Background: prior Plan-B format
    preserved only ``first_sent + (Related: keywords)``.  Keywords carry
    entities but drop the RELATIONAL glue ("directed by", "located in",
    "married to") that binds two hops together.  HotpotQA multi-hop
    questions therefore could not be resolved from the compressed index
    alone and forced a tool call on ~100% of rollouts; the subsequent
    1-tool-call attractor capped F1 at ~0.75.  Keeping one additional
    relation-bearing sentence lifts compression ratio from ~20% to
    ~30-35% on a ~1000-char passage but lets the model resolve the
    easier 2-hop joins WITHOUT a tool call, and gives it a much better
    prior for WHICH block to extract when a tool call IS needed.

    Design choice (unchanged): the prior ``[compressed X% | hidden: ...]``
    machine-log header stays removed -- parenthetical "(Related: ...)"
    is in-distribution for base Qwen3.5-4B prose; bracket/pipe metadata
    is not.

    Marks ``raw['condensed'] = True`` so ``Chunks.to_trajectory`` wraps
    the result in ``<block_N>...</block_N>``.  Non-compressible chunks
    (system / tool-call payload / media) pass through unchanged.
    """
    content = chunk.get('content', '')
    if not isinstance(content, str) or not content.strip():
        return chunk
    if chunk.get('role') == 'system':
        return chunk
    if chunk.get('type') in _PROTECTED_TYPES:
        return chunk
    raw = chunk.get('raw')
    if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
        return chunk

    first_sent, rest = _split_first_sentence(content)
    if not rest:
        # Nothing to hide; leave as-is (and do not mark condensed).
        return chunk

    # Round-6: also keep a second "relational" sentence (if one exists
    # with >=2 entities and not too redundant with the first).
    all_sents = _split_all_sentences(content)
    relational = _pick_relational_sentence(all_sents, first_sent)

    hidden_keywords = _extract_hidden_keywords(first_sent, rest)
    # Natural-language suffix: a parenthetical "Related: ..." tail after
    # the first sentence.  Drops the suffix entirely if no keywords were
    # found, so the rendered block degenerates to first-sentence-only
    # rather than a dangling "(Related: )".
    if hidden_keywords:
        suffix = f' (Related: {", ".join(hidden_keywords)})'
    else:
        suffix = ''

    if relational:
        # Insert the relational sentence between first_sent and the
        # parenthetical.  Use a space (not a newline) so the whole block
        # stays a single prose paragraph, matching the base-model's
        # paragraph-shaped pretraining distribution.
        new_content = f'{first_sent} {relational}{suffix}'
    else:
        new_content = f'{first_sent}{suffix}'

    new_chunk: Chunk = dict(chunk)  # type: ignore[assignment]
    new_chunk['content'] = new_content
    if isinstance(raw, dict):
        new_chunk['raw'] = {**raw, 'condensed': True}
    else:
        new_chunk['raw'] = {'condensed': True}
    return new_chunk


def _generate_structured_index(chunks: Chunks) -> Chunks:
    """Apply :func:`_generate_passage_index` to every chunk in ``chunks``."""
    return Chunks(chunks=[_generate_passage_index(c) for c in chunks.chunks])


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
_REASONING_REWARD = None

# ---- Cold-start curriculum runtime state (Plans A + B) ----
# ``_CURRENT_STEP`` is written by ``main()`` at the start of each training
# iteration and read by the curriculum accessors below.  Kept at module
# scope so ``compute_rewards`` and ``_dump_random_rollout_trace`` can both
# see the same schedule without threading an extra parameter through the
# rollout pipeline.
_CURRENT_STEP: int = 0


def _curriculum_temperature() -> float:
    """Sampling temperature accessor (Plan A DISABLED).

    All three phases default to ``SAMPLING_TEMP_STABLE`` (1.0) so the
    sampler temperature is effectively constant.  The scheduled env
    knobs are preserved so Plan A can be re-enabled if needed without a
    code edit, but the active defaults make the function a no-op.

    Rationale: the previous ``temp=1.3`` cold-start poisoned turn-2
    generation (multi-lingual garbage + premature EOS) and the resulting
    F1 crash wasn't worth the intra-group variance gain."""
    if _CURRENT_STEP < CURRICULUM_COLD_STEPS:
        return SAMPLING_TEMP_COLD
    if _CURRENT_STEP < CURRICULUM_TRANSITION_STEPS:
        return SAMPLING_TEMP_TRANSITION
    return SAMPLING_TEMP_STABLE


def _curriculum_tool_use_weight() -> float:
    """Plan B: amplified ``tool_use`` reward weight during cold start,
    decaying through the transition window back to the stable value
    ``TOOL_USE_REWARD_WEIGHT``.  Weight is kept strictly below
    ``F1_REWARD_WEIGHT`` (1.5) so the policy cannot accumulate more
    reward by "call tool + skip answer" than by "answer correctly"."""
    if _CURRENT_STEP < CURRICULUM_COLD_STEPS:
        return TOOL_USE_WEIGHT_COLD
    if _CURRENT_STEP < CURRICULUM_TRANSITION_STEPS:
        return TOOL_USE_WEIGHT_TRANSITION
    return TOOL_USE_REWARD_WEIGHT


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


class HotpotQALengthPenalty(Reward):
    """Negative reward proportional to terminal-message length overflow.

    Returns 0 if the final assistant message is within
    :data:`FORMAT_MAX_CHARS`, else a linearly growing penalty that reaches
    -1 when the message fills ``MAX_NEW_TOKENS * 4`` chars (the empirical
    4 chars/token ceiling).  This directly counteracts the "fill the budget
    with repetition" exploit -- the Phase-2 blowup observed in the
    SwanLab dashboard.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        budget = max(1, MAX_NEW_TOKENS * 4 - FORMAT_MAX_CHARS)
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            overflow = max(0, len(text) - FORMAT_MAX_CHARS)
            rewards.append(-min(1.0, overflow / budget))
        return rewards


class HotpotQAToolUseReward(Reward):
    """Binary +1 reward for using ``extract_compressed`` at least once.

    Purpose: inject CROSS-ROLLOUT VARIANCE on the tool-calling axis.
    Because ``ExtractReward`` collapses to a ~0.99 constant when the
    policy stops calling tools (its reverse-sigmoid decay is flat in
    [0, 2] calls), GRPO sees zero within-group variance on tool-use and
    has no gradient signal to push the policy back toward the tool branch.
    A binary {0, 1} signal keeps the variance alive: any prompt whose
    8 rollouts split (some used tools, some didn't) generates a non-zero
    advantage for every rollout in that group.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            used = any(
                m.get('role') == 'assistant' and (m.get('tool_calls') or [])
                for m in (t.get('messages') or []))
            rewards.append(1.0 if used else 0.0)
        return rewards


class HotpotQAToolSuccessReward(Reward):
    """Double-incentive: +1 iff the rollout used a tool AND F1 >= threshold.

    Purpose: shift the equilibrium between "guess short no tool" vs "use
    tool then answer" from monotone to DOUBLE-PEAKED:

        {tool + correct}  >  {no-tool + correct}
                          >  {no-tool + wrong}
                          ~  {tool + wrong}

    Without this bonus, ``HotpotQAToolUseReward`` alone teaches the policy
    to call tools blindly (every rollout pays the +1 regardless of whether
    the tool call helped).  Gating on F1 threshold makes tool-use rewarding
    only when it actually helps, so the policy learns *when* to use tools
    rather than *whether* to call them at all.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            used = any(
                m.get('role') == 'assistant' and (m.get('tool_calls') or [])
                for m in (traj.get('messages') or []))
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


class HotpotQAReasoningReward(Reward):
    """Reward pre-tool-call reasoning to break the naked-tool-call attractor.

    For each assistant message that contains a ``<tool_call>`` block,
    measure the length of the text BEFORE the tag (the reasoning
    preamble).  Score ramps linearly from 0 at ``REASONING_FLOOR_CHARS``
    to 1.0 at ``REASONING_MIN_CHARS`` and plateaus beyond.  Rollouts
    without any tool call receive 0 (neutral, not penalised).  The
    best-scored tool-call turn in the trajectory wins (``max`` over
    turns), so later refinement calls with thinner preambles do not
    dilute the signal from a well-reasoned first call.

    Background: prior Round-4/5 rollout_trace showed pre-tool reasoning
    collapsing from ~550 chars (EARLY) to 0 chars (END); the model had
    learned naked ``<tool_call>{"blocks":[...]}</tool_call>`` because
    reasoning tokens paid no reward under the old scheme and cost a tiny
    ``length_pen`` hit.  Without reasoning, block selection degenerates
    to a single-forward-pass keyword match -- F1 ceiling ~0.75.  This
    reward is the direct counter-signal: "think before you retrieve".
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        floor = float(REASONING_FLOOR_CHARS)
        ceil = float(max(REASONING_FLOOR_CHARS + 1, REASONING_MIN_CHARS))
        rewards = []
        for traj in trajectories:
            best = 0.0
            saw_tool_turn = False
            for msg in (traj.get('messages') or []):
                if msg.get('role') != 'assistant':
                    continue
                content = msg.get('content') or ''
                if not isinstance(content, str):
                    continue
                # vLLM's tool-call parser strips the ``<tool_call>...`` tag
                # out of the raw decoded text into the structured
                # ``msg['tool_calls']`` field, leaving ``content`` as
                # preamble-only.  So we must also accept assistant turns
                # that carry structured tool_calls (tag already parsed out).
                has_tc_tag = '<tool_call>' in content
                has_tc_struct = bool(msg.get('tool_calls'))
                if not (has_tc_tag or has_tc_struct):
                    continue
                saw_tool_turn = True
                # If the literal tag is still present, split at it; otherwise
                # the full ``content`` IS the preamble.
                if has_tc_tag:
                    pre = content.split('<tool_call>', 1)[0].strip()
                else:
                    pre = content.strip()
                n = len(pre)
                if n <= floor:
                    score = 0.0
                elif n >= ceil:
                    score = 1.0
                else:
                    score = (n - floor) / (ceil - floor)
                if score > best:
                    best = score
            # No tool turn at all -> 0 (neutral): rollouts that answer
            # the question directly without a tool call are legitimately
            # rewarded by f1 alone and should not be punished for the
            # absence of a reasoning-before-tool-call slot.
            rewards.append(best if saw_tool_turn else 0.0)
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
) -> Tuple[List[float], List[float], List[float], List[float],
           List[float], List[float], List[float], List[float]]:
    """Weighted sum of 7 component rewards.

    Returns ``(total, fmt, f1, extract, length_pen, tool_use,
    tool_success, reasoning)``.  ``total`` uses the ``*_REWARD_WEIGHT``
    env knobs; components are returned unweighted so SwanLab can log the
    raw signal for diagnosis.

    Rationale for the 7-component shape (see class docstrings for detail):

    * ``f1``           -- primary task signal (gold-answer token overlap).
    * ``format``       -- gated by ``FORMAT_MAX_CHARS`` to kill the "pad
                          garbage around \\boxed{}" exploit.
    * ``extract``      -- retained as diagnostic at weight 0 (its +0.99
                          plateau collapses GRPO variance; see ToolUse).
    * ``length_pen``   -- soft fence against the Phase-2 repetition loop.
    * ``tool_use``     -- binary, injects cross-rollout variance on the
                          tool-calling axis when ``extract`` cannot.
    * ``tool_success`` -- bi-modal shaping: tool-use only pays off when
                          the answer is actually correct.
    * ``reasoning``    -- Round-6 addition: reward pre-``<tool_call>``
                          reasoning length so the policy does not
                          collapse to naked tool calls; ramps 0 -> 1
                          over [FLOOR, MIN] chars of preamble.
    """
    global _F1_REWARD, _FORMAT_REWARD, _EXTRACT_REWARD
    global _LENGTH_PENALTY, _TOOL_USE_REWARD, _TOOL_SUCCESS_REWARD
    global _REASONING_REWARD
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _FORMAT_REWARD = HotpotQAFormatReward()
        _EXTRACT_REWARD = ExtractReward(midpoint=3.0, steepness=1.5)
        _LENGTH_PENALTY = HotpotQALengthPenalty()
        _TOOL_USE_REWARD = HotpotQAToolUseReward()
        _TOOL_SUCCESS_REWARD = HotpotQAToolSuccessReward()
        _REASONING_REWARD = HotpotQAReasoningReward()
    accuracy = _F1_REWARD(trajectories)
    fmt = _FORMAT_REWARD(trajectories)
    extract = _EXTRACT_REWARD(trajectories)
    length_pen = _LENGTH_PENALTY(trajectories)
    tool_use = _TOOL_USE_REWARD(trajectories)
    tool_success = _TOOL_SUCCESS_REWARD(trajectories)
    reasoning = _REASONING_REWARD(trajectories)
    total = [
        F1_REWARD_WEIGHT * a
        + FORMAT_REWARD_WEIGHT * f
        + EXTRACT_REWARD_WEIGHT * e
        + LENGTH_PENALTY_WEIGHT * lp
        + _curriculum_tool_use_weight() * tu
        + TOOL_SUCCESS_REWARD_WEIGHT * ts
        + REASONING_REWARD_WEIGHT * rs
        for a, f, e, lp, tu, ts, rs in zip(
            accuracy, fmt, extract, length_pen, tool_use, tool_success, reasoning)
    ]
    return (total, fmt, accuracy, extract, length_pen, tool_use,
            tool_success, reasoning)


# ========== Agentic rollout ==========
# Trajectory-level media fields that may ride on a prompt.  Defined once so
# ``_Rollout.__init__`` and ``_FrozenContext.freeze_delta`` agree on the set.
_MEDIA_KEYS = ('images', 'videos', 'audios')


class _FrozenContext:
    """Per-rollout monotone-append cache for chunked + indexed context.

    Multi-turn rollouts re-render the *entire* trajectory at each turn, which
    naively would run the chunker + index generator over the whole history
    every time -- re-processing already-indexed content, silently drifting
    ``<block_N>`` numbering, and potentially shredding the
    ``<block_N>...</block_N>`` tag pair across sentence splits.

    Solution: **freeze-and-append**.  Each call to :meth:`freeze_delta` only
    chunks and indexes the messages NEW since the last freeze, appending the
    results to :attr:`full_chunks` and :attr:`compressed_chunks`.
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
                 'compressed_chunks', 'media_frozen')

    def __init__(self) -> None:
        self.frozen_msg_count: int = 0
        self.full_chunks: List[Chunk] = []
        self.compressed_chunks: List[Chunk] = []
        self.media_frozen: bool = False

    def freeze_delta(self, trajectory: Dict[str, Any],
                     chunker: NativeChunker) -> None:
        """Chunk + structured-index only unfrozen messages/media; append to cache."""
        total_msgs = trajectory['messages']
        new_msgs = total_msgs[self.frozen_msg_count:]
        needs_media = (not self.media_frozen and
                        any(trajectory.get(k) for k in _MEDIA_KEYS))
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
        """Emit the accumulated indexed state as a sampler-ready dict."""
        return Chunks(chunks=list(self.compressed_chunks)).to_trajectory()

    def render_full(self) -> Chunks:
        """Emit the accumulated pre-index chunks for ExtractCompressed."""
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


def _dump_random_rollout_trace(
    turn: int,
    active: List['_Rollout'],
    displays: List[Dict[str, Any]],
    responses: List[Any],
) -> None:
    """Append ONE randomly-picked active rollout's post-turn state to JSONL.

    Fields captured:
      * ``compressed``    -- sampler-ready display (what the model saw
        this turn, AFTER Plan-B compression).
      * ``full_chunks``   -- pre-compression chunk list, so you can diff
        what was dropped.
      * ``tool_call_count`` / ``done`` / ``rewards`` -- cumulative state.
        Non-terminal turns naturally have ``f1=0`` / ``format=0``; the
        delta turn->turn+1 is the interesting signal, not absolute value.
      * ``last_decoded``  -- raw decoded text this turn (before cleaning).
      * ``final_answer``  -- ``\\boxed{...}`` extraction if rollout
        terminated on this turn, else ``''``.

    Must never crash training: the whole body is wrapped in best-effort
    exception handling.
    """
    if not _ROLLOUT_TRACE_PATH or not active:
        return
    try:
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

        # Per-component reward snapshot for the picked rollout.  Computed
        # mid-trajectory (rather than waiting for the post-batch
        # ``compute_rewards``) so we can line up "compression quality"
        # against "reward collected so far".  Non-terminal turns will
        # typically have f1=0 / format=0 (no ``\boxed{}`` yet) and
        # extract=logistic(tool_call_count) -- that's intentional: we
        # want to see the reward BUILD UP across turns, not only at the end.
        reward_snapshot: Dict[str, Any] = {}
        try:
            global _F1_REWARD, _FORMAT_REWARD, _EXTRACT_REWARD
            global _LENGTH_PENALTY, _TOOL_USE_REWARD, _TOOL_SUCCESS_REWARD
            global _REASONING_REWARD
            if _F1_REWARD is None:
                _F1_REWARD = HotpotQAF1Reward()
                _FORMAT_REWARD = HotpotQAFormatReward()
                _EXTRACT_REWARD = ExtractReward(midpoint=3.0, steepness=1.5)
                _LENGTH_PENALTY = HotpotQALengthPenalty()
                _TOOL_USE_REWARD = HotpotQAToolUseReward()
                _TOOL_SUCCESS_REWARD = HotpotQAToolSuccessReward()
                _REASONING_REWARD = HotpotQAReasoningReward()
            traj_list = [r.trajectory]
            f1_val = _F1_REWARD(traj_list)[0]
            fmt_val = _FORMAT_REWARD(traj_list)[0]
            ext_val = _EXTRACT_REWARD(traj_list)[0]
            len_val = _LENGTH_PENALTY(traj_list)[0]
            tu_val = _TOOL_USE_REWARD(traj_list)[0]
            ts_val = _TOOL_SUCCESS_REWARD(traj_list)[0]
            rs_val = _REASONING_REWARD(traj_list)[0]
            total_val = (
                F1_REWARD_WEIGHT * f1_val
                + FORMAT_REWARD_WEIGHT * fmt_val
                + EXTRACT_REWARD_WEIGHT * ext_val
                + LENGTH_PENALTY_WEIGHT * len_val
                + _curriculum_tool_use_weight() * tu_val
                + TOOL_SUCCESS_REWARD_WEIGHT * ts_val
                + REASONING_REWARD_WEIGHT * rs_val)
            reward_snapshot = {
                'f1': float(f1_val),
                'format': float(fmt_val),
                'extract': float(ext_val),
                'length_pen': float(len_val),
                'tool_use': float(tu_val),
                'tool_success': float(ts_val),
                'reasoning': float(rs_val),
                'total': float(total_val),
            }
        except Exception as e:  # pragma: no cover -- tracing must never crash
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
            'compressed': displays[idx] if idx < len(displays) else None,
            'full_chunks': full_chunks_preview,
            'last_decoded': last_decoded,
            'final_answer': final_answer,
        }
        with open(_ROLLOUT_TRACE_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    except Exception as e:  # pragma: no cover -- tracing must never crash
        logger.warning('rollout trace dump failed: %s', e)


def run_agentic_rollouts(
    prompts: List[Dict[str, Any]],
    sampler: vLLMSampler,
    sampling_params: SamplingParams,
    chunker: NativeChunker,
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
            r.frozen.freeze_delta(r.trajectory, chunker)
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
        # prompt + decoded response + cumulative reward snapshot AFTER
        # processing this turn's tool-call / terminal-answer handling, so
        # the ``done`` + ``final_answer`` fields reflect the turn's outcome.
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
    # Initialise SwanLab INSIDE main() rather than at module top level,
    # so only the driver process creates a run.  With Ray-based
    # ``twinkle.initialize``, every worker subprocess re-imports this
    # module on spawn; a top-level ``swanlab.init`` would therefore fire
    # once per worker (driver + NUM_GPUS workers = 1 + 8 empty runs),
    # spamming the cloud project with blank runs that never receive any
    # ``metrics.accumulate`` calls (those only happen on the driver).
    swanlab.init(project='twinkle')

    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
    ]

    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # Truncate the rollout-trace file so each run starts with a clean
    # append-only log.  Prior runs' traces live in swanlog/; keeping the
    # file open here would mix multiple runs into a single JSONL and make
    # post-mortem grep much harder.
    if _ROLLOUT_TRACE_PATH:
        try:
            open(_ROLLOUT_TRACE_PATH, 'w').close()
        except OSError as e:
            logger.warning('failed to truncate %s: %s', _ROLLOUT_TRACE_PATH, e)

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

    # Chunker lives on the driver (pure-Python, no GPU).
    # ``passage_boundary_re=r'^\[\d+\]\s+'`` makes each numbered HotpotQA
    # passage its own chunk, so ``<block_N>`` index == passage index ``[N]``
    # and ``ExtractCompressed(N)`` returns exactly that passage's original
    # text (not a mixed blob of 3-4 passages that happened to greedy-pack
    # into the same window).
    chunker = NativeChunker(
        model_id=MODEL_ID, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        passage_boundary_re=r'^\[\d+\]\s+')

    GLOBAL_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    dataloader = DataLoader(
        dataset=lambda: _prebuilt_dataset,
        batch_size=GLOBAL_BATCH_SIZE,
        min_batch_size=GLOBAL_BATCH_SIZE,
    )

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()
    # NOTE: ``sampling_params`` is rebuilt inside the training loop because
    # ``_curriculum_temperature()`` schedules a higher temperature during
    # the cold-start window (Plan A).  The initial instance here is only a
    # placeholder used before the first ``optim_step`` advance.
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=SAMPLING_TEMP_STABLE, top_p=0.95,
        # Stop after a tool_call so we can dispatch before the model keeps rambling.
        stop=['</tool_call>'],
    )

    optim_step = 0
    logger.info('Starting HotpotQA GRPO training (agentic + context compression)')
    logger.info(get_device_placement())

    for batch in dataloader:
        if optim_step >= MAX_STEPS:
            break

        # ---- Cold-start curriculum: Plans A + B ----
        # Write ``_CURRENT_STEP`` BEFORE any ``compute_rewards`` call so the
        # curriculum accessors see the correct phase.  Then rebuild
        # ``sampling_params`` with the step-dependent temperature.
        global _CURRENT_STEP
        _CURRENT_STEP = optim_step
        sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
            temperature=_curriculum_temperature(), top_p=0.95,
            stop=['</tool_call>'],
        )

        metrics.reset()
        expand_prompts: List[Dict[str, Any]] = []
        for prompt in batch:
            expand_prompts.extend([prompt] * NUM_GENERATIONS)

        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()

        # ---- Multi-turn rollout (the agentic heart of this script) ----
        rollouts = run_agentic_rollouts(
            expand_prompts, sampler, sampling_params,
            chunker, max_turns=MAX_TURNS,
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
         length_pen_rewards, tool_use_rewards,
         tool_success_rewards, reasoning_rewards) = compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': brevity_rewards,
                'f1': accuracy_rewards,
                'extract': extract_rewards,
                'length_pen': length_pen_rewards,
                'tool_use': tool_use_rewards,
                'tool_success': tool_success_rewards,
                'reasoning': reasoning_rewards,
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
        # Curriculum visibility: surface the active temperature / tool_use
        # weight so the SwanLab chart shows exactly where the schedule
        # transitions and how it lines up against the reward curves.
        log_dict['curriculum_temperature'] = _curriculum_temperature()
        log_dict['curriculum_tool_use_weight'] = _curriculum_tool_use_weight()
        swanlab.log(log_dict)
        metrics.reset()
        logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('hotpotqa-grpo-tools-final')


if __name__ == '__main__':
    main()
