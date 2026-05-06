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

Rewards are computed on the *full* multi-turn trajectory, but the reward
signal is intentionally minimal (see ``Reward weights`` block): only F1
on the boxed answer (strict: no boxed -> f1=0), a length penalty against
repetition, and a light negative for empty / compression-mimic
terminals.  Tool use is NOT directly rewarded -- the policy learns when
to use it via F1 alone through GRPO's group-relative advantage.

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
from twinkle_agentic.data_format.chunk import Chunk, Chunks
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
# ``NUM_EPOCHS`` is the primary training-horizon knob.  The 340-question
# HotpotQA whitelist finishes one epoch in ~42 optim steps at
# BATCH_SIZE=8.  Total steps (used to size the LR cosine schedule) is
# derived as ``NUM_EPOCHS * steps_per_epoch`` after the dataset is built
# (see ``total_steps`` in ``main``).  ``MAX_STEPS`` is kept only as an
# optional hard cap: when unset (``0``) the scheduler horizon matches
# the actual run length and LR does not stay pinned near peak for the
# whole run.  Set ``MAX_STEPS>0`` only for smoke tests / short debug runs.
NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 5))
MAX_STEPS = int(os.environ.get('MAX_STEPS', 0))  # 0 = auto-derive from NUM_EPOCHS
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 8))
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2))
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))
ADAPTER_NAME = 'default'
SAVE_STEPS = int(os.environ.get('SAVE_STEPS', 1000))
LORA_RANK = int(os.environ.get('LORA_RANK', 16))

# ---- Agentic rollout knobs ----
# Hard cap on tool-call turns per rollout.  Multi-hop HotpotQA sometimes
# needs two successive extractions ("find bridge entity, then extract its
# article"); 6 leaves headroom for that without letting a stuck rollout
# spin indefinitely.  This is an UPPER bound -- the policy is free to
# terminate earlier.  Round-8 reward shape makes "no tool + direct answer"
# and "tool + answer" equivalent under F1 alone, so turn budget is not a
# driver of tool-use behaviour, just a safety fuse.
MAX_TURNS = int(os.environ.get('MAX_TURNS', 6))            # hard cap on tool-call turns
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', 1024))        # chars per chunk (NativeChunker)
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', 0))    # sliding-window overlap

# ---- HotpotQA dataset encode knobs ----
HOTPOTQA_NUM_PROC = int(os.environ.get('HOTPOTQA_NUM_PROC', 16))
HOTPOTQA_MAX_LENGTH = int(os.environ.get('HOTPOTQA_MAX_LENGTH', 64000))

# ---- Reward weights (Round-8: minimal three-signal shape) ----
# Design history in three lines: rounds 1-7 accumulated eight overlapping
# reward components (f1, format, tool_use, tool_success, reasoning,
# extract, length_pen, answer_commit) -- each added to patch a hack the
# previous combination opened up.  Trace analysis at step 750 showed the
# policy had found a new attractor -- "call tool + emit bare entity
# token" -- that scored f1=0.5 via the last-line fallback, bypassed
# format_reward (weight 0), and slipped under answer_commit's <80-char
# exemption.  Rather than add a ninth patch, Round-8 collapses the whole
# stack to three orthogonal signals:
#
#   f1              -- single source of truth for correctness.  Made
#                      self-enforcing by tightening _extract_final_answer
#                      to require a \boxed{} span (no fallback).  This
#                      alone kills bare-answer hacking: no \boxed{}
#                      means f1=0, period.
#   length_pen      -- anti-repetition fence; zero unless text overflows
#                      ANSWER_TOO_LONG_CHARS.
#   answer_commit   -- light negative for edge cases f1 cannot see:
#                      empty final messages and compression-schema
#                      mimicry.  Scale ~= one full F1 point so an empty
#                      turn costs the same as a wrong answer.
#
# Explicitly DELETED (and why):
#   format_reward       -- redundant with strict f1; same information.
#   tool_use_reward     -- created a +0.4 floor that made "always call
#                          tool" a dominant GRPO strategy regardless of
#                          outcome.  Let f1 alone drive tool policy.
#   tool_success_reward -- redundant with f1 once tool_use is gone:
#                          "tool + correct" already wins groups via f1.
#   reasoning_reward    -- 100% collinear with tool_use in trace data
#                          (zero mismatches in 84 records).
#   extract_reward      -- reverse-sigmoid collapsed to ~0.99 constant
#                          and contributed no variance to GRPO.
F1_REWARD_WEIGHT = float(os.environ.get('F1_REWARD_WEIGHT', 1.5))
LENGTH_PENALTY_WEIGHT = float(os.environ.get('LENGTH_PENALTY_WEIGHT', 0.3))
ANSWER_COMMIT_PENALTY_WEIGHT = float(os.environ.get(
    'ANSWER_COMMIT_PENALTY_WEIGHT', 1.0))
# Terminal-message length above which length_pen ramps up linearly.
# Legitimate retrieve+reason+answer rollouts run 2000-4000 chars; only
# real repetition loops (>5k) should hit the penalty.
ANSWER_TOO_LONG_CHARS = int(os.environ.get('ANSWER_TOO_LONG_CHARS', 5000))

# ---- Rollout trace dump (post-mortem diagnosis) ----
# Append-only JSONL: one line per turn per run, carrying the compressed
# prompt, pre-compression chunks, decoded completion, cumulative tool
# count, and per-component reward snapshot for ONE randomly picked active
# rollout.  Empty string disables.  File is truncated at main() start.
_ROLLOUT_TRACE_PATH = os.environ.get(
    'ROLLOUT_TRACE_PATH', 'rollout_trace.jsonl')

# ========== System Prompt ==========
# Round-8 note: prompt must advertise the EXACT reward contract, which
# in this round is simply "answer must be inside \boxed{}".  There is
# no separate reward for reasoning length, tool use, or calling the
# tool a certain number of times -- the only thing that pays is a
# correct boxed answer.  Tool use is available but optional.
SYSTEM_PROMPT = (
    'You are a careful multi-hop QA assistant. Answer the user\'s question '
    'using the provided paragraphs. Put your FINAL answer inside \\boxed{} '
    '(e.g. ``\\boxed{Delhi}``).  Answers not inside \\boxed{} will not be '
    'scored.  Keep the boxed text short: a name, entity, date, or '
    '"yes"/"no". Do not include extra words in the box.\n\n'
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
    'You may call ``extract_compressed`` again in a later turn if the '
    'first recall did not contain the answer.')

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


def _extract_chunk_title(chunk: Chunk) -> str:
    """Return the title entity of a passage chunk, or ``''`` if none.

    HotpotQA passages arrive shaped ``[N] Title: body...``.  We accept
    only titles that contain at least one proper noun so garbage like
    ``"[3] 1: 2"`` (numeric leadings) never pollutes the cross-ref map.
    """
    content = chunk.get('content', '')
    if not isinstance(content, str):
        return ''
    m = re.match(r'^\[?\d+\]?\s*([^:]+?):\s', content)
    if not m:
        return ''
    title = m.group(1).strip()
    return title if _PROPER_NOUN_RE.search(title) else ''


def _annotate_cross_refs(
    keywords: List[str], title_map: Dict[str, int], self_idx: int,
) -> List[str]:
    """Annotate keywords whose text matches another block's title.

    Exact match first (``"Kansas State Wildcats" -> block 12``), then a
    substring fallback so partial references (``"Kansas State"`` vs the
    full title ``"Kansas State Wildcats"``) still resolve.  Keywords
    shorter than 4 chars skip the fallback to avoid spurious hits.
    """
    if not title_map:
        return keywords
    annotated: List[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        target = title_map.get(kw_lower)
        if target is not None and target != self_idx:
            annotated.append(f'{kw} (block {target})')
            continue
        if len(kw) >= 4:
            for title_lower, tidx in title_map.items():
                if tidx == self_idx:
                    continue
                if kw_lower in title_lower or title_lower in kw_lower:
                    annotated.append(f'{kw} (block {tidx})')
                    break
            else:
                annotated.append(kw)
        else:
            annotated.append(kw)
    return annotated


def _generate_passage_index(
    chunk: Chunk, idx: int = 0,
    title_map: Optional[Dict[str, int]] = None,
) -> Chunk:
    """Build a summary-index chunk from a text passage chunk.

    Output content layout (wrapped by :meth:`Chunks.to_trajectory` as
    ``<block_N>...</block_N>`` for :class:`ExtractCompressed` recall)::

        <first sentence>[ <relational sentence>] (Related: k1, k2, ...)

    Why this shape:

    * ``first sentence`` is always kept -- carries the passage's topic.
    * ``relational sentence`` (optional) is the non-first sentence with
      highest entity density (years + proper nouns + numbers); skipped if
      <2 entities or >=70% word overlap with the first.  Without it,
      2-hop HotpotQA joins cannot be resolved from the index alone and
      every rollout must call ``extract_compressed``.
    * ``(Related: ...)`` lists concrete keywords mined from the hidden
      body -- years, multi-word proper nouns, numbers -- capped at 12.
      When a keyword matches another passage's title, it is annotated as
      ``Entity (block N)`` to give the model a direct bridge pointer.

    ``title_map`` / ``idx`` are optional: when absent, cross-block
    pointers are skipped (useful for standalone testing).
    """
    content = chunk.get('content', '')
    if not isinstance(content, str) or not content.strip():
        return chunk
    if chunk.get('role') in ('system', 'tool'):
        return chunk
    if chunk.get('type') in _PROTECTED_TYPES:
        return chunk
    raw = chunk.get('raw')
    if isinstance(raw, dict) and raw.get('kind') in _PROTECTED_KINDS:
        return chunk

    first_sent, rest = _split_first_sentence(content)
    if not rest:
        # Nothing to hide; leave verbatim and do not mark condensed.
        return chunk

    all_sents = _split_all_sentences(content)
    relational = _pick_relational_sentence(all_sents, first_sent)

    hidden_keywords = _extract_hidden_keywords(first_sent, rest)
    if title_map:
        hidden_keywords = _annotate_cross_refs(hidden_keywords, title_map, idx)

    suffix = f' (Related: {", ".join(hidden_keywords)})' if hidden_keywords else ''
    if relational:
        # Single-space join keeps the block shaped like one prose paragraph,
        # matching the base Qwen3.5-4B pretraining distribution.
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
    """Apply passage-index compression with cross-block entity pointers.

    Two-pass: (1) build ``{title_lowercase -> block_idx}`` for all content
    passages; (2) compress each passage, annotating keywords that
    reference other blocks with a natural ``(block N)`` parenthetical.
    """
    title_map: Dict[str, int] = {}
    for i, c in enumerate(chunks.chunks):
        title = _extract_chunk_title(c)
        if title:
            title_map[title.lower()] = i
    return Chunks(chunks=[
        _generate_passage_index(c, i, title_map)
        for i, c in enumerate(chunks.chunks)
    ])


def _extract_final_answer(completion: str) -> str:
    """Pull the predicted answer out of a completion.

    STRICT contract: returns the content of the last ``\\boxed{...}`` span,
    or ``''`` if no boxed answer is present.

    Rationale: earlier revisions fell back to "last non-empty line" so
    partially-formatted completions still received a graded F1.  Rollout
    traces then showed the policy exploiting that fallback by emitting
    bare entity tokens (e.g. ``'19 mi<|im_end|>'``) which scored
    ``f1=0.5`` without ever boxing.  Making F1 itself enforce the boxing
    contract is the single-point guarantee the rest of the reward stack
    can build on -- it is the ONLY place the boxing rule needs to live.
    """
    matches = _BOXED_RE.findall(completion or '')
    if matches:
        return matches[-1].strip()
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
# Module-level singletons lazily instantiated by ``compute_rewards`` on
# first call.  Round-8 minimal shape: only three reward objects exist.
_F1_REWARD = None
_LENGTH_PENALTY = None
_ANSWER_COMMIT_PENALTY = None


# Filler modifier / descriptor tokens that should not penalise F1 when
# they are the only reason a prediction and gold differ.  HotpotQA gold
# phrases frequently tack on a unit descriptor (``6.213 km long``,
# ``12 years old``, ``150 m tall``) or a leading year qualifier
# (``1963 Pan American Games``) that the policy either omits from or
# adds to its own shorter answer.  Standard SQuAD F1 treats every such
# extra / missing token as a mismatch, driving a sizeable slice of
# round-9 rollouts (class B + class D in the epoch audit, ~190 samples
# = 7 % of the batch) to sub-0.9 f1 despite being factually correct.
# Only tokens with no independent factual content are listed here;
# genuine nouns like ``County`` (``Ulster`` vs ``Ulster County``) must
# NOT be included -- those are class A partial matches where standard
# partial-credit F1 still correctly pressures the model to complete the
# entity phrase.
_FILLER_TOKENS: frozenset = frozenset([
    'long', 'tall', 'high', 'wide', 'deep', 'heavy', 'old', 'large',
    'small', 'big', 'short', 'away', 'ago', 'approximately', 'about',
    'around', 'over', 'under', 'below', 'above', 'total', 'roughly',
    'nearly', 'almost', 'exactly',
])

# Porter stemmer -- normalises surface inflections (``plant`` ↔
# ``plants``, ``reveal`` ↔ ``revealed``, ``city`` ↔ ``cities``) so the
# F1 reward stops penalising singular/plural and verb-tense noise.
# NLTK is a HARD requirement: we refuse to fall back to identity
# stemming because that silently reintroduces the false-negative class
# the refactor was designed to close.  Install via ``pip install nltk``.
# Stemming is skipped for short (<4 chars) or non-alpha tokens so that
# ``US`` / ``170`` / ``km`` survive intact.
try:
    from nltk.stem import PorterStemmer as _PorterStemmer
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        'HotpotQAF1Reward requires nltk for Porter stemming. '
        'Install it with `pip install nltk`.'
    ) from _e
_STEMMER = _PorterStemmer()


def _stem(tok: str) -> str:
    return _STEMMER.stem(tok) if len(tok) >= 4 and tok.isalpha() else tok


def _normalize_answer(s: str) -> str:
    """SQuAD-style normalisation + Porter stemming.

    Pipeline: lowercase → drop punctuation → drop articles (a/an/the) →
    collapse whitespace → Porter-stem each alpha token of length ≥ 4.
    The stemmer step closes the ``plant`` / ``plants`` and ``revealed``
    / ``reveals`` false-negative classes observed in the epoch audit
    without inflating single-char / numeric tokens.
    """
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(_stem(t) for t in s.split())


def _f1_score(prediction: str, gold: str) -> Tuple[float, float]:
    """Word-level F1 and EM with two HotpotQA-motivated short-circuits.

    Standard SQuAD-style F1 is computed first; on top of that two
    conservative short-circuits flip the reward to ``1.0`` for
    factually-equivalent phrasings that would otherwise be penalised:

    * **Over-answer (``gold ⊂ pred``)** — the prediction contains every
      gold token plus extras that are either numeric (year qualifier)
      or in :data:`_FILLER_TOKENS`.  Example: gold ``Pan American
      Games``, pred ``1963 Pan American Games``.  Without this the
      metric actively discourages the model from providing precise
      qualifiers.
    * **Under-answer with filler tail (``pred ⊂ gold``)** — every
      missing gold token is in :data:`_FILLER_TOKENS`.  Example: gold
      ``6.213 km long``, pred ``6.213 km``.  The content nouns and
      numerics are fully covered; only a descriptor tail is missing.

    Both short-circuits are token-set based to avoid duplicate-token
    inflation and deliberately refuse to match when any content noun
    is missing, so class A partials (``Ulster`` vs ``Ulster County``)
    keep their 0.67 gradient and still incentivise the full entity.
    """
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

    pred_set = set(pred_tokens)
    gold_set = set(gold_tokens)
    # Over-answer: gold ⊆ pred, extras are year / filler only.
    if gold_set < pred_set:
        extras = pred_set - gold_set
        if all(t.isdigit() or t in _FILLER_TOKENS for t in extras):
            return 1.0, em
    # Under-answer with filler tail: pred ⊆ gold, missing all filler.
    if pred_set < gold_set:
        missing = gold_set - pred_set
        if all(t in _FILLER_TOKENS for t in missing):
            return 1.0, em
    return f1, em


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


class HotpotQALengthPenalty(Reward):
    """Negative reward proportional to terminal-message length overflow.

    Returns 0 if the final assistant message is within
    :data:`ANSWER_TOO_LONG_CHARS`, else a linearly growing penalty that
    reaches -1 when the message fills ``MAX_NEW_TOKENS * 4`` chars (the
    empirical 4 chars/token ceiling).  Counteracts the repetition-loop
    exploit where the policy fills the generation budget with garbage.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        budget = max(1, MAX_NEW_TOKENS * 4 - ANSWER_TOO_LONG_CHARS)
        rewards = []
        for t in trajectories:
            text = _last_assistant_text(t) or ''
            overflow = max(0, len(text) - ANSWER_TOO_LONG_CHARS)
            rewards.append(-min(1.0, overflow / budget))
        return rewards


class HotpotQAAnswerCommitPenalty(Reward):
    """Negative signal for terminals that F1 cannot score.

    Round-8 scope: F1 now enforces the \\boxed{} contract itself (no
    fallback), so the only remaining edge cases this penalty addresses
    are patterns where the model degenerated into producing NO usable
    final turn at all:

    * ``-1.0`` -- final assistant text empty or <5 visible chars.
                  Typically the "empty post-tool turn" hack: tool is
                  called, result comes back, model emits ``<|im_end|>``.
    * ``-0.5`` -- terminal ends with a ``(Related: ...)</block_N>``
                  tag.  The model has reflected the condenser's schema
                  back into its own output -- syntactically a non-answer.
    * ``0.0``  -- anything else (including missing ``\\boxed{}``, which
                  is already handled by F1=0).  No short-line exemption,
                  no ``Answer:`` marker -- those loopholes are closed by
                  making F1 strict.
    """

    _COMPRESSION_TAG_RE = re.compile(
        r'\(\s*Related\s*:[^)]*\)\s*</?block_\d+>?\s*$', re.IGNORECASE)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            final = _last_assistant_text(t) or ''
            stripped = final.strip()
            if len(stripped) < 5:
                rewards.append(-1.0)
                continue
            if self._COMPRESSION_TAG_RE.search(stripped):
                rewards.append(-0.5)
                continue
            rewards.append(0.0)
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
        # ``preprocess`` returns ``None`` for rows that fail the hard-level
        # filter (see :meth:`preprocess`).  Drop them before ``map_row_to_col``
        # so the column-major conversion never sees a ``None`` entry.
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
                # HotpotQA sentences sometimes carry leading / trailing spaces
                # and sometimes don't.  Normalise: strip each, then join with
                # a single space.  Joining with ``''`` (the previous code)
                # produced ``"Cat.Dog."`` when source sentences lacked a
                # trailing space, which breaks ``_SENTENCE_END_RE`` (needs
                # whitespace after ``.``) and silently disables compression
                # for that passage -- the chunk would fall through the
                # ``not rest`` branch of ``_split_first_sentence`` and be
                # returned verbatim.
                body = ' '.join(s.strip() for s in sents if s and s.strip())
            else:
                body = str(sents).strip()
            lines.append(f'[{i}] {title}: {body}')
        # Join passages with a BLANK line (``\n\n``) rather than a single
        # ``\n``.  :class:`NativeChunker._split_preserving_code` uses
        # ``_PARAGRAPH_RE = r'\n\s*\n'`` as the paragraph boundary, so only a
        # blank line prevents adjacent passages from being fused into one
        # "mega-paragraph".
        return '\n\n'.join(lines)

    def preprocess(self, row: Dict[str, Any]) -> Optional[Trajectory]:
        # ── Hard-only filter ──────────────────────────────────────────
        # HotpotQA ``level`` takes values ``easy`` / ``medium`` / ``hard``.
        # ``hard`` rows are the ones whose answer actually requires 2-hop
        # reasoning across ≥2 supporting passages; ``easy`` / ``medium``
        # frequently collapse to a single-passage lookup or yes/no, which
        # (a) lets the policy win F1 without ever using the compression
        # index, defeating the A/B contrast we care about, and (b)
        # over-represents yes/no answers that the "short-and-lucky"
        # collapse mode exploits.  Returning ``None`` here signals
        # :meth:`__call__` to drop the row entirely.
        #
        # An optional id whitelist (``WRONG_IDS_FILE``) is applied *before*
        # :meth:`__call__` reaches us -- see :func:`create_hotpotqa_dataset`,
        # which calls ``dataset.filter`` at the HF columnar layer so the
        # filtered dataset is dense (no empty batches) by the time ``map``
        # runs.  Filtering sparsely inside ``preprocess`` would produce
        # batches whose output-column dict is ``{}`` after all rows are
        # dropped, and pyarrow rejects that with ``Schema and number of
        # arrays unequal``.
        if (row.get('level') or '').strip().lower() != 'hard':
            return None

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

    # ── Optional id whitelist ────────────────────────────────────────
    # When ``WRONG_IDS_FILE`` points at a file (typically produced by
    # :mod:`cookbook.rl.extract_wrong_ids`), restrict training to exactly
    # those ids.  We filter at the HF columnar layer via
    # ``datasets.Dataset.filter`` -- this keeps the underlying Arrow
    # table dense (schema + row count stay in lock-step), so the
    # downstream ``dataset.map`` never sees an empty batch.  Doing the
    # same filtering inside :meth:`HotpotQAProcessor.preprocess` (which
    # returns ``None`` for filtered rows) would make most ``map`` batches
    # collapse to zero rows, and pyarrow would raise
    # ``Schema and number of arrays unequal`` once the first non-empty
    # batch has fixed the output schema.
    _wrong_ids_path = os.environ.get('WRONG_IDS_FILE', '').strip()
    if _wrong_ids_path:
        try:
            with open(_wrong_ids_path, 'r', encoding='utf-8') as fh:
                _ids = frozenset(ln.strip() for ln in fh if ln.strip())
        except OSError as exc:
            # Fail loud: silently training on 90k rows because of a typo
            # in the env-var path wastes hours.
            raise RuntimeError(
                f'WRONG_IDS_FILE={_wrong_ids_path!r} could not be read: '
                f'{exc}') from exc
        if _ids:
            _key = next(iter(dataset.datasets.keys()))
            _before = len(dataset.datasets[_key])
            dataset.datasets[_key] = dataset.datasets[_key].filter(
                lambda row: row.get('id') in _ids)
            dataset.dataset = dataset.datasets[_key]
            print(f'[WRONG_IDS_FILE] {_wrong_ids_path}: '
                  f'{_before} -> {len(dataset.dataset)} rows '
                  f'(whitelist size={len(_ids)})')

    dataset.set_template(
        'Qwen3_5Template', model_id=MODEL_ID, max_length=HOTPOTQA_MAX_LENGTH,
        truncation_strategy='delete', enable_thinking=False)
    # ``HotpotQAProcessor.preprocess`` filters to ``level == 'hard'`` only,
    # so the batch size returned by ``__call__`` is typically smaller than
    # the input batch (batched=True is enforced upstream).  HF
    # ``datasets.map`` pairs the filtered output columns against the
    # *untouched* original columns by index, which pyarrow rejects with
    # ``Column ... expected length N but got length K`` as soon as the
    # sizes disagree.  ``remove_columns`` drops the original HotpotQA
    # schema before stitching so the filtered size becomes authoritative.
    _HOTPOTQA_COLS = ['id', 'question', 'answer', 'type', 'level',
                      'supporting_facts', 'context']
    dataset.map(
        HotpotQAProcessor(system=SYSTEM_PROMPT),
        remove_columns=_HOTPOTQA_COLS,
    )
    dataset.encode(
        add_generation_prompt=True,
        load_from_cache_file=True,
        num_proc=HOTPOTQA_NUM_PROC,
    )
    return dataset


def compute_rewards(
    trajectories: List[Dict[str, Any]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Weighted sum of 3 minimal reward components.

    Returns ``(total, f1, length_pen, answer_commit)``.  ``total`` uses
    the ``*_WEIGHT`` constants at the top of the file; component signals
    are returned unweighted so SwanLab can log the raw values.

    Design: three orthogonal signals with no overlap.

    * ``f1``           -- primary task signal. Gated inside
                          :class:`HotpotQAF1Reward` by a strict
                          ``\\boxed{...}`` extractor -- no boxed, no
                          credit. This is the only positive task reward.
    * ``length_pen``   -- soft fence that turns negative past
                          ``ANSWER_TOO_LONG_CHARS`` to kill the Phase-2
                          repetition loop.
    * ``answer_commit``-- negative-only penalty when the final assistant
                          message fails to commit a real answer (empty
                          or compression-tag mimicry).
    """
    global _F1_REWARD, _LENGTH_PENALTY, _ANSWER_COMMIT_PENALTY
    if _F1_REWARD is None:
        _F1_REWARD = HotpotQAF1Reward()
        _LENGTH_PENALTY = HotpotQALengthPenalty()
        _ANSWER_COMMIT_PENALTY = HotpotQAAnswerCommitPenalty()
    f1 = _F1_REWARD(trajectories)
    length_pen = _LENGTH_PENALTY(trajectories)
    answer_commit = _ANSWER_COMMIT_PENALTY(trajectories)
    total = [
        F1_REWARD_WEIGHT * a
        + LENGTH_PENALTY_WEIGHT * lp
        + ANSWER_COMMIT_PENALTY_WEIGHT * ac
        for a, lp, ac in zip(f1, length_pen, answer_commit)
    ]
    return total, f1, length_pen, answer_commit


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
    """Append EVERY active rollout's post-turn state to the JSONL trace.

    Earlier revisions sampled one random active rollout per turn; that was
    cheap but made causal claims about tool use impossible (each trace
    record sat on a *different* prompt, so tcc>0 vs tcc=0 comparisons
    conflated treatment with question difficulty).  Dumping the full
    group (``NUM_GENERATIONS`` rollouts per prompt) restores the
    within-group contrast: all rollouts in a group share the same
    question, so post-hoc filtering by ``tool_call_count`` isolates the
    tool-use effect from question-level confounds.

    Per-record fields match the previous single-rollout schema plus a
    ``group_size`` field giving the active-set size at dump time, which
    lets post-mortem code re-group records by ``(turn, group_size)``
    without re-deriving it.  ``picked_idx`` is kept (pointing at the
    rollout's index inside ``active``) so individual records can still be
    correlated with the display/response slices they came from.

    Must never crash training: the whole body is wrapped in best-effort
    exception handling per record, so one bad rollout does not poison the
    rest of the group.
    """
    if not _ROLLOUT_TRACE_PATH or not active:
        return
    try:
        # One file handle, one sweep -- minimises fsync overhead even on
        # the largest groups (BATCH_SIZE * NUM_GENERATIONS = 64 records).
        records: List[str] = []
        global _F1_REWARD, _LENGTH_PENALTY, _ANSWER_COMMIT_PENALTY
        if _F1_REWARD is None:
            _F1_REWARD = HotpotQAF1Reward()
            _LENGTH_PENALTY = HotpotQALengthPenalty()
            _ANSWER_COMMIT_PENALTY = HotpotQAAnswerCommitPenalty()
        group_size = len(active)
        for idx, r in enumerate(active):
            try:
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
                    final_answer = _extract_final_answer(
                        _last_assistant_text(r.trajectory))

                # Per-component reward snapshot.  Non-terminal turns will
                # typically have f1=0 (no ``\boxed{}`` yet) -- that's
                # intentional: the signal of interest is the DELTA from
                # pre-answer to terminal turn, not the absolute value.
                reward_snapshot: Dict[str, Any]
                try:
                    traj_list = [r.trajectory]
                    f1_val = _F1_REWARD(traj_list)[0]
                    len_val = _LENGTH_PENALTY(traj_list)[0]
                    ac_val = _ANSWER_COMMIT_PENALTY(traj_list)[0]
                    total_val = (
                        F1_REWARD_WEIGHT * f1_val
                        + LENGTH_PENALTY_WEIGHT * len_val
                        + ANSWER_COMMIT_PENALTY_WEIGHT * ac_val)
                    reward_snapshot = {
                        'f1': float(f1_val),
                        'length_pen': float(len_val),
                        'answer_commit': float(ac_val),
                        'total': float(total_val),
                    }
                except Exception as e:  # pragma: no cover -- tracing must never crash
                    logger.warning(
                        'rollout trace reward snapshot failed: %s', e)
                    reward_snapshot = {'error': repr(e)}

                record = {
                    'ts': time.time(),
                    'turn': turn,
                    'active_size': group_size,
                    'group_size': group_size,
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
                records.append(
                    json.dumps(record, ensure_ascii=False, default=str))
            except Exception as e:  # pragma: no cover -- per-record safety net
                logger.warning(
                    'rollout trace record build failed (idx=%d): %s', idx, e)

        if records:
            with open(_ROLLOUT_TRACE_PATH, 'a', encoding='utf-8') as f:
                f.write('\n'.join(records) + '\n')
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

    # Derive the true training horizon from the dataset size and epoch budget.
    # ``DataLoader`` drops the tail batch (``min_batch_size == GLOBAL_BATCH_SIZE``)
    # so ``batches_per_epoch = len(dataset) // GLOBAL_BATCH_SIZE``.
    #
    # IMPORTANT: the main loop below performs MULTIPLE optimizer steps per
    # data batch.  Each batch expands into ``BATCH_SIZE * NUM_GENERATIONS``
    # rollouts, then the inner mini-batch loop slices those rollouts into
    # chunks of ``MINI_BATCH_SIZE`` and fires ``clip_grad_and_step`` +
    # ``optim_step += 1`` on EACH chunk.  So the real conversion is:
    #
    #     optim_steps_per_batch = BATCH_SIZE * NUM_GENERATIONS // MINI_BATCH_SIZE
    #     optim_steps_per_epoch = batches_per_epoch * optim_steps_per_batch
    #
    # Previous revisions set ``total_steps = NUM_EPOCHS * batches_per_epoch``,
    # which silently compressed the run by ``optim_steps_per_batch`` (8x under
    # default envs: NUM_EPOCHS=5 actually produced only ~0.63 epoch of data
    # coverage before the ``optim_step >= total_steps`` break fired).  The
    # corrected formula multiplies by ``optim_steps_per_batch`` so ``NUM_EPOCHS``
    # matches what the user requested, and the LR cosine schedule (sized
    # against ``total_steps``) decays over the full run instead of stopping
    # early while LR is still near peak.
    _global_batch_size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    batches_per_epoch = max(1, len(_prebuilt_dataset) // _global_batch_size)
    optim_steps_per_batch = max(
        1, (BATCH_SIZE * NUM_GENERATIONS + MINI_BATCH_SIZE - 1) // MINI_BATCH_SIZE)
    steps_per_epoch = batches_per_epoch * optim_steps_per_batch
    derived_total_steps = NUM_EPOCHS * steps_per_epoch
    if MAX_STEPS > 0:
        total_steps = min(MAX_STEPS, derived_total_steps)
    else:
        total_steps = derived_total_steps
    logger.info(
        'Training horizon: dataset=%d, global_batch=%d, batches_per_epoch=%d, '
        'optim_steps_per_batch=%d, optim_steps_per_epoch=%d, num_epochs=%d, '
        'derived_total_steps=%d, max_steps_cap=%s, effective_total_steps=%d',
        len(_prebuilt_dataset), _global_batch_size, batches_per_epoch,
        optim_steps_per_batch, steps_per_epoch,
        NUM_EPOCHS, derived_total_steps,
        MAX_STEPS if MAX_STEPS > 0 else 'auto',
        total_steps,
    )

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
    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1,
        temperature=1.0, top_p=0.95,
        # Stop after a tool_call so we can dispatch before the model keeps rambling.
        stop=['</tool_call>'],
    )

    optim_step = 0
    logger.info('Starting HotpotQA GRPO training (agentic + context compression)')
    logger.info(get_device_placement())

    # Re-enter the DataLoader up to ``NUM_EPOCHS`` times so the 340-item
    # whitelist is cycled.  The inner iterator exhausts after one pass;
    # without this wrapper, training would stop after a single epoch.
    # The break on ``optim_step >= total_steps`` caps the run so the LR
    # schedule (sized against ``total_steps``) and the epoch counter stay
    # consistent.
    def _epoch_cycle(dl, n_epochs):
        for ep in range(1, n_epochs + 1):
            logger.info(f'=== Epoch {ep}/{n_epochs} '
                        f'(optim_step={optim_step}/{total_steps}) ===')
            for batch in dl:
                yield batch

    for batch in _epoch_cycle(dataloader, NUM_EPOCHS):
        if optim_step >= total_steps:
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

        (total_rewards, f1_rewards, length_pen_rewards,
         answer_commit_rewards) = compute_rewards(all_trajectories)

        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'f1': f1_rewards,
                'length_pen': length_pen_rewards,
                'answer_commit': answer_commit_rewards,
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

            if optim_step >= total_steps:
                break
            if optim_step % SAVE_STEPS == 0:
                model.save(f'hotpotqa-grpo-tools-checkpoint-{optim_step}')

        log_dict = metrics.calculate()
        log_dict.update(model.calculate_metric(is_training=True))
        # Rollout depth is a useful diagnostic: if it collapses to 1 every step
        # the policy has stopped using tools entirely.
        if n_turns_per_rollout:
            log_dict['avg_turns'] = sum(n_turns_per_rollout) / len(n_turns_per_rollout)
        # Tool-use diagnostics.  Round-8 removed the tool_use reward, so the
        # policy is free to call / skip the tool purely under F1 pressure --
        # these two signals are how we read that behaviour off SwanLab.
        #   avg_tool_calls  -- mean ``extract_compressed`` calls per rollout
        #                      (across all turns of the rollout).  Falling to
        #                      0 = policy collapsed to "never use tool";
        #                      spiking upward = stuck in tool-calling loop.
        #   tool_use_rate   -- fraction of rollouts that made >=1 tool call.
        #                      Complements avg_tool_calls: you can have
        #                      avg_tool_calls=0.5 from everyone calling once
        #                      half the time, OR from half the rollouts
        #                      calling twice -- only rate disambiguates.
        if all_trajectories:
            tool_call_counts = [
                sum(len(m.get('tool_calls') or [])
                    for m in t.get('messages', [])
                    if m.get('role') == 'assistant')
                for t in all_trajectories
            ]
            log_dict['avg_tool_calls'] = (
                sum(tool_call_counts) / len(tool_call_counts))
            log_dict['tool_use_rate'] = (
                sum(1 for c in tool_call_counts if c > 0)
                / len(tool_call_counts))
            # tool_correct_rate -- fraction of ALL rollouts that BOTH called
            # the tool at least once AND produced a usable boxed answer
            # (f1 >= 0.5).  This is the "tool actually paid off" signal:
            # tool_use_rate high but tool_correct_rate low = the policy is
            # calling the tool for show and still answering wrong.  The
            # 0.5 threshold matches the "correct" cut in standard HotpotQA
            # reporting and lines up with the F1 mid-point.
            log_dict['tool_correct_rate'] = (
                sum(1 for cnt, f1 in zip(tool_call_counts, f1_rewards)
                    if cnt > 0 and f1 >= 0.5)
                / len(tool_call_counts))
        # Diagnostic: fraction of rollouts that did NOT produce a ``\boxed{}``
        # answer in the final assistant message.  Since ``f1`` is strictly
        # gated on ``\boxed{}``, a high rate here directly explains low f1.
        if all_trajectories:
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
