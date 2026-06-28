"""AoPS math competition evaluation: direct vs RAG-augmented with Qwen3.5-4B.

Two modes:
  - ``direct``: The model solves problems directly (4 GPUs, TP=4).
  - ``rag``:    Retrieve top-k thinking traces from LanceDB as 1-shot
                examples, then solve (8 GPUs: DP=4 embedding + TP=4 vLLM).

Only problems with ``metadata.boxed == True`` are used (auto-gradable via
``\\boxed{...}`` extraction).  A random subset is sampled for efficiency.

Launch examples:
    # Direct (4 GPUs, default 500 problems)
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode direct

    # RAG-augmented (8 GPUs)
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode rag

    # Smaller sample for quick test
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode direct --n 100

    # RAG with condenser (retrieves thinking_raw, compresses with local 4B / API)
    EVAL_CONDENSER_GPUS=2 python cookbook/exp/embedding/eval_gpqa_rag.py --mode rag --condense

    # RAG without condenser (uses thinking_raw by default, truncated to max-trace-len)
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode rag

    # RAG with pre-compressed field (opt-in, not recommended for reader LM)
    python cookbook/exp/embedding/eval_gpqa_rag.py --mode rag --use-cot-compressed
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np
import torch

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.data_format import SamplingParams as TwinkleSamplingParams
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Qwen3_5Template
from twinkle_agentic.protocol.openai import OpenAI as OpenAIClient

logger = get_logger()

# -- Condenser config ----------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
CONDENSE_API_KEY = os.environ.get('COMPRESS_API_KEY', '')
CONDENSE_BASE_URL = os.environ.get('COMPRESS_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
CONDENSE_API_MODEL = os.environ.get('COMPRESS_MODEL', 'qwen3.7-max')
CONDENSE_API_CONCURRENCY = int(os.environ.get('API_CONCURRENCY', 32))
CONDENSE_API_MIN_INTERVAL = float(os.environ.get('API_MIN_INTERVAL', 0.1))
CONDENSE_TEMPERATURE = 0.2
CONDENSE_MAX_TOKENS = 8192

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# -- Gen/Embed config ---------------------------------------------------------
GEN_MODEL_ID = os.environ.get('GEN_MODEL_ID', 'Qwen/Qwen3.5-4B')
EMBED_MODEL_ID = os.environ.get(
    'EMBED_MODEL_ID', 'output.oldemb/embedding_full_transformers/last-checkpoint')

GEN_GPUS = int(os.environ.get('GEN_GPUS', 4))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 2))
EMBED_MAX_LENGTH = int(os.environ.get('EMBED_MAX_LENGTH', 20000))

GEN_GPU_MEM = float(os.environ.get('GEN_GPU_MEM', 0.85))
GEN_MAX_MODEL_LEN = int(os.environ.get('GEN_MAX_MODEL_LEN', 65536))
GEN_MAX_TOKENS = int(os.environ.get('GEN_MAX_TOKENS', 65536))
GEN_TEMPERATURE = float(os.environ.get('GEN_TEMPERATURE', 0.6))
GEN_TOP_P = float(os.environ.get('GEN_TOP_P', 0.95))

AOPS_DATASET_ID = os.environ.get('AOPS_DATASET_ID', 'AI-MO/aops')


# ---------------------------------------------------------------------------
# Condenser prompts & validation
# ---------------------------------------------------------------------------

COMPRESS_SYSTEM = """\
You are a reasoning-trace condenser. Given a verbose reasoning trace, \
extract the TRANSFERABLE KNOWLEDGE as an EXECUTABLE SOLUTION SKELETON \
that would help a reader solve SIMILAR problems in the same domain.

Your output is the ENTIRE useful content — there is no expansion tool, no second pass. \
The reader will apply this knowledge to a DIFFERENT problem, so focus on what transfers.

Principles:
1. OUTPUT AN EXECUTABLE STEP CHAIN: numbered steps that a solver can directly follow. \
Each step should state WHAT to do and HOW (with the formula/technique), not just \
name the concept.
2. INCLUDE FULL FORMULAS: theorems, identities, inequalities — state each \
with its COMPLETE MATHEMATICAL EXPRESSION, not just the name.
3. STATE APPLICABILITY: what structural features of a problem signal that this \
approach works (e.g. "when the constraint is a sum of squares").
4. PRESERVE KEY INSIGHTS: the non-obvious ideas or tricks that make the approach \
work — the things a solver would NOT think of without guidance.
5. REMOVE: problem-specific numeric calculations, dead-end explorations, \
hesitations, verbose restatements, and trivial arithmetic.
6. FORMAT: Start with a one-line "Applicability" statement, then numbered steps, \
then key formulas. Keep it concise and actionable.
7. NO meta-commentary about the compression process. NO preamble.
"""

COMPRESS_USER = (
    '## Reader Problem (context only — do NOT solve it)\n{query}\n\n'
    '## Reasoning Trace to Condense\n{text}')


def _is_truncated_compression(text: str) -> bool:
    if not text or not text.strip():
        return True
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return True
    last_line = lines[-1]
    # Truncated if last line looks incomplete (no terminal punctuation/formula)
    if last_line and last_line[-1] not in '.。!！)）]】}\\$':
        # Allow lines ending with numbers, boxed answers, etc.
        if not re.search(r'\d$|\\boxed|\$|\)$', last_line):
            return True
    return False


# -- API rate limiter ----------------------------------------------------------
_api_semaphore = threading.Semaphore(CONDENSE_API_CONCURRENCY)
_api_bucket_lock = threading.Lock()
_api_tokens = [float(CONDENSE_API_CONCURRENCY)]
_api_last_refill = [time.monotonic()]


def _api_throttle():
    _api_semaphore.acquire()
    try:
        with _api_bucket_lock:
            now = time.monotonic()
            elapsed = now - _api_last_refill[0]
            refill = elapsed / CONDENSE_API_MIN_INTERVAL
            _api_tokens[0] = min(float(CONDENSE_API_CONCURRENCY), _api_tokens[0] + refill)
            _api_last_refill[0] = now
            if _api_tokens[0] >= 1.0:
                _api_tokens[0] -= 1.0
            else:
                wait = (1.0 - _api_tokens[0]) * CONDENSE_API_MIN_INTERVAL
                _api_tokens[0] = 0.0
                time.sleep(wait)
    finally:
        _api_semaphore.release()


def _api_condense_single(api_client: OpenAIClient, messages: List[Dict]) -> Optional[str]:
    _api_throttle()
    trajectory = {'messages': messages}
    sp = TwinkleSamplingParams(temperature=CONDENSE_TEMPERATURE, max_tokens=CONDENSE_MAX_TOKENS)
    try:
        reply = api_client(trajectory, sp, extra_body={'enable_thinking': False})
    except Exception as exc:
        logger.warning(f'[condense-api] error: {exc}')
        return None
    content = (reply.get('content') or '').strip()
    if not content:
        return None
    m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
    if m:
        content = m.group(1).strip()
    return content


def condense_traces(
    examples_batch: List[List[Dict[str, str]]],
    problems: List[str],
    api_client: OpenAIClient,
    condenser_sampler=None,
    compress_params=None,
    special_tokens: set = None,
    max_output_len: int = 2000,
) -> List[List[Dict[str, str]]]:
    """Compress retrieved thinking traces with query-aware condenser.

    Primary: local vLLM condenser (if provided).
    Fallback: API condenser.
    Final fallback: raw trace truncated to max_output_len.
    """
    result: List[List[Dict[str, str]]] = []
    # Flatten all (batch_idx, ex_idx, problem, example) for batch processing
    tasks = []
    for bi, (exs, prob) in enumerate(zip(examples_batch, problems)):
        for ei, ex in enumerate(exs):
            tasks.append((bi, ei, prob, ex))

    if not tasks:
        return [[] for _ in examples_batch]

    # Build condense prompts (aligned with make_embedding_dataset.py hard path)
    prompts = []
    for _, _, prob, ex in tasks:
        user_msg = COMPRESS_USER.format(query=prob, text=ex['thinking'])
        prompts.append([{'role': 'system', 'content': COMPRESS_SYSTEM},
                        {'role': 'user', 'content': user_msg}])

    # Phase 1: local vLLM condenser
    condensed = [None] * len(tasks)
    condense_sources = ['raw'] * len(tasks)
    fallback_indices = []

    if condenser_sampler is not None and compress_params is not None:
        sampler_inputs = [{'messages': p} for p in prompts]
        try:
            responses = condenser_sampler.sample(sampler_inputs, compress_params)
        except Exception as exc:
            logger.warning(f'[condense] sampler error: {exc}')
            responses = [None] * len(sampler_inputs)
        for ri, resp in enumerate(responses):
            seq = resp.sequences[0] if resp and resp.sequences else None
            text = ''
            if seq and seq.stop_reason != 'length' and seq.decoded:
                text = seq.decoded
                if special_tokens:
                    for tok in special_tokens:
                        text = text.replace(tok, '')
                text = text.rstrip()
            if text and not _is_truncated_compression(text):
                condensed[ri] = text
                condense_sources[ri] = 'local'
            else:
                fallback_indices.append(ri)
    else:
        fallback_indices = list(range(len(tasks)))

    # Phase 2: API fallback
    if fallback_indices and api_client:
        with ThreadPoolExecutor(max_workers=CONDENSE_API_CONCURRENCY) as pool:
            futures = {}
            for ri in fallback_indices:
                futures[pool.submit(_api_condense_single, api_client, prompts[ri])] = ri
            for fut in as_completed(futures):
                ri = futures[fut]
                api_result = fut.result()
                if api_result and not _is_truncated_compression(api_result):
                    condensed[ri] = api_result
                    condense_sources[ri] = 'api'

    # Phase 3: assemble results (fallback to raw truncation)
    result = [[] for _ in examples_batch]
    for ti, (bi, ei, prob, ex) in enumerate(tasks):
        compressed = condensed[ti]
        raw_len = len(ex['thinking'])
        sim_val = ex.get('_sim', 0.0)
        if compressed:
            result[bi].append({'query': ex['query'],
                               'thinking': _strip_condenser_markers(compressed),
                               '_condense_source': condense_sources[ti],
                               '_raw_trace_len': raw_len, '_sim': sim_val})
        else:
            result[bi].append({'query': ex['query'],
                               'thinking': ex['thinking'][:max_output_len],
                               '_condense_source': 'raw',
                               '_raw_trace_len': raw_len, '_sim': sim_val})

    n_ok = sum(1 for c in condensed if c)
    logger.info(f'[condense] {n_ok}/{len(tasks)} compressed ok, '
                f'{len(tasks) - n_ok} fell back to raw truncation')
    return result


def _strip_condenser_markers(text: str) -> str:
    """Light cleanup of condenser output.

    Removes any residual markdown headers or meta-lines that don't carry
    solution content. Keeps numbered steps and equations intact.
    """
    # Remove legacy ## headers if condenser still emits them
    if '## More' in text:
        text = text.split('## More', 1)[0]
    text = re.sub(r'^##\s*Summary\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Topic:\s*.*\n?', '', text, flags=re.MULTILINE)
    # Remove meta-commentary lines
    text = re.sub(r'^\s*\(Note:.*\)\s*$', '', text, flags=re.MULTILINE)
    return text.strip()


# ---------------------------------------------------------------------------
# Boxed answer extraction
# ---------------------------------------------------------------------------
_BOXED_RE = re.compile(r'\\boxed\s*\{')


def extract_boxed(text: str) -> Optional[str]:
    """Extract the last \\boxed{...} content, handling nested braces."""
    if not text:
        return None
    last_match = None
    for m in _BOXED_RE.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        if depth == 0:
            last_match = text[start:i - 1].strip()
    return last_match


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    if not ans:
        return ''
    s = ans.strip()
    # MCQ: extract bare letter from \textbf{(D)}, \text{(A)}, (B), etc.
    m = re.match(r'^\\?(?:textbf|text|mathrm|mathbf)?\{?\(?([A-E])\)?\}?$', s)
    if m:
        return m.group(1)
    s = s.replace(' ', '')
    s = s.replace(r'\,', '')
    s = s.replace(r'\;', '')
    s = s.replace(r'\!', '')
    s = s.replace(r'\text', '')
    s = s.replace(r'\mathrm', '')
    s = s.replace(r'\displaystyle', '')
    s = re.sub(r'\\(?:left|right)[.()\[\]|]', '', s)
    s = s.replace(r'\dfrac', r'\frac')
    s = s.replace(r'\tfrac', r'\frac')
    s = s.strip('$').strip()
    # Strip unit-like brace suffixes: {cm}, {m}, {kg}, etc.
    s = re.sub(r'\{[a-zA-Z]+\}$', '', s)
    # Normalize degree: ^\circ, ^{\circ}, ° → deg
    s = re.sub(r'\^\\circ|\^\{\\circ\}|°', 'deg', s)
    # Canonicalize \frac{a}{b} → (a)/(b)
    def _frac_to_slash(m):
        # Handle nested braces in numerator/denominator
        text = m.group(0)
        pos = text.index('{') + 1
        depth, num_start = 1, pos
        while depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        numer = text[num_start:pos - 1]
        pos += 1  # skip '{'
        den_start = pos
        depth = 1
        while depth > 0:
            if text[pos] == '{': depth += 1
            elif text[pos] == '}': depth -= 1
            pos += 1
        denom = text[den_start:pos - 1]
        return f'({numer})/({denom})'
    s = re.sub(r'\\frac\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', _frac_to_slash, s)
    # Also handle bare a/b → (a)/(b) for consistent comparison
    # Only simple integer/variable fractions: 17/5 → (17)/(5)
    s = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'(\1)/(\2)', s)
    return s


def _try_numeric_equal(a: str, b: str) -> bool:
    """Try to evaluate both as floats; match if within 1e-9 relative tolerance."""
    try:
        va = float(a.replace('(', '').replace(')', ''))
        vb = float(b.replace('(', '').replace(')', ''))
        return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
    except (ValueError, ZeroDivisionError):
        pass
    # Try evaluating simple fraction expressions like (17)/(5)
    frac_re = re.compile(r'^\(([^)]+)\)/\(([^)]+)\)$')
    def _eval_frac(s):
        m = frac_re.match(s)
        if m:
            try:
                return float(m.group(1)) / float(m.group(2))
            except (ValueError, ZeroDivisionError):
                pass
        return None
    va, vb = _eval_frac(a), _eval_frac(b)
    if va is not None and vb is not None:
        return abs(va - vb) < 1e-9 * max(1, abs(va), abs(vb))
    return False


def answers_match(predicted: str, reference: str) -> bool:
    """Check if two math answers are equivalent."""
    if not predicted or not reference:
        return False
    norm_p = normalize_answer(predicted)
    norm_r = normalize_answer(reference)
    if norm_p == norm_r:
        return True
    return _try_numeric_equal(norm_p, norm_r)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_aops(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Load AoPS boxed problems, sample n, extract reference answers."""
    from modelscope import MsDataset
    ds = MsDataset.load(AOPS_DATASET_ID, split='train',
                        download_mode='reuse_dataset_if_exists')
    boxed = []
    for row in ds:
        if not row['metadata'].get('boxed'):
            continue
        ref = extract_boxed(row['solution'])
        if not ref:
            continue
        boxed.append({
            'problem': row['problem'],
            'solution': row['solution'],
            'reference_answer': ref,
            'tags': row.get('tags', []),
        })
    sys.stderr.write(f'[aops] {len(boxed)} boxed problems with extractable answers\n')
    rng = random.Random(seed)
    rng.shuffle(boxed)
    if n > 0 and n < len(boxed):
        boxed = boxed[:n]
        sys.stderr.write(f'[aops] sampled {n} problems\n')
    return boxed


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.'
)

RAG_SYSTEM = (
    'You are an expert competition mathematician. Solve the following problem '
    'step by step. Provide your final answer inside \\boxed{}.\n\n'
    'You will first see example problem-solving traces or skills. '
    'Learn from the reasoning methodology demonstrated in these examples, '
    'then thinking to solve the actual problem.'
)

RAG_FOLLOWUP = (
    'The above is a reference solution to a similar problem. '
    'You may use any applicable techniques from it, or ignore it '
    'if you find a better approach. '
    'Solve the problem step by step and put your final answer in \\boxed{}.'
)


def build_direct_prompt(problem: str) -> Dict[str, Any]:
    return {
        'messages': [
            {'role': 'system', 'content': DIRECT_SYSTEM},
            {'role': 'user', 'content': problem},
        ]
    }


def build_rag_prompt(problem: str,
                     examples: List[Dict[str, str]]) -> Dict[str, Any]:
    """Approach B: multi-turn assistant format.

    The trace is presented as an assistant "retrieval" turn, followed by
    a user instruction that constrains the model to use methodology only.
    """
    messages: List[Dict[str, str]] = [{'role': 'system', 'content': DIRECT_SYSTEM}]
    messages.append({'role': 'user', 'content': problem})
    # Build trace content from retrieved examples
    trace_parts = []
    for i, ex in enumerate(examples, 1):
        trace_parts.append(f'[Retrieved Example {i}]\nProblem: {ex["query"]}\n'
                           f'Reasoning:\n{ex["thinking"]}')
    trace_text = '\n\n'.join(trace_parts)
    messages.append({'role': 'assistant',
                     'content': f'I found relevant reasoning traces from the knowledge base!\n\n{trace_text}'})
    messages.append({'role': 'user', 'content': RAG_FOLLOWUP})
    return {'messages': messages}


# ---------------------------------------------------------------------------
# 13-gram Jaccard decontamination
# ---------------------------------------------------------------------------

def _normalize_for_ngram(text: str) -> str:
    """Normalize text for n-gram comparison: strip LaTeX markup, lowercase."""
    text = text.lower()
    text = re.sub(r'\$+', '', text)
    text = re.sub(r'\\[a-z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'\\[a-z]+', ' ', text)
    text = re.sub(r'[{}\\^_$]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _ngram_jaccard(text_a: str, text_b: str, n: int = 13) -> float:
    """13-gram character-level Jaccard similarity."""
    a = _normalize_for_ngram(text_a)
    b = _normalize_for_ngram(text_b)
    if len(a) < n or len(b) < n:
        return 0.0
    grams_a = set(a[i:i + n] for i in range(len(a) - n + 1))
    grams_b = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not grams_a or not grams_b:
        return 0.0
    return len(grams_a & grams_b) / len(grams_a | grams_b)


# ---------------------------------------------------------------------------
# Embedding / RAG helpers
# ---------------------------------------------------------------------------

def _wrap_anchor(text: str) -> List[Dict[str, str]]:
    return [
        {'role': 'user', 'content': text},
        {'role': 'assistant', 'content': 'Match the correct response here.'},
    ]


def get_embeddings(model: TransformersModel, template: Qwen3_5Template,
                   texts: List[str], dp_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0,), dtype=np.float32)
    n = len(texts)
    pad_n = (-n) % dp_size
    padded = list(texts) + [' '] * pad_n if pad_n else list(texts)
    features = []
    for t in padded:
        feat = template.encode({'messages': _wrap_anchor(t or ' ')})
        feat['labels'] = [1]
        features.append(feat)
    out = model.forward_only(inputs=features, task='embedding', return_logits=True)
    emb = out['embeddings']
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().to(torch.float32).cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32)
    return emb[:n] if pad_n else emb


def retrieve_examples(tbl, query_vecs: np.ndarray, top_k: int,
                      use_thinking_raw: bool, sim_threshold: float = 0.0,
                      problems: List[str] = None,
                      decontam_threshold: float = 0.0,
                      ) -> List[List[Dict[str, str]]]:
    thinking_field = 'thinking_raw' if use_thinking_raw else 'cot_compressed'
    # Fetch extra candidates when decontamination is active
    fetch_limit = top_k + 50 if decontam_threshold > 0 else top_k
    all_examples: List[List[Dict[str, str]]] = []
    decontam_skipped = 0
    for qi, vec in enumerate(query_vecs):
        results = (
            tbl.search(vec.astype(np.float32).tolist())
            .metric('dot')
            .limit(fetch_limit)
            .select(['query_raw', thinking_field, '_distance'])
            .to_list()
        )
        problem_text = problems[qi] if problems else ''
        examples = []
        for r in results:
            if len(examples) >= top_k:
                break
            sim = 1.0 - r.get('_distance', 0.0)
            if sim < sim_threshold:
                continue
            q = r.get('query_raw', '')
            t = r.get(thinking_field, '')
            if not t:
                continue
            # 13-gram decontamination: skip if retrieved query overlaps with problem
            if decontam_threshold > 0 and problem_text and q:
                ng_sim = _ngram_jaccard(problem_text, q)
                if ng_sim > decontam_threshold:
                    decontam_skipped += 1
                    continue
            examples.append({'query': q, 'thinking': t, '_sim': round(sim, 4),
                             '_raw_trace_len': len(t)})
        all_examples.append(examples)
    if decontam_skipped > 0:
        logger.info(f'[decontam] skipped {decontam_skipped} leaked retrievals '
                    f'(13-gram Jaccard > {decontam_threshold})')
    return all_examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['direct', 'rag'], default='direct')
    p.add_argument('--n', type=int, default=500,
                   help='Number of problems to sample (0 = all boxed).')
    p.add_argument('--db-path', default='./output/thinking_rag/lance.db')
    p.add_argument('--table', default='thinking_traces')
    p.add_argument('--top-k', type=int, default=3)
    p.add_argument('--use-cot-compressed', action='store_true',
                   help='Use pre-compressed cot_compressed field instead of thinking_raw.')
    p.add_argument('--sim-threshold', type=float, default=0.75)
    p.add_argument('--rag-fallback-sim', type=float, default=0.70,
                   help='Fallback to direct prompt when best retrieval similarity '
                        'is below this threshold (avoids weak-trace loops).')
    p.add_argument('--decontam-threshold', type=float, default=0.20,
                   help='13-gram Jaccard threshold for leak detection. '
                        'Retrieved traces above this are skipped (0=disabled).')
    p.add_argument('--max-trace-len', type=int, default=12000)
    p.add_argument('--condense', action='store_true',
                   help='Enable condenser re-compression on retrieved traces.')
    p.add_argument('--condense-max-len', type=int, default=2000,
                   help='Max chars of condensed trace (fallback truncation).')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = f'./output/thinking_rag/aops_{args.mode}_results.jsonl'

    if args.condense and args.use_cot_compressed:
        logger.warning('--condense requires thinking_raw, ignoring --use-cot-compressed')
        args.use_cot_compressed = False

    records = load_aops(n=args.n, seed=args.seed)

    is_rag = (args.mode == 'rag')

    condenser_gpus = int(os.environ.get('EVAL_CONDENSER_GPUS', 0)) if args.condense else 0

    if is_rag:
        num_gpus = EMB_GPUS + GEN_GPUS + condenser_gpus
        device_groups = [
            DeviceGroup(name='emb_model', ranks=list(range(EMB_GPUS)),
                        device_type='GPU'),
            DeviceGroup(name='sampler',
                        ranks=list(range(EMB_GPUS, EMB_GPUS + GEN_GPUS)),
                        device_type='GPU', gpus_per_worker=GEN_GPUS),
        ]
        if condenser_gpus > 0:
            cond_start = EMB_GPUS + GEN_GPUS
            device_groups.append(
                DeviceGroup(name='condenser',
                            ranks=list(range(cond_start, cond_start + condenser_gpus)),
                            device_type='GPU'))
        emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
        gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, tp_size=GEN_GPUS)
        twinkle.initialize(mode='ray', nproc_per_node=num_gpus,
                           groups=device_groups, lazy_collect=False)
    else:
        device_groups = [
            DeviceGroup(name='sampler', ranks=list(range(GEN_GPUS)),
                        device_type='GPU', gpus_per_worker=GEN_GPUS),
        ]
        gen_mesh = DeviceMesh.from_sizes(world_size=GEN_GPUS, tp_size=GEN_GPUS)
        twinkle.initialize(mode='ray', nproc_per_node=GEN_GPUS,
                           groups=device_groups, lazy_collect=False)

    sampler = vLLMSampler(
        model_id=GEN_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': GEN_GPU_MEM,
            'max_model_len': GEN_MAX_MODEL_LEN,
        },
        device_mesh=gen_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=GEN_MODEL_ID,
                         enable_thinking=True, max_length=GEN_MAX_MODEL_LEN)
    sys.stderr.write(f'[aops] vLLM sampler ready (model={GEN_MODEL_ID})\n')

    gen_params = TwinkleSamplingParams(
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
        num_samples=1,
    )

    emb_model = emb_template = tbl = None
    if is_rag:
        import lancedb
        db = lancedb.connect(args.db_path)
        if args.table not in db.table_names():
            raise SystemExit(f'Table "{args.table}" not found in {args.db_path}')
        tbl = db.open_table(args.table)
        sys.stderr.write(f'[aops] LanceDB rows={tbl.count_rows()}\n')

        emb_model = TransformersModel(
            model_id=EMBED_MODEL_ID, device_mesh=emb_mesh,
            remote_group='emb_model')
        emb_model.set_processor(InputProcessor)
        emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
        emb_template = Qwen3_5Template(
            model_id=EMBED_MODEL_ID, max_length=EMBED_MAX_LENGTH,
            truncation_strategy='delete', enable_thinking=False)
        sys.stderr.write('[aops] embedding model ready\n')

    # -- Condenser setup (API primary + optional local vLLM) -------------------
    condenser_api_client = None
    condenser_sampler_obj = None
    condenser_params = None
    condenser_special_tokens = None

    if args.condense and is_rag:
        condenser_api_client = OpenAIClient(
            model=CONDENSE_API_MODEL, api_key=CONDENSE_API_KEY,
            base_url=CONDENSE_BASE_URL)
        sys.stderr.write(f'[condense] API client ready (model={CONDENSE_API_MODEL})\n')

        if condenser_gpus > 0:
            condenser_mesh = DeviceMesh.from_sizes(
                world_size=condenser_gpus, dp_size=condenser_gpus)
            condenser_sampler_obj = vLLMSampler(
                model_id=CONDENSE_MODEL_ID,
                engine_args={'gpu_memory_utilization': 0.8, 'max_model_len': 32768},
                device_mesh=condenser_mesh,
                remote_group='condenser',
            )
            condenser_sampler_obj.set_template(
                'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
                enable_thinking=False, truncation_strategy='delete',
                max_length=32768)
            condenser_template = Qwen3_5Template(
                model_id=CONDENSE_MODEL_ID, max_length=32768,
                enable_thinking=False, truncation_strategy='delete')
            condenser_special_tokens = set(condenser_template.tokenizer.all_special_tokens)
            condenser_params = TwinkleSamplingParams(
                max_tokens=CONDENSE_MAX_TOKENS,
                temperature=CONDENSE_TEMPERATURE,
                top_p=0.5, num_samples=1)
            sys.stderr.write(f'[condense] local vLLM ready (model={CONDENSE_MODEL_ID})\n')

    correct_count = 0
    total_count = 0
    debug_records: List[Dict[str, Any]] = []

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    out_f = open(args.output, 'w', encoding='utf-8')

    for batch_start in range(0, len(records), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(records))
        batch = records[batch_start:batch_end]

        if is_rag:
            problems = [r['problem'] for r in batch]
            query_vecs = get_embeddings(emb_model, emb_template, problems,
                                        EMB_GPUS)
            use_raw = not args.use_cot_compressed
            all_examples = retrieve_examples(tbl, query_vecs, args.top_k,
                                             use_raw,
                                             args.sim_threshold,
                                             problems=problems,
                                             decontam_threshold=args.decontam_threshold)
            # Strip structural markers from cot_compressed field
            if args.use_cot_compressed:
                for exs in all_examples:
                    for ex in exs:
                        ex['thinking'] = _strip_condenser_markers(ex['thinking'])

            # Condenser re-compression
            if args.condense and condenser_api_client:
                all_examples = condense_traces(
                    all_examples, problems, condenser_api_client,
                    condenser_sampler=condenser_sampler_obj,
                    compress_params=condenser_params,
                    special_tokens=condenser_special_tokens,
                    max_output_len=args.condense_max_len)

            prompts = []
            for r, examples in zip(batch, all_examples):
                if not examples:
                    prompts.append(build_direct_prompt(r['problem']))
                else:
                    # Drop traces exceeding max_trace_len instead of truncating
                    filtered = [{'query': ex['query'], 'thinking': ex['thinking']}
                                for ex in examples
                                if len(ex['thinking']) <= args.max_trace_len]
                    # Fallback to direct if best similarity is too low
                    best_sim = max(ex.get('_sim', 0.0) for ex in examples)
                    if filtered and best_sim >= args.rag_fallback_sim:
                        prompts.append(build_rag_prompt(r['problem'], filtered))
                    else:
                        prompts.append(build_direct_prompt(r['problem']))
        else:
            prompts = [build_direct_prompt(r['problem']) for r in batch]

        responses = sampler.sample(prompts, gen_params)

        for i, (rec, resp) in enumerate(zip(batch, responses)):
            seq = resp.sequences[0] if resp and resp.sequences else None
            raw_output = ''
            if seq is not None:
                raw_output = seq.decoded or ''
                raw_output = re.sub(r'<\|[^|]+\|>', '', raw_output).rstrip()

            predicted = extract_boxed(raw_output)
            is_correct = answers_match(predicted, rec['reference_answer'])
            if is_correct:
                correct_count += 1
            total_count += 1

            debug_rec = {
                'idx': batch_start + i,
                'reference_answer': rec['reference_answer'],
                'predicted': predicted,
                'is_correct': is_correct,
                'problem': rec['problem'],
                'model_output': raw_output,
            }
            if is_rag:
                debug_rec['num_traces'] = len(all_examples[i])
                if all_examples[i]:
                    ex0 = all_examples[i][0]
                    debug_rec['similarity'] = ex0.get('_sim', 0.0)
                    debug_rec['retrieved_query'] = ex0.get('query', '')
                    debug_rec['raw_trace_len'] = ex0.get('_raw_trace_len', 0)
                    debug_rec['condensed_trace'] = ex0['thinking']
                    debug_rec['condensed_trace_len'] = len(ex0['thinking'])
                    debug_rec['condense_source'] = ex0.get('_condense_source', '')
            debug_records.append(debug_rec)
            out_f.write(json.dumps(debug_rec, ensure_ascii=False) + '\n')
            out_f.flush()

        acc = correct_count / total_count if total_count else 0
        sys.stderr.write(
            f'  [{batch_end}/{len(records)}] '
            f'acc={acc:.4f} ({correct_count}/{total_count})\n')

    overall_acc = correct_count / total_count if total_count else 0
    print(f'\n{"=" * 60}')
    print(f'AoPS Math — mode={args.mode}, model={GEN_MODEL_ID}')
    print(f'  n={total_count}, seed={args.seed}')
    print(f'{"=" * 60}')
    print(f'Overall accuracy: {overall_acc:.4f}  ({correct_count}/{total_count})')

    out_f.close()
    print(f'\n[output] {len(debug_records)} records saved to {args.output}')


if __name__ == '__main__':
    main()
