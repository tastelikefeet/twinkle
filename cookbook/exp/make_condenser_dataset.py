import argparse
import hashlib
import json
import os
import re
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, Iterator, List, Optional, Set

from tqdm import tqdm

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


# ═══════════════════════════════════════════════════════════════════════════════
# Prompts
# ═══════════════════════════════════════════════════════════════════════════════

QUERY_GEN_SYSTEM = """\
You are a query designer. Given a source passage, enumerate distinct information \
queries a reader might ask of it. Each query must steer toward a meaningfully \
DIFFERENT compression of the same source — different facets, not rephrasings of \
the same need.

Category hints (not exhaustive — combine or invent as fits the source):
- Interface extraction (code): class / method signatures, parameter and return types
- Functional summary: what the passage accomplishes at a high level
- Error & pitfall analysis: bugs, anti-patterns, failure modes, edge cases
- Experience distillation: lessons learned, best practices, do's and don'ts
- Skill extraction (knowledge-as-skill): WHAT this passage lets you do, HOW to \
apply it as reusable steps, WHEN to invoke it (trigger conditions / use cases)
- Abstract analysis: design patterns, architectural decisions, trade-offs
- Information summary: key facts, entities, numbers, relationships
- Dependency & context: prerequisites, imports, environment, related modules

Rules:
1. SHAPE — each query is one short imperative or interrogative sentence (e.g. \
"List all public method signatures with parameter and return types", "What race \
conditions does this code contain?").
2. DISTINCT — reject any pair whose answers would substantially overlap; \
rephrasings of the same information need do NOT count as separate queries.
3. SKILL FOR KNOWLEDGE — when the source reads as tutorial / experience / \
how-to / domain knowledge, ALWAYS include exactly one skill-style query asking \
what the reader can accomplish with it and how to apply it (phrased in the \
source language).
4. ANSWERABLE — skip queries the source cannot actually answer, and skip \
trivial queries that would just reproduce the source verbatim.
5. SCALE — short / single-purpose → 1; medium → 2; rich / multi-topic → 3–4. \
Do not pad.
6. LANGUAGE — query language MUST match the source language.
7. OUTPUT — a single JSON array of strings; no preamble, no code fences, \
nothing else.\
"""

QUERY_GEN_USER = "Analyze the following text and return a JSON array of queries.\n\n{text}"

COMPRESS_SYSTEM = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Read inline` directly, then calls `extract_compressed` \
on any topic-key listed under `## Call extract_compressed for` to recover its \
fuller content.

  `## Read inline`               — extreme-density text the reader reads directly.
  `## Call extract_compressed for` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Read inline
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## Call extract_compressed for
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Format selection for the inline body (pick the MOST COMPACT form per query, mix \
when helpful):
- Interface / signature → code notation directly: `func(a:int)->str`
- Factual / entity → telegraphic prose; drop function words; ":" for "is", "," \
for "has"
- Skill / how-to / usage → lead with `Use when: <trigger>`; numbered telegraphic \
steps `1.do X 2.then Y`; close with `Output: <result>` when relevant
- Procedural → numbered short steps
- Analytical / design → hierarchical bullets with abbreviations

`## Read inline` rules:
1. TOPIC LINE — line 1 is ALWAYS `Topic: <subject — scope>`, even when the \
query is narrow. Anchors both the reader and the tool.
2. DENSITY — every token in the body carries query-relevant signal; cut filler.
3. PRIMARY-COMPLETE — never silently drop a fact essential to answering the \
query. Anything cut for length MUST appear as a key under \
`## Call extract_compressed for`.
4. NON-MISLEADING — phrasing must not let the reader infer anything the source \
does not support; partial truths that mislead are worse than honest omissions \
flagged in the index.
5. SELF-CONTAINED — the reader can act on the answer without re-opening the source.
6. FAITHFUL — only content the source supports; no fabrication, no extrapolation.
7. LANGUAGE — match the source language.
8. NO outer code fences around the whole answer; no meta-commentary.

`## Call extract_compressed for` rules (MANDATORY — this section is never omitted):
1. FORMAT — each bullet is `- <topic-key>: <one-line hint>`:
   • topic-key — short, unambiguous, grounded in source vocabulary so the \
`extract_compressed` tool can locate the aspect (e.g. `decorators`, \
`error handling`, `pitfalls`).
   • hint — tells WHAT the reader gains by expanding (concrete numbers, code \
listings, secondary cases, edge details, related context, …); do NOT restate \
the inline answer.
2. CRITERION — each bullet names an aspect that EXISTS in the source but is \
NOT fully captured inline. Material that genuinely fits inline without \
distortion MUST NOT be duplicated here.
3. FAITHFUL — hints must be grounded in the source; never speculate or invent.
4. ORDER — by relevance to the query, then by importance.
5. EMPTY CASE — if the source is so short / single-purpose that everything \
fits inline, write a single line `- (none)`.

Examples:

Query: List all public method signatures with parameter and return types
Source: (a Python HTTP client class with retry decorator, structured logging, \
and request helpers)
## Read inline
Topic: Python HTTP client class — public surface of retried request helpers.
retry_request(url:str, max_retries:int=3, timeout:float=10.0) -> Response
fetch_json(endpoint:str, params:dict|None=None) -> dict
post_data(endpoint:str, payload:dict, headers:dict|None=None) -> Response

## Call extract_compressed for
- decorators: @retry config — exponential backoff (base=2.0, max=60s)
- logging: structured per-request logs with request_id and latency_ms
- private helpers: _build_headers, _parse_error — not in public surface
───
Query: What can this passage help you accomplish, and how to use it?
Source: (a tutorial on configuring Linux cgroups v2 caps for a systemd service)
## Read inline
Topic: Linux cgroups v2 — per-service CPU / memory caps via systemd slice units.
Use when: needing per-service CPU/memory caps on systemd hosts.
1.create slice unit /etc/systemd/system/<name>.slice with CPUQuota=, MemoryMax=
2.attach service via Slice=<name>.slice in [Service]
3.systemctl daemon-reload + restart service
4.verify: systemctl status <svc> shows Tasks/CPU/Memory inside slice
Output: hard caps enforced by kernel cgroup v2.

## Call extract_compressed for
- pitfalls: cgroup v1/v2 mode detection, MemorySwapMax behavior on OOM
- delegation: Delegate=yes for nested controllers in container managers
- examples: nginx and postgres slice templates with concrete numeric caps
- diagnostics: systemd-cgls / systemd-cgtop walkthrough
───
Query: 总结这段代码的错误和改进经验
Source: (一段有 race condition 和未关闭资源的 Go 代码)
## Read inline
Topic: Go HTTP fetch 循环 — 并发写共享 map + 未关闭响应体导致的稳定性缺陷。
1.race: 并发写 map 未锁 → sync.RWMutex 或 sync.Map
2.泄漏: resp.Body 未 Close → 请求后立即 defer resp.Body.Close()
3.吞错: err 未检查 → 每处 err!=nil 必处理或上抛

## Call extract_compressed for
- (none)

Now begin.\
"""

COMPRESS_USER = "## Query\n{query}\n\n## Source\n{text}"

# Short system prompt embedded in emitted SFT samples — the long COMPRESS_SYSTEM
# is for data generation only; training samples carry only the binding contract.
COMPRESS_SYSTEM_TRAIN = """\
You are a compression assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Read inline` directly, then calls `extract_compressed` \
on any topic-key listed under `## Call extract_compressed for` to recover its \
fuller content.

Output skeleton:

## Read inline
Topic: <subject — scope, one line>
<dense body answering the query>

## Call extract_compressed for
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Rules:
1. Line 1 of `## Read inline` is ALWAYS `Topic: ...`.
2. Body is maximally dense; every token carries query-relevant signal.
3. Never silently drop a fact — anything cut for length MUST appear as a key \
under `## Call extract_compressed for` (do not duplicate inline material here).
4. No fabrication, no extrapolation, no misleading partial truths.
5. Match the source language. No outer code fences, no meta-commentary.\
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_json_array(text: str) -> Optional[List[str]]:
    """Best-effort extraction of a JSON string array from LLM output."""
    text = text.strip()
    # Try direct parse first
    if text.startswith('['):
        try:
            arr = json.loads(text)
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
    # Fallback: find first [...] block
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            arr = json.loads(m.group())
            if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
                return arr
        except json.JSONDecodeError:
            pass
    return None


def generate_queries(api: OpenAI, text: str) -> List[str]:
    """Phase 1: ask the LLM what queries can be asked about ``text``."""
    trajectory = {
        'messages': [
            {'role': 'system', 'content': QUERY_GEN_SYSTEM},
            {'role': 'user', 'content': QUERY_GEN_USER.format(text=text)},
        ]
    }
    sp = SamplingParams(temperature=0.7, max_tokens=1024)
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[query_gen] error: {exc}\n')
            return []
        content = reply.get('content') or ''
        queries = _extract_json_array(content)
        if queries:
            return queries
        if attempt == 0:
            sys.stderr.write('[query_gen] retry: failed to parse JSON array\n')
    return []


def compress_for_query(api: OpenAI, text: str, query: str,
                       thinking_budget: int = 1024) -> Optional[str]:
    """Phase 2: compress ``text`` w.r.t. ``query``. Returns compressed content or None."""
    trajectory = {
        'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
        ]
    }
    sp = SamplingParams(temperature=0.3, max_tokens=16384)
    for attempt in range(2):
        try:
            reply = api(trajectory, sp, extra_body={
                'enable_thinking': False,
                'thinking_budget': thinking_budget,
            })
        except Exception as exc:
            sys.stderr.write(f'[compress] error: {exc}\n')
            return None
        content = (reply.get('content') or '').strip()
        if not content:
            if attempt == 0:
                sys.stderr.write('[compress] retry: empty response\n')
            continue
        # Strip whole-answer code fence if present.
        m = re.match(r'^```[a-zA-Z]*\n(.*?)\n```\s*$', content, re.DOTALL)
        if m:
            content = m.group(1).strip()
        if not (re.search(r'(?im)^##\s*Read\s+inline\b', content)
                and re.search(r'(?im)^##\s*Call\s+extract_compressed\s+for\b', content)):
            if attempt == 0:
                sys.stderr.write('[compress] retry: missing required sections\n')
            continue
        return content
    return None


def _query_hash(query: str) -> str:
    """Stable short hash of a query string — embedded in sample id for resume."""
    return hashlib.md5(query.strip().encode('utf-8')).hexdigest()[:8]


def process_item(
    api: OpenAI,
    item: Dict[str, Any],
    done_sample_ids: Optional[Set[str]] = None,
    thinking_budget: int = 1024,
) -> List[Dict[str, Any]]:
    """Run both phases on one dataset item. Returns list of SFT samples.

    Input rows come from ``dataset.py``: each row carries a SINGLE assistant
    message holding the passage to compress. ``done_sample_ids`` (full sample
    ids already on disk for this item) lets resume skip queries that were
    already emitted, keyed by query content hash so a phase-1 reorder still
    resolves correctly.
    """
    done = done_sample_ids or set()
    messages = item.get('messages') or []
    text = ''
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get('role') != 'assistant':
            continue
        content = m.get('content')
        if isinstance(content, str) and content.strip():
            text = content.strip()
            break
    if not text or len(text) < 100:
        return []

    item_id = item.get('id')
    if not item_id:
        return []
    source = item.get('source', 'unknown')

    queries = generate_queries(api, text)
    if not queries:
        return []
    queries = queries[:2]

    samples: List[Dict[str, Any]] = []
    for query in queries:
        sample_id = f'{item_id}__{_query_hash(query)}'
        if sample_id in done:
            continue
        compressed = compress_for_query(api, text, query, thinking_budget=thinking_budget)
        if not compressed:
            continue
        sft_messages = [
            {'role': 'system', 'content': COMPRESS_SYSTEM_TRAIN},
            {'role': 'user', 'content': COMPRESS_USER.format(query=query, text=text)},
            {'role': 'assistant', 'content': compressed},
        ]
        samples.append({
            'id': sample_id,
            'source': source,
            'query': query,
            'original_len': len(text),
            'compressed_len': len(compressed),
            'original_tokens': 0,
            'compressed_tokens': 0,
            'messages': sft_messages,
            # Stashed for sparse tokenization on main thread; popped before write.
            '__src': text,
            '__cmp': compressed,
        })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def iter_input(path: str) -> Iterator[Dict[str, Any]]:
    """Stream JSONL dataset row-by-row (no full-file load)."""
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_dataset_py(total: Optional[int], load_from_cache_file: bool) -> Iterator[Dict[str, Any]]:
    """Stream rows directly from ``dataset.py::get_dataset`` without any JSONL hop."""
    # Lazy import: dataset.py triggers HF / ModelScope downloads at module load.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataset import get_dataset
    hf = get_dataset(total=total, load_from_cache_file=load_from_cache_file)
    sys.stderr.write(f'Loaded dataset.py::get_dataset: {len(hf)} rows\n')
    for row in hf:
        yield row


def load_done_sample_ids(path: str) -> Set[str]:
    """Collect already-written full sample ids (``base__hash``) for resume."""
    if not os.path.exists(path):
        return set()
    done: Set[str] = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = obj.get('id', '')
            if sid:
                done.add(sid)
    return done


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Two-phase query-diverse condenser dataset builder.')
    parser.add_argument('--input', default=None,
                        help='Optional JSONL override; default uses dataset.py::get_dataset')
    parser.add_argument('--output', required=True,
                        help='Output JSONL file for SFT samples')
    parser.add_argument('--total', type=int, default=0,
                        help='Total input rows for proportional scaling in dataset.py (0 = base sizes)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable load_from_cache_file when calling dataset.py::get_dataset')
    parser.add_argument('--model', required=True,
                        help='API model name')
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--concurrency', type=int, default=32,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max items to process (0 = all)')
    parser.add_argument('--thinking-budget', type=int, default=1024,
                        help='Max thinking tokens for phase-2 compress (shorter = faster, cheaper)')
    parser.add_argument('--tokenizer', default='Qwen/Qwen3.5-4B',
                        help='HF/ModelScope tokenizer id for sparse token-ratio probe')
    parser.add_argument('--tokenize-every', type=int, default=1000,
                        help='Tokenize one sample every N writes; others get tokens=0')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done_sample_ids = load_done_sample_ids(args.output)
    # Group done sample ids by base item id so each worker only sees its slice.
    done_per_item: Dict[str, Set[str]] = {}
    for sid in done_sample_ids:
        if '__' in sid:
            base = sid.rsplit('__', 1)[0]
            done_per_item.setdefault(base, set()).add(sid)
    sys.stderr.write(
        f'Resume: {len(done_sample_ids)} samples on disk across '
        f'{len(done_per_item)} items.\n')

    api = OpenAI(model=args.model, api_key=args.api_key, base_url=args.base_url)

    from modelscope import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def iter_pending() -> Iterator[Dict[str, Any]]:
        if args.input:
            source_iter = iter_input(args.input)
        else:
            source_iter = iter_dataset_py(
                total=args.total or None,
                load_from_cache_file=not args.no_cache,
            )
        emitted = 0
        for it in source_iter:
            iid = it.get('id')
            if not iid:
                sys.stderr.write('[skip] row missing "id" field\n')
                continue
            if args.limit > 0 and emitted >= args.limit:
                return
            yield it
            emitted += 1

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    items_done = 0
    items_failed = 0
    samples_emitted = 0
    pbar = tqdm(desc='condense', unit='item', dynamic_ncols=True)

    items_iter = iter_pending()
    in_flight: Dict[Any, str] = {}
    # Sliding window: keep ~2x concurrency tasks queued so the pool never starves.
    window = max(args.concurrency * 2, args.concurrency + 4)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            exhausted = False
            while True:
                while not exhausted and len(in_flight) < window:
                    try:
                        it = next(items_iter)
                    except StopIteration:
                        exhausted = True
                        break
                    iid = it['id']
                    fut = ex.submit(
                        process_item, api, it, done_per_item.get(iid),
                        args.thinking_budget,
                    )
                    in_flight[fut] = iid
                if not in_flight:
                    break
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    iid = in_flight.pop(fut)
                    try:
                        samples = fut.result()
                    except Exception as exc:
                        sys.stderr.write(f'[item {iid}] crashed: {exc}\n')
                        items_failed += 1
                        pbar.update(1)
                        continue
                    if not samples:
                        items_failed += 1
                        pbar.update(1)
                        continue
                    with write_lock:
                        for s in samples:
                            src = s.pop('__src', '')
                            cmp = s.pop('__cmp', '')
                            samples_emitted += 1
                            if (samples_emitted - 1) % args.tokenize_every == 0:
                                s['original_tokens'] = len(tokenizer(src).input_ids)
                                s['compressed_tokens'] = len(tokenizer(cmp).input_ids)
                            out_fh.write(json.dumps(s, ensure_ascii=False) + '\n')
                        out_fh.flush()
                    items_done += 1
                    pbar.set_postfix(
                        done=items_done, failed=items_failed,
                        samples=samples_emitted, refresh=False,
                    )
                    pbar.update(1)
    finally:
        out_fh.close()
        pbar.close()

    sys.stderr.write(
        f'Done. items_done={items_done}, samples={samples_emitted}, '
        f'failed={items_failed}\n')


if __name__ == '__main__':
    main()
