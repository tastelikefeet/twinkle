"""Extract HotpotQA ``id`` values for rollouts the policy got wrong.

Input
-----
``rollout_trace.jsonl`` at the repository root (or ``--trace`` path).
Each line is one rollout record with ``full_chunks``, ``rewards.f1`` and
``tool_call_count`` fields (see :mod:`cookbook.rl.short_math_grpo_with_tools`).

Output
------
``cookbook/rl/wrong_ids.txt`` (or ``--out`` path) — one HotpotQA id per
line, sorted and de-duplicated.

The training script reads this file through the ``WRONG_IDS_FILE`` env var
and treats it as a whitelist: any row whose ``id`` is NOT in the set is
dropped by :meth:`HotpotQAProcessor.preprocess`, so the next training run
retries exactly the subset the policy failed on.

Wrong definition
----------------
A question is "wrong" if ANY of its rollouts scored ``rewards.f1 < 0.5``.
This captures both
  * ``tcc == 0`` (answered short, missed the needle), and
  * ``tcc > 0`` (called the tool but still failed),
which jointly constitute >99% of wrong rollouts in the current run.

Question-text → HotpotQA-id matching
------------------------------------
The trace only stores the full user prompt
("Question: <q>\\n\\nContext:\\n\\n[1] ..."), not the original id.  We
strip the ``"Question: "`` prefix and the ``"\\n\\nContext:"`` suffix to
recover ``q``, then look it up in a ``q → id`` dict built from
``datasets.load_dataset('hotpot_qa', 'fullwiki', split='train')``.
Duplicate questions in HotpotQA are rare but do exist; when encountered we
keep the first id (matches HF's stable row order).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


F1_THRESHOLD = 0.5


def _strip_question(user_content: str) -> Optional[str]:
    """Recover the raw HotpotQA ``question`` string from the user prompt.

    Returns ``None`` if the expected ``Question: ... \\n\\nContext:`` layout
    is not found (e.g. a non-HotpotQA record has leaked into the trace).
    """
    if not isinstance(user_content, str):
        return None
    if not user_content.startswith('Question: '):
        return None
    body = user_content[len('Question: '):]
    ctx = body.find('\n\nContext:')
    if ctx < 0:
        return None
    return body[:ctx].strip()


def _first_user_content(record: dict) -> Optional[str]:
    for c in record.get('full_chunks') or []:
        if c.get('role') == 'user':
            cont = c.get('content')
            if isinstance(cont, str):
                return cont
    return None


def _iter_trace(path: Path) -> Iterable[dict]:
    with path.open('r', encoding='utf-8') as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue


def collect_wrong_questions(trace_path: Path) -> Tuple[Set[str], Dict[str, dict]]:
    """Return ``(wrong_questions, stats)``.

    ``wrong_questions`` is a set of raw HotpotQA ``question`` strings whose
    trace contained at least one rollout with ``f1 < F1_THRESHOLD``.
    ``stats`` holds counters for reporting.
    """
    # Index rollouts by question text.  Use a dict-of-list instead of
    # ``defaultdict(list)`` so we can report the exact set of seen questions
    # at the end (orderless set comparison with HF ids).
    by_q: Dict[str, List[dict]] = defaultdict(list)
    unparsed = 0
    total = 0
    for rec in _iter_trace(trace_path):
        total += 1
        user = _first_user_content(rec)
        q = _strip_question(user) if user is not None else None
        if q is None:
            unparsed += 1
            continue
        by_q[q].append(rec)

    wrong: Set[str] = set()
    wrong_no_tool = 0
    wrong_with_tool = 0
    groups_all_right = 0
    groups_with_wrong = 0
    for q, rolls in by_q.items():
        any_wrong = False
        for r in rolls:
            f1 = ((r.get('rewards') or {}).get('f1') or 0.0)
            if f1 < F1_THRESHOLD:
                any_wrong = True
                if (r.get('tool_call_count') or 0) > 0:
                    wrong_with_tool += 1
                else:
                    wrong_no_tool += 1
        if any_wrong:
            wrong.add(q)
            groups_with_wrong += 1
        else:
            groups_all_right += 1

    stats = {
        'total_rollouts': total,
        'unparsed_rollouts': unparsed,
        'unique_questions': len(by_q),
        'groups_with_wrong': groups_with_wrong,
        'groups_all_right': groups_all_right,
        'wrong_no_tool': wrong_no_tool,
        'wrong_with_tool': wrong_with_tool,
    }
    return wrong, stats


def build_question_to_id(subset_name: str, split: str) -> Dict[str, str]:
    """Load HotpotQA and return a ``question → id`` dict.

    Tries three progressively hackier fallbacks so the script works on
    machines with / without network + Hub credentials:

    1. ``datasets.load_dataset('hotpot_qa', subset, split)`` — normal path.
    2. On failure, scan the HF datasets cache for
       ``hotpotqa___hotpot_qa/<subset>/**/hotpot_qa-<split>-*.arrow``
       and concat them via :func:`datasets.Dataset.from_file`.
    3. If that also fails, read the arrow files with ``pyarrow.ipc``
       directly and pull only the ``id`` / ``question`` columns.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Need `datasets` installed to reverse-lookup HotpotQA ids. "
            "Run: pip install datasets") from exc

    # Attempt 1: normal Hub path.
    try:
        ds = load_dataset('hotpot_qa', subset_name, split=split)
        return _index_by_question(ds)
    except Exception as exc1:
        print(f'[warn] load_dataset failed ({exc1.__class__.__name__}): {exc1}',
              file=sys.stderr)

    # Attempts 2+3: scan cache for the raw shard files.
    cache_root = Path(os.environ.get('HF_DATASETS_CACHE',
                                     os.path.expanduser('~/.cache/huggingface/datasets')))
    pattern = f'hotpot_qa-{split}-*.arrow'
    shard_paths: List[Path] = sorted(
        cache_root.glob(f'hotpotqa___hotpot_qa/{subset_name}/*/*/{pattern}'))
    if not shard_paths:
        raise SystemExit(
            f'[error] could not find cached HotpotQA shards matching '
            f'{pattern} under {cache_root}.  Set HF_DATASETS_CACHE or run '
            f'`datasets.load_dataset(\'hotpot_qa\', \'{subset_name}\')` '
            f'once online first.')
    print(f'[info] found {len(shard_paths)} cached shard(s), reading '
          f'directly', file=sys.stderr)

    # Attempt 2: datasets.Dataset.from_file (preserves schema).
    try:
        from datasets import Dataset as _DS, concatenate_datasets  # type: ignore
        parts = [_DS.from_file(str(p)) for p in shard_paths]
        ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        return _index_by_question(ds)
    except Exception as exc2:
        print(f'[warn] Dataset.from_file failed ({exc2.__class__.__name__}): '
              f'{exc2}', file=sys.stderr)

    # Attempt 3: pyarrow direct read, column-sliced.
    import pyarrow as pa
    q_to_id: Dict[str, str] = {}
    dup = 0
    for p in shard_paths:
        with pa.memory_map(str(p), 'r') as src:
            reader = pa.ipc.open_stream(src) if p.suffix == '.arrows' \
                else pa.ipc.open_file(src)
            tbl = reader.read_all().select(['id', 'question'])
            ids = tbl.column('id').to_pylist()
            qs = tbl.column('question').to_pylist()
            for rid, q in zip(ids, qs):
                q = (q or '').strip()
                if not q or not rid:
                    continue
                if q in q_to_id:
                    dup += 1
                    continue
                q_to_id[q] = rid
    if dup:
        print(f'[info] {dup} duplicate question strings; kept first id',
              file=sys.stderr)
    return q_to_id


def _index_by_question(ds) -> Dict[str, str]:
    """Helper: build ``question → id`` from a HF :class:`Dataset`."""
    q_to_id: Dict[str, str] = {}
    dup = 0
    for row in ds:
        q = (row.get('question') or '').strip()
        rid = row.get('id')
        if not q or not rid:
            continue
        if q in q_to_id:
            dup += 1
            continue  # keep first occurrence
        q_to_id[q] = rid
    if dup:
        print(f'[info] HotpotQA had {dup} duplicate question strings; '
              f'kept first id for each', file=sys.stderr)
    return q_to_id


def main() -> int:
    here = Path(__file__).resolve().parent
    repo = here.parent.parent
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--trace', type=Path,
                        default=repo / 'rollout_trace.jsonl')
    parser.add_argument('--out', type=Path,
                        default=here / 'wrong_ids.txt')
    parser.add_argument('--subset', default='fullwiki')
    parser.add_argument('--split', default='train')
    args = parser.parse_args()

    if not args.trace.exists():
        print(f'[error] trace file not found: {args.trace}', file=sys.stderr)
        return 2

    print(f'[step 1/3] scanning {args.trace} ...')
    wrong_qs, stats = collect_wrong_questions(args.trace)
    for k, v in stats.items():
        print(f'    {k:24s} = {v}')
    if not wrong_qs:
        print('[warn] no wrong rollouts found; nothing to write', file=sys.stderr)
        return 1

    print(f'[step 2/3] loading HotpotQA {args.subset}/{args.split} '
          f'to map question → id ...')
    q_to_id = build_question_to_id(args.subset, args.split)
    print(f'    HotpotQA questions loaded: {len(q_to_id)}')

    matched: Set[str] = set()
    unmatched: List[str] = []
    for q in wrong_qs:
        rid = q_to_id.get(q)
        if rid is None:
            # Fallback: try trimming trailing whitespace variants (HF sometimes
            # stores a trailing space that survives into the prompt).
            rid = q_to_id.get(q.rstrip()) or q_to_id.get(q.strip())
        if rid is None:
            unmatched.append(q)
        else:
            matched.add(rid)

    print(f'[step 3/3] writing {len(matched)} ids to {args.out} ...')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', encoding='utf-8') as fh:
        for rid in sorted(matched):
            fh.write(rid + '\n')

    print(f'    matched   = {len(matched)}')
    print(f'    unmatched = {len(unmatched)}  (not written)')
    if unmatched:
        # Show at most 3 for sanity check — too many unmatched signals
        # either a subset-name mismatch or a post-hoc change to the prompt
        # formatting that broke our prefix stripping.
        for q in unmatched[:3]:
            print(f'      e.g. {q[:90]!r}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
