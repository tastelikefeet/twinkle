"""Re-annotate HotpotQA ground truth using a super-LLM to ensure correctness.

The original HotpotQA dataset has annotation issues:
  - GT doesn't match the question type (asks "where", GT gives a name)
  - Partial/incomplete answers for multi-hop questions
  - Single form when multiple valid forms exist (e.g. "2" vs "two")

This script:
  1. Loads HotpotQA fullwiki train split, stratified 3000 per level.
  2. Force-includes all IDs from wrong_ids.txt (the 340 hard cases).
  3. For each row, sends question + full context + original GT to a super-LLM.
  4. The LLM verifies/corrects the GT and returns a list of acceptable answers.
  5. Outputs JSONL with the corrected ground truth.

Run:
    python reannotate_groundtruth.py \
        --model qwen-max --api-key $OPENAI_API_KEY \
        --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
        --output hotpotqa_reannotated.jsonl --concurrency 16
"""
import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.protocol.openai import OpenAI


VERIFY_SYSTEM = """You are a dataset quality auditor for a multi-hop QA benchmark (HotpotQA).

Your job: given a Question, supporting Context passages, and the dataset's Original Answer, determine ALL correct short answers.

Rules:
1. Read the context carefully. The answer MUST be supported by the given passages.
2. If the Original Answer is correct, keep it. If it is wrong or incomplete, fix it.
3. Return ALL acceptable surface forms as a JSON list. Include:
   - The canonical answer
   - Common abbreviations (e.g. "New York City", "NYC", "New York")
   - Numeric variants (e.g. "2", "two", "2.0")
   - Name variants (e.g. "J.K. Rowling", "Joanne Rowling", "J. K. Rowling")
   - With/without titles (e.g. "Dr. Smith", "Smith")
   - Different date formats if applicable (e.g. "July 4, 1776", "4 July 1776")
4. Each answer in the list should be SHORT (a name, entity, number, date, or yes/no).
5. If the question cannot be answered from the given context at all, return ["UNANSWERABLE"].
6. Do NOT hallucinate. Every answer must be grounded in the provided passages.
7. For yes/no questions, return ["yes"] or ["no"] (lowercase).

Output format (JSON only, no markdown fence, no explanation):
{"answers": ["answer1", "answer2", ...], "reasoning": "one-sentence explanation of your judgment"}"""

VERIFY_USER = """## Question
{question}

## Original Answer (may be wrong)
{original_answer}

## Supporting Passages
{context}

## Task
Verify whether the Original Answer correctly answers the Question based on the passages above.
Return a JSON object with:
- "answers": a list of ALL acceptable short answer forms (if original is wrong, give the correct one(s))
- "reasoning": one sentence explaining your judgment (e.g. "Original is correct", "Original is wrong because X, correct answer is Y")"""


LEVELS: Tuple[str, str, str] = ('easy', 'medium', 'hard')


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


_JSON_RE = re.compile(r'\{[^{}]*"answers"\s*:\s*\[.*?\][^{}]*\}', re.DOTALL)


def _parse_response(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith('```'):
        first_nl = text.find('\n')
        last_fence = text.rfind('```')
        if first_nl != -1 and last_fence > first_nl:
            text = text[first_nl + 1:last_fence].strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and 'answers' in obj:
            return obj
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def verify_answer(
    api: OpenAI, model: str, row: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    question = row['question']
    original_answer = row.get('answer', '') or ''
    context_str = _format_context(row.get('context', {}) or {})

    user_content = VERIFY_USER.format(
        question=question,
        original_answer=original_answer,
        context=context_str)

    trajectory = {
        'messages': [
            {'role': 'system', 'content': VERIFY_SYSTEM},
            {'role': 'user', 'content': user_content},
        ]
    }
    sp = SamplingParams(temperature=0.1, max_tokens=512)

    for attempt in range(3):
        try:
            reply = api(trajectory, sp, extra_body={'enable_thinking': True})
        except Exception as exc:
            sys.stderr.write(f'[verify] {row["id"]}: API error: {exc}\n')
            if attempt < 2:
                continue
            return None

        content = reply.get('content') or ''
        parsed = _parse_response(content)
        if parsed and isinstance(parsed.get('answers'), list) and parsed['answers']:
            answers = [str(a).strip() for a in parsed['answers'] if str(a).strip()]
            if not answers:
                continue
            return {
                'id': row['id'],
                'question': question,
                'original_answer': original_answer,
                'answers': answers,
                'reasoning': parsed.get('reasoning', ''),
                'level': row.get('level', ''),
                'type': row.get('type', ''),
                'context': row.get('context', {}),
                'supporting_facts': row.get('supporting_facts', {}),
            }
        sys.stderr.write(
            f'[verify retry {attempt+1}] {row["id"]}: '
            f'parse failed, content={content[:200]!r}\n')

    sys.stderr.write(f'[verify drop] {row["id"]}: all attempts failed\n')
    return None


def stratified_sample_with_forced(
    ds, per_level: int, forced_ids: frozenset, seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    buckets: Dict[str, List[int]] = {lv: [] for lv in LEVELS}
    forced_indices: List[int] = []
    forced_levels: Dict[str, int] = {lv: 0 for lv in LEVELS}

    for i in range(len(ds)):
        row_id = ds[i]['id']
        level = (ds[i].get('level') or '').strip().lower()
        if row_id in forced_ids:
            forced_indices.append(i)
            if level in forced_levels:
                forced_levels[level] += 1
        elif level in buckets:
            buckets[level].append(i)

    picked_set = set(forced_indices)
    for lv in LEVELS:
        need = max(0, per_level - forced_levels[lv])
        pool = [idx for idx in buckets[lv] if idx not in picked_set]
        if len(pool) < need:
            sys.stderr.write(
                f'Warning: level={lv} has {len(pool)} available, need {need}\n')
            need = len(pool)
        sampled = rng.sample(pool, need)
        picked_set.update(sampled)

    picked = sorted(picked_set)
    rng.shuffle(picked)
    return [ds[int(i)] for i in picked]


def load_done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get('id')
            if rid:
                done.add(rid)
    return done


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key', default=os.environ.get('OPENAI_API_KEY'))
    parser.add_argument('--base-url', default=os.environ.get('OPENAI_BASE_URL'))
    parser.add_argument('--total', type=int, default=9000)
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--wrong-ids', default='cookbook/rl/wrong_ids.txt')
    parser.add_argument('--hf-subset', default='fullwiki')
    parser.add_argument('--hf-split', default='train')
    args = parser.parse_args()

    if args.total % len(LEVELS) != 0:
        raise ValueError(
            f'--total must be divisible by {len(LEVELS)}, got {args.total}')
    per_level = args.total // len(LEVELS)

    forced_ids: frozenset = frozenset()
    if args.wrong_ids and os.path.exists(args.wrong_ids):
        with open(args.wrong_ids, 'r', encoding='utf-8') as fh:
            forced_ids = frozenset(ln.strip() for ln in fh if ln.strip())
        sys.stderr.write(f'Forced IDs loaded: {len(forced_ids)}\n')

    sys.stderr.write(
        f'Loading hotpotqa/hotpot_qa:{args.hf_subset}:{args.hf_split}...\n')
    ds = load_dataset(
        'hotpotqa/hotpot_qa', args.hf_subset, split=args.hf_split)

    rows = stratified_sample_with_forced(
        ds, per_level=per_level, forced_ids=forced_ids, seed=args.seed)
    sys.stderr.write(f'Selected {len(rows)} rows (forced={len(forced_ids)})\n')

    done = load_done_ids(args.output)
    sys.stderr.write(f'Resume: {len(done)} rows already done, skipping.\n')
    pending = [row for row in rows if row['id'] not in done]
    sys.stderr.write(f'Pending: {len(pending)} / {len(rows)}\n')

    api = OpenAI(
        model=args.model, api_key=args.api_key, base_url=args.base_url)

    write_lock = threading.Lock()
    out_fh = open(args.output, 'a', encoding='utf-8')
    rows_done = 0
    rows_failed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futures = {
                ex.submit(verify_answer, api, args.model, row): row['id']
                for row in pending
            }
            for fut in as_completed(futures):
                rid = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    sys.stderr.write(f'[row {rid}] crashed: {exc}\n')
                    rows_failed += 1
                    continue
                if result is None:
                    rows_failed += 1
                    continue
                with write_lock:
                    out_fh.write(
                        json.dumps(result, ensure_ascii=False) + '\n')
                    out_fh.flush()
                rows_done += 1
                if rows_done % 100 == 0:
                    sys.stderr.write(
                        f'[progress] done={rows_done} '
                        f'failed={rows_failed}\n')
    finally:
        out_fh.close()

    sys.stderr.write(
        f'Done. rows_done={rows_done}, failed={rows_failed}, '
        f'total_pending={len(pending)}\n')


if __name__ == '__main__':
    main()
