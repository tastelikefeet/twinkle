import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from modelscope import dataset_snapshot_download

from twinkle.dataset import Dataset, DatasetMeta

MUSIQUE_REPO = 'ms://voidful/MuSiQue'
# 仓库内仅包含这两份原始 JSONL，没有 HF datasets 元数据，
# 因此不能直接用 ``DatasetMeta(repo_id)`` 加载，只能落本地后再读。
MUSIQUE_RAW_FILES = (
    'musique_full_v1.0_train.jsonl',  # 含 answerable + 对抗式不可答样本
    'musique_ans_v1.0_train.jsonl',   # 仅 answerable，2/3/4-hop 全量
)


def _musique_row_to_passages(row: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """把单条 MuSiQue 样本 flatten 成多个 passage row，供压缩 SFT 单 passage 训练。"""
    parent_id = str(row.get('id', ''))
    # id 形如 ``2hop__482757_12019``，前缀直接当作 hop 类型
    hop_type = parent_id.split('__', 1)[0] if '__' in parent_id else ''
    question = row.get('question', '') or ''

    primary = (row.get('answer') or '').strip()
    answers = [primary] if primary else []
    for alias in (row.get('answer_aliases') or []):
        a = (alias or '').strip()
        if a and a not in answers:
            answers.append(a)

    for idx, p in enumerate(row.get('paragraphs') or []):
        passage = (p.get('paragraph_text') or '').strip()
        if not passage:
            continue
        yield {
            'id': f'{parent_id}__{idx}',
            'row_id': parent_id,
            'source': 'musique',
            'type': hop_type,
            'paragraph_idx': idx,
            'question': question,
            'title': p.get('title', '') or '',
            'passage': passage,
            'is_supporting': bool(p.get('is_supporting')),
            'answer': primary,
            'answers': answers,
        }


def prepare_musique_dataset(
    local_dir: Optional[str] = None,
    file_name: str = 'musique_ans_v1.0_train.jsonl',
    cache_path: Optional[str] = None,
) -> str:
    """把 MuSiQue 落本地后 flatten 成 passage-per-row JSONL，返回 JSONL 路径。

    Args:
        local_dir: 已下载好的 MuSiQue 目录；为 ``None`` 时调用
            ``dataset_snapshot_download`` 自动拉取。
        file_name: 选用哪份原始 JSONL，``_ans_`` 只含可答样本，
            ``_full_`` 还混入了对抗式不可答样本（会被自动过滤掉）。
        cache_path: 输出路径，默认放在 ``local_dir`` 下，stem 形如
            ``passages_musique_ans_v1.0_train.jsonl``。
    """
    if local_dir is None:
        local_dir = dataset_snapshot_download(MUSIQUE_REPO)
    local_dir = Path(local_dir)
    src = local_dir / file_name
    if not src.is_file():
        raise FileNotFoundError(
            f'MuSiQue raw file not found: {src} (expected one of {MUSIQUE_RAW_FILES})')

    if cache_path is None:
        cache_path = str(local_dir / f'passages_{Path(file_name).stem}.jsonl')
    cache = Path(cache_path)
    if cache.is_file() and cache.stat().st_size > 0:
        return str(cache)

    is_ans = '_ans_' in file_name
    tmp = cache.with_suffix('.jsonl.tmp')
    with src.open('r', encoding='utf-8') as fin, tmp.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not is_ans and not row.get('answerable', True):
                continue
            for passage_row in _musique_row_to_passages(row):
                fout.write(json.dumps(passage_row, ensure_ascii=False) + '\n')
    os.replace(tmp, cache)
    return str(cache)


dataset = Dataset()
dataset.add_dataset(DatasetMeta(prepare_musique_dataset()))
