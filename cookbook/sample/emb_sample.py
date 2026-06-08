"""Embedding quality validation: compress (query, cot) pairs via vLLM condenser,
extract embeddings via TransformersModel.forward_only(task='embedding'),
report cosine similarity.

Covers three domains: basic math, code logic, open-ended reasoning.

Architecture (2 GPUs):
  - GPU 0: vLLM condenser (compression)
  - GPU 1: TransformersModel (embedding, same path as training)

Launch:
    python cookbook/sample/emb_sample.py
    EMB_MODEL=./output/embedding_lora_transformers/step_16000 python cookbook/sample/emb_sample.py
"""
import os
import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.loss import InfonceLoss
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template

logger = get_logger()

# -- Config -------------------------------------------------------------------
CONDENSE_MODEL_ID = os.environ.get('CONDENSE_MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
EMB_MODEL_ID = os.environ.get('EMB_MODEL', 'output/embedding_lora_transformers/step_16000')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))
EMB_GPUS = int(os.environ.get('EMB_GPUS', 1))
EMB_MAX_LENGTH = 8192

# -- Prompts (aligned with train_embedding_full_ddp.py) -----------------------
COMPRESS_SYSTEM = """\
You are a compression and summary assistant. For the (query, source) pair, emit a Markdown \
answer with TWO sections, designed to pair with the `extract_compressed` tool: \
the reader absorbs `## Summary` directly, then calls `extract_compressed` \
on any topic-key listed under `## More` to recover its \
fuller content.

  `## Summary`               — extreme-density text the reader reads directly.
  `## More` — a topic index whose keys are valid arguments \
to `extract_compressed` for recovering material not captured inline.

Together the two sections must form a COMPLETE, NON-DISTORTING inventory of the \
source for the query — nothing essential lost, nothing implied that the source \
does not support. NO preamble, NO meta-commentary, NO code fences wrapping the \
whole output.

Output skeleton:

## Summary
Topic: <what the source is about + scope, one line>
<dense body answering the query>

## More
- <topic-key>: <one-line hint of what is revealed when expanded>
- ...

Now begin.\
"""

COMPRESS_USER = (
    'Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts.\n\n'
    '## Query (ordering hint only)\n{query}\n\n'
    '## Passage\n{text}')

EMBED_QUERY_Q = (
    'What problem does this passage address, and what skill or method is needed? '
    'Compress into a retrieval-friendly need description.')
EMBED_QUERY_COT = (
    'Extract the reusable skill: trigger conditions, key steps, and expected output. '
    'Compress into a standardized procedure for retrieval.')

# =============================================================================
# Test pairs: (domain, query, cot_reasoning)
# =============================================================================
TEST_PAIRS: List[Dict[str, str]] = [
    # --- Math ---
    {
        'domain': 'math-arithmetic',
        'query': '计算 (17 × 23) + (89 - 45) 的结果。',
        'cot': (
            '分步计算：第一步 17×23=391，第二步 89-45=44，第三步 391+44=435。'
            '验证：17×23 = 17×20+17×3 = 340+51 = 391 ✓；89-45=44 ✓；391+44=435 ✓。'
            '最终答案为 435。'),
    },
    {
        'domain': 'math-algebra',
        'query': '解方程 2x² - 5x - 3 = 0，求所有实数解。',
        'cot': (
            '使用求根公式 x = (-b ± √(b²-4ac)) / (2a)，其中 a=2, b=-5, c=-3。'
            '判别式 Δ = 25 - 4×2×(-3) = 25+24 = 49 > 0，有两个不等实根。'
            '√49 = 7，x₁ = (5+7)/4 = 3，x₂ = (5-7)/4 = -1/2。'
            '验证：2(3)²-5(3)-3 = 18-15-3 = 0 ✓；2(1/4)-5(-1/2)-3 = 1/2+5/2-3 = 0 ✓。'),
    },
    {
        'domain': 'math-probability',
        'query': '一个袋子里有3个红球和5个蓝球，不放回地连续取2个球，求两个都是红球的概率。',
        'cot': (
            '总共8个球，第一次取红球概率 P₁=3/8；取出后剩余7个球其中2红5蓝，'
            '第二次取红球概率 P₂=2/7。两次都取红球概率 P=P₁×P₂=(3/8)×(2/7)=6/56=3/28≈0.107。'
            '也可用组合方法：C(3,2)/C(8,2) = 3/28。'),
    },
    # --- Code ---
    {
        'domain': 'code-sorting',
        'query': '用Python实现归并排序，要求支持自定义比较函数。',
        'cot': (
            'def merge_sort(arr, key=None):\n'
            '    if len(arr) <= 1: return arr\n'
            '    mid = len(arr) // 2\n'
            '    left = merge_sort(arr[:mid], key)\n'
            '    right = merge_sort(arr[mid:], key)\n'
            '    return merge(left, right, key)\n\n'
            'def merge(left, right, key):\n'
            '    result, i, j = [], 0, 0\n'
            '    while i < len(left) and j < len(right):\n'
            '        lv = key(left[i]) if key else left[i]\n'
            '        rv = key(right[j]) if key else right[j]\n'
            '        if lv <= rv:\n'
            '            result.append(left[i]); i += 1\n'
            '        else:\n'
            '            result.append(right[j]); j += 1\n'
            '    result.extend(left[i:]); result.extend(right[j:])\n'
            '    return result\n\n'
            '时间复杂度 O(n log n)，空间复杂度 O(n)，稳定排序。'
            '支持 key=lambda x: x.name 等自定义比较。'),
    },
    {
        'domain': 'code-design-pattern',
        'query': '如何实现一个线程安全的单例模式？给出Python和Java两种实现。',
        'cot': (
            'Python 方案一：模块级变量天然单例。方案二：使用 __new__ + threading.Lock:\n'
            'class Singleton:\n'
            '    _instance = None\n'
            '    _lock = threading.Lock()\n'
            '    def __new__(cls):\n'
            '        if cls._instance is None:\n'
            '            with cls._lock:\n'
            '                if cls._instance is None:\n'
            '                    cls._instance = super().__new__(cls)\n'
            '        return cls._instance\n\n'
            'Java 方案：双重检查锁定 + volatile:\n'
            'public class Singleton {\n'
            '    private static volatile Singleton instance;\n'
            '    private Singleton() {}\n'
            '    public static Singleton getInstance() {\n'
            '        if (instance == null) {\n'
            '            synchronized (Singleton.class) {\n'
            '                if (instance == null) instance = new Singleton();\n'
            '            }\n'
            '        }\n'
            '        return instance;\n'
            '    }\n'
            '}\n\n'
            '关键：双重检查避免每次加锁开销；volatile 防止指令重排序导致半初始化对象泄露。'),
    },
    {
        'domain': 'code-debugging',
        'query': 'Python中 list.sort() 和 sorted() 有什么区别？什么时候用哪个？',
        'cot': (
            'list.sort() 是原地排序（in-place），返回 None，修改原列表，节省内存。'
            'sorted() 返回新列表，不修改原数据，可作用于任何可迭代对象（tuple、generator等）。'
            '使用场景：需要保留原列表时用 sorted()；大列表且不需要原顺序时用 .sort() 避免额外内存。'
            '性能：两者底层都用 Timsort，O(n log n)；.sort() 少一次列表拷贝。'
            '注意：sorted(dict) 返回排序后的 key 列表，不是 key-value pairs。'),
    },
    # --- Open-ended ---
    {
        'domain': 'open-philosophy',
        'query': '如何看待"技术是中性的"这一观点？',
        'cot': (
            '这一观点认为技术本身无善恶，善恶取决于使用者。支持论据：刀可以做手术也可以伤人。'
            '反对论据：技术设计内嵌价值取向——社交媒体的推荐算法天然倾向于放大极端内容以增加停留时间；'
            '核武器的存在本身改变了国际政治博弈结构，无论是否使用。'
            '中间立场（Winner, Feenberg）：技术是"政治性的人造物"，其设计反映了创造者的意图和社会结构，'
            '但使用者仍有一定的重新诠释空间。'
            '结论：纯粹的技术中性论过于简化；应关注技术设计中的默认设定、激励结构和权力不对称。'),
    },
    {
        'domain': 'open-education',
        'query': '为什么很多人觉得学习数学没有用？如何改变这种认知？',
        'cot': (
            '原因分析：1）教学脱离应用场景——解方程但不知道何时需要；'
            '2）反馈周期长——数学能力的回报在编程、金融、科研等领域才显现；'
            '3）恐惧循环——早期挫折→回避→更大差距→更强恐惧。'
            '改变策略：1）项目驱动教学（用统计分析真实数据、用几何设计3D模型）；'
            '2）展示隐性应用（推荐算法=线性代数、保险定价=概率论、路径规划=图论）；'
            '3）成长型思维训练——强调"还不会"而非"学不会"；'
            '4）分层目标——让学生看到每一步小进展的价值。'
            '核心：数学不是"有用/没用"的二元判断，而是一种结构化思维训练。'),
    },
    {
        'domain': 'open-creative',
        'query': '如果让你设计一个火星城市，最重要的三个设计原则是什么？',
        'cot': (
            '原则一：冗余生命维持——火星无磁场无大气，任何单点故障都致命。'
            '所有氧气/水/能源系统至少三重冗余，居住模块物理隔离且可独立密封。'
            '原则二：心理可持续性——封闭空间+通信延迟14-24分钟+无法快速撤离，'
            '需要大面积模拟自然光的穹顶、隔音私人空间、社区活动中心、'
            '与地球的异步社交系统（类似论坛而非即时通讯）。'
            '原则三：就地资源利用（ISRU）——从火星土壤提取水和金属，'
            '用二氧化碳大气制造燃料（Sabatier反应），3D打印建筑结构。'
            '减少对地球供应链的依赖是长期可持续的唯一路径。'),
    },
]

# Negative control pairs (semantically unrelated query-cot)
NEGATIVE_PAIRS: List[Dict[str, str]] = [
    {
        'domain': 'negative-math-vs-code',
        'query': '计算 sin(π/6) + cos(π/3) 的精确值。',
        'cot': (
            'def merge_sort(arr, key=None):\n'
            '    if len(arr) <= 1: return arr\n'
            '    mid = len(arr) // 2\n'
            '    left = merge_sort(arr[:mid], key)\n'
            '    right = merge_sort(arr[mid:], key)\n'
            '    return merge(left, right, key)\n\n'
            '时间复杂度 O(n log n)，空间复杂度 O(n)，稳定排序。'),
    },
    {
        'domain': 'negative-code-vs-philosophy',
        'query': '如何实现一个LRU缓存？',
        'cot': (
            '技术中性论认为技术本身无善恶，善恶取决于使用者。支持论据：刀可以做手术也可以伤人。'
            '反对论据：技术设计内嵌价值取向——社交媒体推荐算法天然倾向放大极端内容。'
            '结论：纯粹技术中性论过于简化。'),
    },
]


# =============================================================================
# Compression via vLLM
# =============================================================================

def compress_texts(sampler, texts: List[str], query_hint: str) -> List[str]:
    """Compress a list of texts using the vLLM condenser."""
    prompts = []
    for text in texts:
        user_msg = COMPRESS_USER.format(query=query_hint, text=text)
        prompts.append({'messages': [
            {'role': 'system', 'content': COMPRESS_SYSTEM},
            {'role': 'user', 'content': user_msg},
        ]})

    params = SamplingParams(max_tokens=8192, temperature=0.2, top_p=0.5, num_samples=1)
    responses = sampler.sample(prompts, params)

    results = []
    for resp in responses:
        seq = resp.sequences[0] if resp and resp.sequences else None
        text = ''
        if seq and seq.decoded:
            text = seq.decoded
            text = re.sub(r'<\|[^|]+\|>', '', text).rstrip()
        results.append(text)
    return results


# =============================================================================
# Embedding extraction (TransformersModel, same path as training)
# =============================================================================

def _build_features(texts: List[str], template: Template, role: str) -> List[Dict[str, Any]]:
    """Encode texts into embedding features, matching _get_first_feature in training."""
    features = []
    for text in texts:
        if not text.strip():
            continue
        if role == 'anchor':
            feat = template.encode({'messages': [
                {'role': 'user', 'content': text},
                {'role': 'assistant', 'content': 'Match the correct response here.'},
            ]})
            feat['labels'] = [1]
        else:
            feat = template.encode({'messages': [
                {'role': 'user', 'content': 'Match the correct query here.'},
                {'role': 'assistant', 'content': text},
            ]})
            feat['labels'] = [0]
        features.append(feat)
    return features


def get_embeddings(model: TransformersModel, template: Template,
                   texts: List[str], role: str = 'anchor') -> torch.Tensor:
    """Get embeddings via forward_only(task='embedding'), same code path as training."""
    features = _build_features(texts, template, role)
    if not features:
        return torch.zeros(0)
    outputs = model.forward_only(inputs=features, task='embedding', return_logits=True)
    return outputs['embeddings']


# =============================================================================
# Main
# =============================================================================

def main():
    NUM_GPUS = SAMPLER_GPUS + EMB_GPUS

    # 1. Initialize Twinkle with both device groups
    device_groups = [
        DeviceGroup(name='sampler',
                    ranks=list(range(SAMPLER_GPUS)),
                    device_type='GPU',
                    gpus_per_worker=SAMPLER_GPUS),
        DeviceGroup(name='emb_model',
                    ranks=list(range(SAMPLER_GPUS, NUM_GPUS)),
                    device_type='GPU'),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    emb_mesh = DeviceMesh.from_sizes(world_size=EMB_GPUS, dp_size=EMB_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)

    # 2. vLLM condenser sampler
    sampler = vLLMSampler(
        model_id=CONDENSE_MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32768,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(
        'Qwen3_5Template', model_id=CONDENSE_MODEL_ID,
        enable_thinking=False, max_length=32768)

    # 3. Embedding model (same as training: TransformersModel + InputProcessor + InfonceLoss)
    emb_model = TransformersModel(
        model_id=EMB_MODEL_ID,
        device_mesh=emb_mesh,
        remote_group='emb_model',
    )
    emb_model.set_processor(InputProcessor)
    emb_model.set_loss(InfonceLoss, temperature=0.03, use_batch=True)
    emb_template = Template(model_id=EMB_MODEL_ID, max_length=EMB_MAX_LENGTH, enable_thinking=False)

    logger.info(get_device_placement())

    # 4. Compress all pairs
    all_pairs = TEST_PAIRS + NEGATIVE_PAIRS
    queries = [p['query'] for p in all_pairs]
    cots = [p['cot'] for p in all_pairs]

    logger.info(f'Compressing {len(queries)} queries ...')
    compressed_queries = compress_texts(sampler, queries, EMBED_QUERY_Q)
    logger.info(f'Compressing {len(cots)} CoTs ...')
    compressed_cots = compress_texts(sampler, cots, EMBED_QUERY_COT)

    # Print compression results
    for i, pair in enumerate(all_pairs):
        qc, cc = compressed_queries[i], compressed_cots[i]
        logger.info(
            f'\n{"=" * 70}\n'
            f'[{pair["domain"]}] Query ({len(pair["query"])}→{len(qc)} chars):\n'
            f'  Raw: {pair["query"][:80]}...\n'
            f'  Compressed: {qc[:120]}...\n'
            f'CoT ({len(pair["cot"])}→{len(cc)} chars):\n'
            f'  Compressed: {cc[:120]}...')

    # 5. Get embeddings via TransformersModel.forward_only(task='embedding')
    logger.info('Computing query embeddings ...')
    q_embs = get_embeddings(emb_model, emb_template, compressed_queries, role='anchor')
    logger.info('Computing CoT embeddings ...')
    c_embs = get_embeddings(emb_model, emb_template, compressed_cots, role='positive')

    logger.info('Computing raw query embeddings (no compression) ...')
    raw_q_embs = get_embeddings(emb_model, emb_template, queries, role='anchor')
    logger.info('Computing raw CoT embeddings (no compression) ...')
    raw_c_embs = get_embeddings(emb_model, emb_template, cots, role='positive')

    # 6. Compute similarities
    n_positive = len(TEST_PAIRS)
    n_negative = len(NEGATIVE_PAIRS)

    logger.info(f'\n{"=" * 70}')
    logger.info('RESULTS: Cosine Similarity (compressed query ↔ compressed CoT)')
    logger.info(f'{"=" * 70}')
    logger.info(f'{"Domain":<30} {"Compressed":>12} {"Raw":>12} {"Δ":>8}')
    logger.info('-' * 70)

    pos_sims_compressed, pos_sims_raw = [], []
    neg_sims_compressed, neg_sims_raw = [], []

    for i, pair in enumerate(all_pairs):
        sim_c = F.cosine_similarity(q_embs[i:i+1], c_embs[i:i+1]).item()
        sim_r = F.cosine_similarity(raw_q_embs[i:i+1], raw_c_embs[i:i+1]).item()
        delta = sim_c - sim_r
        marker = '✓' if i < n_positive else '✗'
        logger.info(f'  {marker} {pair["domain"]:<28} {sim_c:>10.4f}   {sim_r:>10.4f}   {delta:>+.4f}')

        if i < n_positive:
            pos_sims_compressed.append(sim_c)
            pos_sims_raw.append(sim_r)
        else:
            neg_sims_compressed.append(sim_c)
            neg_sims_raw.append(sim_r)

    # Summary statistics
    avg_pos_c = sum(pos_sims_compressed) / len(pos_sims_compressed)
    avg_pos_r = sum(pos_sims_raw) / len(pos_sims_raw)
    avg_neg_c = sum(neg_sims_compressed) / len(neg_sims_compressed) if neg_sims_compressed else 0
    avg_neg_r = sum(neg_sims_raw) / len(neg_sims_raw) if neg_sims_raw else 0

    logger.info(f'\n{"=" * 70}')
    logger.info('SUMMARY')
    logger.info(f'  Positive pairs (matched):   compressed={avg_pos_c:.4f}  raw={avg_pos_r:.4f}')
    logger.info(f'  Negative pairs (mismatched): compressed={avg_neg_c:.4f}  raw={avg_neg_r:.4f}')
    logger.info(f'  Margin (pos - neg):          compressed={avg_pos_c - avg_neg_c:.4f}  '
                f'raw={avg_pos_r - avg_neg_r:.4f}')
    logger.info(f'{"=" * 70}')

    # 6. Cross-similarity matrix for positive pairs
    logger.info('\nCross-similarity matrix (compressed, positive pairs only):')
    logger.info(f'  {"":>20}')
    for i in range(n_positive):
        logger.info(f' {TEST_PAIRS[i]["domain"][:8]:>9}')
    logger.info('')

    cross_sim = F.cosine_similarity(
        q_embs[:n_positive].unsqueeze(1),
        c_embs[:n_positive].unsqueeze(0), dim=2)
    for i in range(n_positive):
        row = f'  {TEST_PAIRS[i]["domain"][:20]:<20}'
        for j in range(n_positive):
            val = cross_sim[i, j].item()
            mark = ' *' if i == j else '  '
            row += f' {val:>7.4f}{mark}'
        logger.info(row)

    logger.info('\n* = diagonal (true positive pair)')
    logger.info('Done.')


if __name__ == '__main__':
    main()
