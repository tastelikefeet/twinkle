"""LoRA embedding training: query ↔ CM-v2-compressed thinking_content (Transformers + Megatron).

Pipeline:
  - 4 GPUs (``sampler`` group) load ``ms://twinkle-kit/Qwen3.5-4B-CM-v2`` via
    :class:`vLLMSampler` and run as a frozen online compressor.
  - 4 GPUs (``model`` group) load the same checkpoint with a LoRA adapter and
    train an embedding head against InfoNCE.
  - Each row from :func:`dataset_think.get_dataset` provides ``(query, cot)``;
    every step compresses ``cot`` through CM-v2 (with the production
    Condenser system+user prompt) and treats ``(query, compressed_cot)`` as
    the anchor/positive pair. In-batch + cross-DP samples become negatives.

Switch ``BACKEND`` between ``'transformers'`` and ``'megatron'``; the rest of
the script is backend-agnostic.

Launch:
    python cookbook/exp/train_embedding_lora_ddp.py
"""
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_device_placement, get_logger
from twinkle.data_format import InputFeature, SamplingParams
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.preprocessor import Preprocessor
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler
from twinkle.template import Template
from twinkle.utils import Platform

# allow importing the sibling dataset_think module without packaging
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_think import get_dataset  # noqa: E402

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

MODEL_ID = os.environ.get('MODEL_ID', 'ms://twinkle-kit/Qwen3.5-4B-CM-v2')
TEMPLATE_NAME = 'Qwen3_5Template'

# -- GPU placement ------------------------------------------------------------
MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 4))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 4))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

# -- Embedding training hyper-params ------------------------------------------
EMB_MAX_LENGTH = 4096
HARD_NEGATIVES = None  # rely on in-batch negatives only
TEMPERATURE = 0.05
LORA_RANK = 16
ADAPTER_NAME = 'default'

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 8))
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 1
LOG_INTERVAL = 20
SAVE_INTERVAL = 4000
NUM_EPOCHS = 1

# None → use full _BASE_SIZES from dataset_think; int to subsample.
TOTAL_SAMPLES: Optional[int] = None

# -- Online-compression knobs (CM-v2 inference) -------------------------------
MIN_COT_CHARS = 256                           # skip too-short cot rows entirely
COMPRESS_RATIO = 2.0                          # used to derive the prompt char budget
COMPRESS_MAX_TOKENS = 2048
COMPRESS_TEMPERATURE = 0.4
COMPRESS_TOP_P = 0.9
COMPRESS_MAX_MODEL_LEN = 32768

OUTPUT_DIR = f'./output/embedding_lora_{BACKEND}'

# Production CM-v2 prompt (kept verbatim — same as cookbook/sample/sample.py).
CONDENSER_SYSTEM = """You are a text compression assistant. A downstream model will read your compressed output to decide whether the detail it needs is inside this block; if yes, it will fetch and read the original passage.

Downstream model workflow:
Read your compressed output -> Decide whether needed info is in this block -> If yes -> Fetch original.

Therefore your compression MUST NOT lose major information from the source.

Output format:

```text
## Summary
Overview plus facts STRONGLY RELATED to the Query, stated explicitly.

## More
A collapsed index; expansion required to see specific information.
```

Rules:
1. Telegraphic style — drop function words ("the", "a", "is", "are", "of", ...); colons and commas mean "is" / "has".
2. Summary MUST contain the passage's primary topic + 2–4 concrete core facts drawn from the source (entities, numbers, dates, relations). If a Query is given, order Query-relevant facts first, but STILL include other core facts within the budget. A Query is an ORDERING HINT, NOT a filter.
3. Summary MUST NOT be meta-commentary about the Query. Forbidden patterns: "no X mention", "Query info: absent", "passage covers Y only", "does not contain ...", "no relevant info", or summaries that are only abstract category words like "structure/order/usage" with no facts. If the passage is unrelated to the Query, you still summarize the passage normally.
4. More is an INDEX of category keywords, NOT inline data. Enumerate what CAN be recovered from the source (e.g. "birthplace, death place, age"); do NOT paste dates/numbers/names inline. Make sure all category of useful facts are introduced here.
5. Output language MUST match the source language.
6. Do NOT fabricate. Do NOT omit major information. Any fact not in the source MUST NOT appear in your output.

Now begin.
"""

CONDENSER_USER = (
    'Downstream model will read your compressed block to decide whether to '
    'expand it. Compress faithfully: preserve the passage topic + core facts. '
    'Do NOT invent facts. Do NOT drop major facts. Do NOT write meta-commentary '
    'about the Query (never write "Query info: absent", "no X mention", etc.); '
    'if the passage does not address the Query, still summarize the passage.\n\n'
    '## Query (ordering hint only — still summarize the whole passage)\n{query}\n\n'
    '## Target length\n'
    'Compress AS MUCH AS faithfully possible. HARD CEILING: {budget} chars '
    '(~50% of the source). If core facts fit in far fewer chars, output fewer. '
    'Never exceed the ceiling.\n\n'
    '## Passage\n{text}')


# ------------------------------------------------------------------- Dataset
class FlattenForEmbeddingProcessor(Preprocessor):
    """``{id, source, messages}`` (from dataset_think) → ``{id, source, query, cot}``.

    Drops rows whose ``cot`` is shorter than ``min_cot_chars`` (compression
    is a no-op below that, and InfoNCE quality drops on near-empty positives).
    """

    def __init__(self, min_cot_chars: int = MIN_COT_CHARS):
        self.min_cot_chars = min_cot_chars

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        out: List[Dict[str, Any]] = []
        for row in rows:
            messages = row.get('messages') or []
            query, cot = '', ''
            for m in messages:
                if not isinstance(m, dict):
                    continue
                role = m.get('role') or ''
                if role == 'user' and not query:
                    query = (m.get('content') or '').strip()
                elif role == 'assistant':
                    cot = (m.get('reasoning_content') or '').strip()
                    break
            if not query or len(cot) < self.min_cot_chars:
                continue
            out.append({
                'id': row.get('id', ''),
                'source': row.get('source', ''),
                'query': query,
                'cot': cot,
            })
        return self.map_row_to_col(out, keys=['id', 'source', 'query', 'cot'])


# ------------------------------------------------------------------ Embedding
class EmbeddingTemplate(Template):
    """Flatten ``{query, positive, negatives}`` into per-sentence ``InputFeature`` rows.

    Order within each row is ``anchor → positive → negatives`` — the layout
    :class:`InfonceLoss` requires (``group_start=1`` marks each anchor).
    """

    def batch_encode(self, trajectories, add_generation_prompt=False, **kwargs):
        columnar = isinstance(trajectories, Mapping)
        if columnar:
            trajectories = self.map_col_to_row(trajectories)

        out: List[InputFeature] = []
        for row in trajectories:
            anchor = row['query']
            positives = row['positive']
            if isinstance(positives, str):
                positives = [positives]
            negatives = list(row.get('negatives') or row.get('negative') or [])
            sentences = [anchor, *positives, *negatives]
            for i, text in enumerate(sentences):
                ids = self.processor(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    add_special_tokens=True,
                )['input_ids']
                out.append(InputFeature(
                    input_ids=ids,
                    attention_mask=[1] * len(ids),
                    group_start=int(i == 0),
                ))

        if columnar:
            out = self.map_row_to_col(out)
        return out


class EmbeddingProcessor(InputProcessor):
    """Single-step collator producing the flat embedding batch.

    ``labels`` here is the 1-D group-start mask consumed by :class:`InfonceLoss`,
    not token-level labels — so it must NOT pass through the standard pipeline
    (which would pad with ``-100`` and stack as a 2-D tensor).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_pipeline = [self._embed_collate, self._maybe_wrap_microbatch]

    def _embed_collate(self, inputs, **kwargs):
        device = Platform.get_local_device()
        max_len = max(len(row['input_ids']) for row in inputs)
        n = len(inputs)
        # default pad id 0 is harmless: only the last valid (attention_mask=1) position is read.
        input_ids = torch.zeros(n, max_len, dtype=torch.long)
        attention_mask = torch.zeros(n, max_len, dtype=torch.long)
        labels = torch.zeros(n, dtype=torch.long)
        for i, row in enumerate(inputs):
            ids = row['input_ids']
            ids = ids if isinstance(ids, torch.Tensor) else torch.as_tensor(ids, dtype=torch.long)
            seq_len = ids.shape[0]
            input_ids[i, :seq_len] = ids
            am = row.get('attention_mask')
            if am is None:
                attention_mask[i, :seq_len] = 1
            else:
                am = am if isinstance(am, torch.Tensor) else torch.as_tensor(am, dtype=torch.long)
                attention_mask[i, :seq_len] = am[:seq_len]
            labels[i] = int(row.get('group_start', 0))

        return InputFeature(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device),
        )

    def _maybe_wrap_microbatch(self, feature, **kwargs):
        # Megatron's forward_backward iterates a list of microbatch dicts;
        # treat the whole flat embedding batch as one microbatch.
        if self.framework == 'megatron':
            return [feature]
        return feature


# ------------------------------------------------------------------- Builders
def build_dataset():
    dataset = get_dataset(total=TOTAL_SAMPLES, load_from_cache_file=True, dropped_log='output/emb')
    dataset.map(FlattenForEmbeddingProcessor(), remove_columns=['messages'],
                num_proc=16, load_from_cache_file=True)
    return dataset


def build_model(device_mesh: DeviceMesh):
    if BACKEND == 'transformers':
        from twinkle.model import TransformersModel
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            ddp_config={'find_unused_parameters': True},
        )
        model.model._no_split_modules = {'Qwen3_5DecoderLayer'}
        return model
    if BACKEND == 'megatron':
        from twinkle.model import MegatronModel
        return MegatronModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            remote_group='model',
            mixed_precision='bf16',
            variable_seq_lengths=True,
        )
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def setup_optimizer(model, total_steps: int):
    if BACKEND == 'transformers':
        model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='CosineWarmupScheduler',
            num_warmup_steps=50,
            num_training_steps=total_steps,
        )
        return
    if BACKEND == 'megatron':
        model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='default',
            lr_warmup_steps=50,
            lr_decay_steps=total_steps,
        )
        return
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def save_checkpoint(model, name: str):
    model.save(name, output_dir=OUTPUT_DIR, adapter_name=ADAPTER_NAME)


# --------------------------------------------------------------------- Loop
def _build_compress_prompts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompts: List[Dict[str, Any]] = []
    for row in rows:
        cot = row['cot']
        budget = max(1, int(len(cot) / COMPRESS_RATIO))
        user = CONDENSER_USER.format(query=row['query'], budget=budget, text=cot)
        prompts.append({'messages': [
            {'role': 'system', 'content': CONDENSER_SYSTEM},
            {'role': 'user', 'content': user},
        ]})
    return prompts


def _decode_first_sequence(response) -> str:
    seqs = getattr(response, 'sequences', None) or []
    if not seqs:
        return ''
    return getattr(seqs[0], 'decoded', '') or ''


def train():
    # -------- Ray + device groups --------------------------------------------
    device_groups = [
        DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
        DeviceGroup(name='sampler',
                    ranks=list(range(MODEL_GPUS, NUM_GPUS)),
                    device_type='GPU'),
    ]
    model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)

    # -------- Data -----------------------------------------------------------
    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    total_steps = len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS

    # -------- Trainable embedding model + LoRA -------------------------------
    model = build_model(model_mesh)
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2, lora_dropout=0.05,
        target_modules='all-linear')
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    model.set_template(EmbeddingTemplate, model_id=MODEL_ID, max_length=EMB_MAX_LENGTH)
    model.set_processor(EmbeddingProcessor)
    model.set_loss(
        InfonceLoss,
        temperature=TEMPERATURE,
        use_batch=True,
        hard_negatives=HARD_NEGATIVES,
    )
    setup_optimizer(model, total_steps)

    # -------- Frozen CM-v2 sampler (online compressor) -----------------------
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': COMPRESS_MAX_MODEL_LEN,
            'enable_lora': False,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template(TEMPLATE_NAME, model_id=MODEL_ID, enable_thinking=False)
    compress_params = SamplingParams(
        max_tokens=COMPRESS_MAX_TOKENS,
        temperature=COMPRESS_TEMPERATURE,
        top_p=COMPRESS_TOP_P,
        num_samples=1,
    )

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {total_steps}')

    # -------- Train loop -----------------------------------------------------
    optimizer_group = model.optimizer_group[ADAPTER_NAME]
    for epoch in range(NUM_EPOCHS):
        for raw_batch in dataloader:
            # raw_batch: List[{id, source, query, cot}]
            compress_prompts = _build_compress_prompts(raw_batch)
            responses = sampler.sample(compress_prompts, compress_params)
            compressed = [_decode_first_sequence(r) for r in responses]

            # Drop rows where compression yielded empty text (vLLM sequence loss / OOM).
            emb_rows: List[Dict[str, Any]] = []
            for row, comp in zip(raw_batch, compressed):
                comp = (comp or '').strip()
                if not comp:
                    continue
                emb_rows.append({
                    'query': row['query'],
                    'positive': comp,
                    'negatives': [],
                })

            if len(emb_rows) < 2:
                # InfoNCE needs ≥2 anchors for a meaningful in-batch loss.
                logger.warning('Skipping step: only %d valid compressions in batch of %d',
                               len(emb_rows), len(raw_batch))
                continue

            # ``task='embedding'`` swaps lm_head → identity and writes pooled
            # per-sequence vectors to ``outputs['embeddings']`` for InfonceLoss.
            model.forward_backward(inputs=emb_rows, task='embedding')
            model.clip_grad_and_step()
            cur_step = optimizer_group.cur_step

            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    f'Epoch {epoch} Step {cur_step}/{total_steps}, '
                    f'kept={len(emb_rows)}/{len(raw_batch)}, metric: {metric}')
            if cur_step and cur_step % SAVE_INTERVAL == 0:
                save_checkpoint(model, f'step_{cur_step}')

        save_checkpoint(model, f'epoch-{epoch}')
    save_checkpoint(model, 'last-checkpoint')


if __name__ == '__main__':
    train()
