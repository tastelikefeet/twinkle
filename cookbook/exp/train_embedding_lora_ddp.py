"""LoRA embedding training for Qwen3.5-4B with InfoNCE loss (Transformers + Megatron).

Each row of the source JSONL must contain::

    {"query": "...", "positive": "...", "negatives": ["...", "...", ...]}

``positive`` may be a string or a list. ``negatives`` is optional when in-batch
negatives suffice (``use_batch=True``).

Pipeline (identical for both backends):
  - ``EmbeddingTemplate.batch_encode`` flattens each row in
    ``anchor + positive + negatives`` order — the layout :class:`InfonceLoss`
    expects — and tags the anchor with ``group_start=1``.
  - ``EmbeddingProcessor`` pads & stacks the flat batch into
    ``input_ids``/``attention_mask`` and gathers ``group_start`` into the 1-D
    ``labels`` tensor consumed by :class:`InfonceLoss`.
  - ``forward_backward(..., task='embedding')`` swaps ``lm_head`` /
    ``output_layer`` for identity (TransformersEmbeddingPatch /
    MegatronEmbeddingPatch) and writes per-sequence vectors to
    ``outputs['embeddings']`` after SP/CP-aware last-token pooling.

Switch ``BACKEND`` between ``'transformers'`` and ``'megatron'``; the rest of
the script is backend-agnostic.

Launch:
    torchrun --nproc_per_node=8 cookbook/exp/train_embedding_lora_ddp.py
"""
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch
from peft import LoraConfig

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.data_format import InputFeature
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.loss import InfonceLoss
from twinkle.processor import InputProcessor
from twinkle.template import Template

logger = get_logger()

# -- Backend selection --------------------------------------------------------
BACKEND: Literal['transformers', 'megatron'] = 'transformers'

MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
DATASET_PATH = str(
    Path(__file__).resolve().parent.parent.parent / 'embedding_train.jsonl')

MAX_LENGTH = 512
HARD_NEGATIVES = 7
TEMPERATURE = 0.05

# Parallelism (megatron uses TP/PP/CP; transformers ignores them).
DP_SIZE = 8
TP_SIZE = 1
PP_SIZE = 1
CP_SIZE = 1

# query rows per micro-batch; each row expands to 1 + 1 + HARD_NEGATIVES sentences
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GRADIENT_ACCUMULATION_STEPS = 1
LOG_INTERVAL = 20
NUM_EPOCHS = 1

OUTPUT_DIR = f'./output/embedding_lora_{BACKEND}'
ADAPTER_NAME = 'default'


class EmbeddingTemplate(Template):
    """Flatten ``{query, positive, negatives}`` into per-sentence ``InputFeature`` rows.

    Order within each row is ``anchor → positive(s) → negatives`` — the layout
    :class:`InfonceLoss` requires (``group_start=1`` marks each anchor).
    """

    def batch_encode(self, trajectories, add_generation_prompt=False, **kwargs):
        columnar = isinstance(trajectories, Mapping)
        if columnar:
            trajectories = self.map_col_to_row(trajectories)

        out = []
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


device_mesh = DeviceMesh.from_sizes(
    dp_size=DP_SIZE, tp_size=TP_SIZE, pp_size=PP_SIZE, cp_size=CP_SIZE)
twinkle.initialize(mode='local', global_device_mesh=device_mesh)


def build_dataset() -> Dataset:
    return Dataset(dataset_meta=DatasetMeta(DATASET_PATH))


def build_model():
    if BACKEND == 'transformers':
        from twinkle.model import TransformersModel
        model = TransformersModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            ddp_config={'find_unused_parameters': True})
        model.model._no_split_modules = {'Qwen3_5DecoderLayer'}
        return model
    if BACKEND == 'megatron':
        from twinkle.model import MegatronModel
        return MegatronModel(
            model_id=MODEL_ID,
            device_mesh=device_mesh,
            mixed_precision='bf16',
            variable_seq_lengths=True)
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def setup_optimizer(model, total_steps: int):
    if BACKEND == 'transformers':
        model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='CosineWarmupScheduler',
            num_warmup_steps=50,
            num_training_steps=total_steps)
        return
    if BACKEND == 'megatron':
        model.set_optimizer(optimizer_cls='default', lr=LEARNING_RATE)
        model.set_lr_scheduler(
            scheduler_cls='default', lr_warmup_steps=50, lr_decay_steps=total_steps)
        return
    raise ValueError(f'Unknown BACKEND={BACKEND!r}')


def save_checkpoint(model, name: str, dataloader: DataLoader):
    model.save(
        name,
        output_dir=OUTPUT_DIR,
        adapter_name=ADAPTER_NAME,
        save_optimizer=True,
        consumed_train_samples=dataloader.get_state()['consumed_train_samples'],
    )


def train():
    dataset = build_dataset()
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE)

    model = build_model()

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules='all-linear')
    model.add_adapter_to_model(
        ADAPTER_NAME, lora_config, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS)

    model.set_template(EmbeddingTemplate, max_length=MAX_LENGTH)
    model.set_processor(EmbeddingProcessor)
    model.set_loss(
        InfonceLoss,
        temperature=TEMPERATURE,
        use_batch=True,
        hard_negatives=HARD_NEGATIVES,
    )
    setup_optimizer(model, len(dataloader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS)

    logger.info(get_device_placement())
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader) * NUM_EPOCHS}')

    optimizer_group = model.optimizer_group[ADAPTER_NAME]

    for epoch in range(NUM_EPOCHS):
        for batch in dataloader:
            # task='embedding' selects the backend-appropriate embedding patch
            # and routes pooled per-sequence vectors into outputs['embeddings'].
            model.forward_backward(inputs=batch, task='embedding')
            model.clip_grad_and_step()
            cur_step = optimizer_group.cur_step
            if cur_step % LOG_INTERVAL == 0:
                metric = model.calculate_metric(is_training=True)
                logger.info(
                    f'Epoch {epoch} Step {cur_step}/{len(dataloader) * NUM_EPOCHS}, metric: {metric}')
        save_checkpoint(model, f'epoch-{epoch}', dataloader)
    save_checkpoint(model, 'last-checkpoint', dataloader)


if __name__ == '__main__':
    train()
