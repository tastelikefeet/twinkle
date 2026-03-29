import os
from peft import LoraConfig
from tqdm import tqdm

import twinkle
from twinkle import DeviceMesh, Platform, get_device_placement, get_logger
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.model import TransformersModel
from twinkle.data_format import Message, Trajectory
from twinkle.preprocessor import SelfCognitionProcessor, Preprocessor

# Construct a device_mesh, dp=2
device_mesh = DeviceMesh.from_sizes(dp_size=8)
# use torchrun mode
twinkle.initialize(mode='local', global_device_mesh=device_mesh)

logger = get_logger()


def eval(model):
    # 100 Samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://swift/self-cognition', data_slice=range(100)))
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    dataset.map(SelfCognitionProcessor('twinkle大模型', 'ModelScope社区'))
    dataset.encode()
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    for step, batch in tqdm(enumerate(dataloader)):
        model.forward_only(inputs=batch)
        model.calculate_loss()
    metrics = model.calculate_metric(is_training=False)
    return metrics


class EmojiDPOProcessor(Preprocessor):
    def __init__(
        self,
        system = 'You are a helpful assistant.',
        chosen_key: str = 'answer_zh',
        rejected_key: str = 'answer_en',
        prompt_key: str = 'prompt',
    ):
        self.system = system
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.prompt_key = prompt_key

    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row):
        """Process a single row."""
        prompt = row.get(self.prompt_key, '')
        chosen = row.get(self.chosen_key, '')
        rejected = row.get(self.rejected_key, '')

        prompt_messages = []
        if self.system:
            prompt_messages.append(Message(role='system', content=self.system))
        prompt_messages.append(Message(role='user', content=prompt))

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected)]

        return Trajectory(messages=chosen_messages)


def train():
    # 1000 samples
    dataset = Dataset(dataset_meta=DatasetMeta('ms://hjh0119/shareAI-Llama3-DPO-zh-en-emoji'))
    # Set template to prepare encoding
    dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    # Preprocess the dataset to standard format
    dataset.map(EmojiDPOProcessor)
    # Encode dataset
    dataset.encode()
    # Global batch size = 8, for GPUs, so 1 sample per GPU
    dataloader = DataLoader(dataset=dataset, batch_size=8)
    # Use a TransformersModel
    model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct')
    model.model._no_split_modules = {'Qwen3_5DecoderLayer'}

    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules='all-linear')

    # Add a lora to model, with name `default`
    # Comment this to use full-parameter training
    model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=2)
    # Add Optimizer for lora `default`
    model.set_optimizer(optimizer_cls='AdamW', lr=1e-4)
    # Add LRScheduler for lora `default`
    model.set_lr_scheduler(
        scheduler_cls='CosineWarmupScheduler', num_warmup_steps=5, num_training_steps=len(dataloader))
    logger.info(get_device_placement())
    # Print the training config
    logger.info(model.get_train_configs())
    logger.info(f'Total steps: {len(dataloader)}')
    loss_metric = 99.0
    # lora: 8G * 8
    # full: 18G * 8
    for step, batch in enumerate(dataloader):
        # Do forward and backward
        model.forward_backward(inputs=batch)
        # Step
        model.clip_grad_and_step()
        if step % 20 == 0:
            # Print metric
            metric = model.calculate_metric(is_training=True)
            logger.info(f'Current is step {step} of {len(dataloader)}, metric: {metric}')
    model.save(f'last-checkpoint')


if __name__ == '__main__':
    train()
