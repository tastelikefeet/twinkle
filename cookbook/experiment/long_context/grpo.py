import os
import random
from copy import copy
from typing import List, Dict, Any

from twinkle.template import Qwen3_5Template

import twinkle
from data import create_dataset
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, Message, Trajectory
from twinkle.dataloader import DataLoader
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3.5-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '0')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 8))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS',8))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16)) # global prompt-level, global completion-level batch size = BATCH_SIZE * num_generations * dp_size
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 16)) # global completion-level mini-batch-size
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE', 2)) # per-device-micro-batch-size (completion-level), batch_size in forward_backward
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))


def poison_dataset(batch: List[Trajectory], sampler: vLLMSampler):
    SYSTEM = """
You are a data augmentation assistant for training robust language models.

Your task is to generate *irrelevant but natural-sounding distractor content* for marked positions in a text.

You will receive a text that contains one marker like:
<NOISE_SLOT_1>, <NOISE_SLOT_2>, ...

Instructions:

1. For each marker, generate a short piece of distractor content.

2. The distractor content must follow ALL rules:
   - Fluent, natural, and stylistically consistent with the surrounding text.
   - Topically plausible and may appear relevant at first glance.
   - However, it must be *actually unnecessary* for solving the task.
   - It must NOT change the original intent or correct answer.
   - It must NOT introduce contradictions or false claims that would alter the answer.
   - It must NOT provide hints, shortcuts, or key information for solving the task.

3. Length per distractor:
   - 200~500 tokens

4. Important output constraint:
   - ONLY output the generated distractor contents.
   - Do NOT output the original text.
   - Do NOT repeat the markers.
   - Do NOT add explanations.

5. Output format:
   - Return one line per marker
   - Preserve marker order
   - Format:
     <NOISE_SLOT_1>: ...
     <NOISE_SLOT_2>: ...

---

Examples:

Input:
Question: What is 12 + 7? <NOISE_SLOT_1>

Output:
<NOISE_SLOT_1>: some people prefer doing mental math while commuting, though it doesn't affect this calculation

---

Input:
Context: The capital of France is Paris. <NOISE_SLOT_1> It is known for the Eiffel Tower.
Question: What is the capital of France?

Output:
<NOISE_SLOT_1>: many European capitals have long histories with shifting political importance over centuries

---

Input:
Passage: Water boils at 100°C at sea level. <NOISE_SLOT_1>
Question: At what temperature does water boil at sea level?

Output:
<NOISE_SLOT_1>: in cooking, slight variations in temperature can still produce similar results depending on altitude

Now Begin:
"""

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1)
    def construct_messages():
        all_trajectory = []
        heads = []
        for trajectory in batch:
            messages = trajectory['messages']
            messages = copy(messages)
            if messages[0]['role'] == 'system':
                messages = messages[1:]
            response = messages[-1]
            messages = messages[:-1]
            user_info = ''
            for message in messages:
                user_info += 'Role:\n' + message['role'] + '\nContent:\n' + message['content'] + '\n\n'
            head = random.randint(0, 2) == 0
            if head:
                user_info = '<NOISE_SLOT_1>\n' + user_info
            else:
                user_info = user_info + '<NOISE_SLOT_1>\n'
            heads.append(head)
            _new_messages = [
                Message(role='system', content=SYSTEM),
                Message(role='user', content=f'The Query is: {user_info}, the response is {response["content"]}, now generate your augmentation:'),
            ]
            all_trajectory.append(Trajectory(messages = _new_messages))
        return all_trajectory, heads

    samples, heads = construct_messages(batch)
    sample_responses = sampler.sample(samples, sampling_params=sampling_params)
    for i, response in enumerate(sample_responses):
        decoded = response.sequences[0].decoded
        head = heads[i]
        messages = batch[i]['messages']
        if head:
            query = [m for m in messages if m['role'] == 'user'][0]
            query['content'] = decoded + ' ' + query['content']
        else:
            query = [m for m in messages if m['role'] == 'user'][-1]
            query['content'] = query['content'] + ' ' + decoded
    return batch


def multi_round_sample(samples: List[Trajectory], sampler: vLLMSampler, sampling_params: SamplingParams, num_generations, template, max_round=10) -> List[Trajectory]:
    results = samples * num_generations
    for i in range(max_round):
        responses = sampler.sample(results, sampling_params=sampling_params)
        for j, response in enumerate(responses):
            new_input_features = response.sequences[0].new_input_feature
            new_input_features.pop('input_ids', None)
            results[j] = new_input_features
            last_content = new_input_features['messages'][-1]['content']
            output_dict = template.tokenizer.parse_response(last_content)
            

def main():
    # set sampler and model separate to use different gpus
    device_groups = [
        DeviceGroup(name='model',ranks=MODEL_GPUS,device_type='GPU'),
        DeviceGroup(name='sampler',ranks=SAMPLER_GPUS,device_type='GPU'),
    ]
    if USE_MEGATRON:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    else:
        model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups)
    if USE_MEGATRON:
        from twinkle.model.megatron import MegatronModel
        model = MegatronModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model', mixed_precision='bf16')
    else:
        model = TransformersModel(model_id=MODEL_ID, device_mesh=model_mesh, remote_group='model')

    template = Qwen3_5Template(MODEL_ID)
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        device_mesh=model_mesh,
        remote_group='model',
    )

    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=len(dataloader), max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=len(dataloader), eta_min=0)
    model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 64000,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1)

    optim_step = 0
    logger.info(get_device_placement())

    for batch in dataloader:
        ckpt_manager.sync_weights(merge_and_sync=False)
        sampler.reset_prefix_cache()
        batch = poison_dataset(batch, sampler)
        sample_responses = multi_round_sample(batch, sampler, sampling_params, NUM_GENERATIONS, template)

        all_input_data: List[Dict[str, Any]] = []
        all_old_logps: List[List[float]] = []
        all_completion_lengths: List[int] = []

        for sample_response in sample_responses:
            for sequence in sample_response.sequences:
                all_input_data.append(sequence.new_input_feature)
                all_old_logps.append([logprob[0][1] for logprob in sequence.logprobs])
                all_completion_lengths.append(len(sequence.tokens))
        total_rewards, format_rewards, accuracy_rewards = compute_rewards(
            all_input_data
        )
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'format': format_rewards,
                'accuracy': accuracy_rewards,
            },
        )

        advantages = advantage_fn(total_rewards, num_generations=NUM_GENERATIONS, scale='group').tolist()

        # Split completions into mini-batches and run one optim step per mini-batch.
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

            if optim_step >= MAX_STEPS:
                break
            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')

if __name__ == '__main__':
    main()
