# Twinkle Training Script Skill

You are an expert at writing training scripts for the Twinkle framework. Follow these conventions when generating training code.

## Ray Cluster Configuration

### DeviceGroup & DeviceMesh

Training typically splits GPUs into groups (model vs sampler). Define them explicitly:

```python
from twinkle import DeviceMesh, DeviceGroup

device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
    DeviceGroup(name='sampler', ranks=list(range(MODEL_GPUS, NUM_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, dp_size=SAMPLER_GPUS)
```

DeviceMesh supports: `dp_size`, `tp_size`, `pp_size`, `fsdp_size`, `cp_size`, `ep_size`, `sp_size`.

### Initialize

```python
import twinkle
twinkle.initialize(mode='ray', nproc_per_node=NUM_GPUS, groups=device_groups, lazy_collect=False)
```

- `mode='local'`: single node with torchrun. `mode='ray'`: multi-node Ray cluster.
- `lazy_collect=True`: results collected lazily (better throughput). `False`: synchronous collect.
- `groups` is required for Ray mode.

### Result Collection

Remote functions support `collect='first'` (return from rank-0 only) or `collect='all'` (gather from all ranks).

## Model Backend Initialization

### Transformers Backend

```python
from twinkle.model import TransformersModel
model = TransformersModel(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=model_mesh,
    remote_group='model',
)
```

- Supports FSDP2 natively via DeviceMesh with `fsdp_size`.
- `model_cls` optional: pass HuggingFace model class for custom architectures.

### Megatron Backend

```python
from twinkle.model import MegatronModel
model = MegatronModel(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=model_mesh,
    remote_group='model',
    mixed_precision='bf16',
)
```

- Requires DeviceMesh with `tp_size` / `pp_size` configured.
- Optimizer: `model.set_optimizer('default', lr=1e-5)` uses Megatron's built-in optimizer.
- LR scheduler: `model.set_lr_scheduler('default', lr_decay_steps=MAX_STEPS, max_lr=1e-5)`.

### Common Model Setup

```python
from peft import LoraConfig

lora_config = LoraConfig(
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    r=32, lora_alpha=64, lora_dropout=0.05,
)
model.add_adapter_to_model('default', lora_config, gradient_accumulation_steps=1)
model.set_optimizer('AdamW', lr=1e-5)
model.set_lr_scheduler('CosineAnnealingLR', T_max=MAX_STEPS, eta_min=0)
model.set_loss('GRPOLoss', epsilon=0.2)
model.set_processor(InputProcessor)
model.set_template('Qwen3_5Template', model_id=MODEL_ID)
```

## Dataset

### Loading

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(
    dataset_id='ms://modelscope/gsm8k',  # ms:// prefix for ModelScope hub
    subset_name='main',
    split='train',
    data_slice=range(5000),  # optional: limit data
))
```

- `dataset_id`: local path, HuggingFace ID, or `ms://` prefixed ModelScope ID.
- `data` parameter: pass in-memory `List[Dict]` or generator callable.
- `streaming=True` in kwargs for streaming mode.

### Template & Encode

```python
dataset.set_template('Qwen3_5Template', model_id=MODEL_ID, max_length=8192)
dataset.map(GSM8KProcessor())      # preprocessor transforms raw data -> Trajectory
dataset.encode(add_generation_prompt=True)  # encode to input_ids/labels
```

- `encode()` internally calls `template.batch_encode()` which runs the full pre_pipeline.
- `add_generation_prompt=True`: appends generation prompt (for sampling inputs).
- `add_generation_prompt=False`: includes assistant content in labels (for training).

## Preprocessor

### Built-in Preprocessors

- `GSM8KProcessor`: Extracts question/answer from GSM8K format
- `SelfCognitionProcessor(model_name, model_author)`: Generates self-cognition QA
- `EmojiDPOProcessor`: DPO pairs for emoji preference
- `CompetitionMathProcessor`, `CountdownProcessor`, etc.

### Custom Preprocessor

```python
from twinkle.preprocessor import Preprocessor
from twinkle.data_format import Trajectory, Message

class MyProcessor(Preprocessor):
    def __call__(self, rows):
        rows = self.map_col_to_row(rows)
        results = []
        for row in rows:
            messages = [
                Message(role='user', content=row['question']),
                Message(role='assistant', content=row['answer']),
            ]
            results.append(Trajectory(messages=messages))
        return self.map_row_to_col(results)
```

### Data Format

- `Trajectory`: `{'messages': [Message(role, content)], 'extend_message': [('key', value)]}`
- `Message`: `TypedDict` with `role` and `content` keys (access via bracket notation: `msg['role']`)
- DPO format: `Trajectory(messages=chosen, extend_message=[('rejected_messages', rejected)])`

## MultiTurnRollout

```python
from twinkle_agentic.envs import OpenEnv, EnvTool
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager

# Create environment and tools
env = OpenEnv(base_url='http://localhost:8000', tool_schema=TOOL_SCHEMA)
env_tools = EnvTool.from_env(env)
tool_manager = ToolManager(env_tools)

# Run multi-turn rollout
rollout = MultiTurnRollout(
    sampler=sampler,
    template=template,
    tool_managers=[tool_manager] * batch_size,
    max_turns=6,
)
trajectories = rollout.run(initial_trajectories, sampling_params)
```

- Each trajectory starts with system + initial observation.
- MultiTurnRollout drives: model generates tool calls -> env.step() -> observations fed back.
- Episode reward extracted from env after rollout completes.

## Sampler

### vLLMSampler

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams

sampler = vLLMSampler(
    model_id=MODEL_ID,
    engine_args={
        'gpu_memory_utilization': 0.8,
        'max_model_len': 4096,
        'enable_lora': True,
        'max_lora_rank': 32,
    },
    device_mesh=sampler_mesh,
    remote_group='sampler',
)
sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)
```

### SamplingParams

```python
sampling_params = SamplingParams(
    max_tokens=4096,
    temperature=1.0,
    top_p=1.0,
    num_samples=1,
    logprobs=1,    # needed for GRPO old_logps
)
```

### Input/Output

- Input: `List[Trajectory]` or `List[InputFeature]`
- Output: `List[SampleResponse]`, each containing `.sequences[].tokens`, `.sequences[].logprobs`, `.sequences[].new_input_feature`

### Weight Sync

```python
from twinkle.checkpoint_engine import CheckpointEngineManager

ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)
# In training loop:
ckpt_manager.sync_weights(merge_and_sync=True)   # merge LoRA -> full model -> sync to vLLM
ckpt_manager.sync_weights(merge_and_sync=False)  # sync LoRA weights only (requires enable_lora=True)
```

## Async Sampling (Server Mode)

### Server-side

```bash
twinkle-server launch -c server_config.yaml
```

### Client-side

```python
from twinkle import init_twinkle_client
from twinkle_client.model import MultiLoraTransformersModel
from twinkle_client.sampler import vLLMSampler

init_twinkle_client(base_url='http://server:8000', api_key='...')
model = MultiLoraTransformersModel(model_id=MODEL_ID)
sampler = vLLMSampler(model_id=MODEL_ID)

# Weight sync via adapter_uri
path = model.save('checkpoint-1')
twinkle_path = model.get_checkpoint_twinkle_path(run_id, 'checkpoint-1')
responses = sampler.sample(inputs, adapter_uri=twinkle_path)
```

## Training Methods

### SFT (Supervised Fine-Tuning)

```python
model.set_loss('CrossEntropyLoss')
dataset.encode(add_generation_prompt=False)  # include assistant in labels
for batch in dataloader:
    model.forward_backward(inputs=batch)
    model.clip_grad_and_step()
```

### GRPO (Group Relative Policy Optimization)

```python
model.set_loss('GRPOLoss', epsilon=0.2)
# Loop: sample -> reward -> advantage -> train
ckpt_manager.sync_weights(merge_and_sync=False)
responses = sampler.sample(prompts, sampling_params)
rewards = reward_fn(responses)
advantages = GRPOAdvantage()(rewards, num_generations=NUM_GENERATIONS, scale='group')
model.forward_backward(inputs=inputs, old_logps=old_logps, advantages=advantages, micro_batch_size=2)
model.clip_grad_and_step()
```

### DPO (Direct Preference Optimization)

```python
model.set_loss('DPOLoss', beta=0.1)
# Data: chosen + rejected concatenated as a single batch (first half chosen, second half rejected)
# Preprocessor outputs Trajectory with extend_message=[('rejected_messages', rejected)]
model.forward_backward(inputs=dpo_batch)
model.clip_grad_and_step()
```

### GKD (Generalized Knowledge Distillation)

```python
model.set_loss('GKDLoss', beta=0.5, temperature=1.0)
# On-policy: sample from student, score with teacher
# Off-policy: use pre-computed teacher logits
```

### PT (Pre-Training)

```python
# No LoRA, full parameter training
model = TransformersModel(model_id=MODEL_ID, device_mesh=mesh)
model.set_optimizer('AdamW', lr=1e-4)
# Use streaming dataset for large corpora
dataset = Dataset(DatasetMeta(dataset_id='...'), streaming=True)
```

## TUI Integration & Training Observability

### CRITICAL CONSTRAINT

**DO NOT modify the Twinkle SDK itself (`src/twinkle/` or `src/twinkle_client/`).** All training logic MUST be implemented as external scripts that import and use the SDK. The TUI monitors training through local files written by the training script.

### TrainingRuntime Integration

Every training script MUST use `TrainingRuntime` for TUI integration:

```python
from twinkle_tui.runtime import TrainingRuntime, TrainingStoppedError

rt = TrainingRuntime(run_id='my-grpo-run')  # run_id should be descriptive and unique
rt.start(model_id=MODEL_ID, config={'lr': LR, 'batch_size': BATCH_SIZE, ...})

try:
    for step, batch in enumerate(dataloader):
        rt.check_signals()  # MUST be called every step (handles pause/stop)

        # ... training logic ...
        rt.log_metrics(
            step=step,
            total_steps=MAX_STEPS,
            loss=loss_val,
            reward_total=total_reward,
            reward_accuracy=acc_reward,
            reward_format=fmt_reward,
            grad_norm=grad_norm,
            lr=current_lr,
            kl_divergence=kl_div,
            entropy=entropy,
            throughput_samples_per_sec=throughput,
            completion_avg_length=avg_len,
        )
        rt.log(f'[Step {step}/{MAX_STEPS}] loss={loss_val:.4f} reward={total_reward:.3f}')

    rt.finish(status='completed')
except TrainingStoppedError:
    rt.finish(status='stopped')
except Exception as e:
    rt.log(f'ERROR: {e}')
    rt.finish(status='error')
```

### Signal Handling (Pause/Stop)

`rt.check_signals()` checks for signal files in `~/.cache/twinkle/{run_id}/`:
- **pause file**: Training blocks until the file is removed (resume)
- **stop file**: Raises `TrainingStoppedError`, training exits gracefully

Always call `check_signals()` at the beginning of each training step (before forward/backward). This enables the TUI user to pause/resume/stop training at step boundaries.

### Required Metrics Logging

Log ALL available metrics every step for LLM analysis. The more data, the better the AI monitor can diagnose issues:

| Metric | Key | Why |
|--------|-----|-----|
| Step progress | `step`, `total_steps` | Track completion |
| Loss | `loss` | Core training signal |
| Reward (total) | `reward_total` | RL objective |
| Reward (per-component) | `reward_accuracy`, `reward_format`, etc. | Diagnose reward hacking |
| Gradient norm | `grad_norm` | Detect explosion/vanishing |
| Learning rate | `lr` | Verify scheduler |
| KL divergence | `kl_divergence` | Policy drift detection |
| Entropy | `entropy` | Detect mode collapse |
| Throughput | `throughput_samples_per_sec` | Performance regression |
| Completion length | `completion_avg_length` | Reward gaming detection |
| Epoch | `epoch` | Multi-epoch tracking |

Additional recommended metrics for specific scenarios:
- **GRPO**: `advantage_mean`, `advantage_std`, `clip_fraction`
- **DPO**: `chosen_reward`, `rejected_reward`, `reward_margin`
- **Multi-turn**: `avg_turns`, `success_rate`, `tool_call_count`

### Required Log Messages

Use `rt.log()` for human-readable events that help the LLM understand context:

```python
# Good: rich, contextual logs
rt.log(f'[Step {step}/{MAX_STEPS}] loss={loss:.4f} reward={reward:.3f} grad_norm={gn:.2f}')
rt.log(f'Sampling {NUM_GENERATIONS} completions per prompt, batch_size={BATCH_SIZE}')
rt.log(f'Weight sync to vLLM completed (merge_and_sync={merge})')
rt.log(f'Checkpoint saved: {checkpoint_path}')
rt.log(f'Dataset loaded: {len(dataset)} samples, template={template_name}')
rt.log(f'Reward breakdown: accuracy={acc:.3f} brevity={brev:.3f} format={fmt:.3f}')
rt.log(f'GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated')

# Bad: too terse, no context
rt.log(f'step {step}')  # ← useless, already in metrics
rt.log('training')       # ← meaningless
```

### File Layout

```
~/.cache/twinkle/{run_id}/
├── meta.json          # Run metadata (model_id, config, start_time, status)
├── metrics.jsonl      # One JSON line per step (all metrics)
├── logs.jsonl         # One JSON line per event (ts + msg)
├── pause              # Signal file: exists = paused
└── stop               # Signal file: exists = stop requested
```

## Experiment Management

Each experiment MUST be organized in a dedicated folder:

```
experiments/{exp_name}/
├── plan.md              # Experiment design and hypothesis
├── config.yaml          # Training configuration (reproducible)
├── train.py             # Training script
├── train.sh             # Launch command with environment vars
├── logs/                # Training logs (auto-generated)
├── checkpoints/         # Model checkpoints (auto-generated)
└── results.md           # Results, metrics, conclusions
```

Always record in `plan.md`:
- Objective and hypothesis
- Baseline comparison
- Key hyperparameters and why
- Expected outcome and success criteria
