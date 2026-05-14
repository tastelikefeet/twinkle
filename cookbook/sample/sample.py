"""
Standalone inference example using Ray + vLLMSampler with LoRA adapter.

This script demonstrates how to:
1. Initialize Twinkle with Ray for distributed inference
2. Create a vLLMSampler with LoRA enabled on dedicated GPUs
3. Load a LoRA adapter from a local checkpoint path
4. Send prompts (Trajectory format) and collect generated responses

Usage:
    # Single GPU inference
    SAMPLER_GPUS=1 python sample.py

    # Multi-GPU inference (tensor parallel)
    SAMPLER_GPUS=2 python sample.py

    # Use a different model / adapter
    MODEL_ID=/path/to/model LORA_PATH=/path/to/adapter SAMPLER_GPUS=1 python sample.py
"""

import os
from typing import List, Dict, Any

import twinkle
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.data_format import SamplingParams
from twinkle.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'Qwen/Qwen3.5-4B')
LORA_PATH = os.environ.get('LORA_PATH', 'output/condenser_ddp/last-checkpoint')
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 1))


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

Example:

Source:
```text
Marie Curie (7 Nov 1867 – 4 Jul 1934), born Maria Sklodowska in Warsaw (then Russian Poland); parents were teachers. Barred from Polish universities, she and her sister agreed to take turns funding each other's overseas study.

In 1891 Marie reached Paris and enrolled at the Sorbonne, earning a physics degree (1893) and a mathematics degree (1894), becoming the school's first female physics lecturer. In 1895 she married French physicist Pierre Curie; they spent the rest of their lives on radioactivity research.

In July 1898 she discovered polonium, named after her homeland Poland; in December she and Pierre announced the discovery of radium. She coined "radioactivity" and showed it is an atomic property, not a chemical reaction.

In 1903 she shared the Nobel Prize in Physics with Pierre and Henri Becquerel. In 1911 she alone won the Nobel Prize in Chemistry for polonium and radium. She is the first woman to win a Nobel, and the only person to win Nobels in two different sciences. After Pierre died in a carriage accident in 1906, Marie took his chair and became the first female professor at the Sorbonne.

During World War I she developed mobile X-ray units, called "Petites Curies" in French; about 20 were deployed to the front, examining over 1,000,000 wounded soldiers.

She died of aplastic anaemia from radiation exposure on 4 July 1934 in Passy, Haute-Savoie, France, aged 66. Her notebooks remain highly radioactive, kept in lead boxes; researchers must wear protective gear to consult them.
```

Compressed:
```text
## Summary
Marie Curie: French-Polish physicist/chemist, founder of radioactivity research, first female Sorbonne professor.
- Nobel x2 (Physics + Chemistry); first woman Nobel laureate; only person with Nobels in two sciences.
- Discovered polonium + radium; coined "radioactivity"; proved it is an atomic property.

## More
- birthplace, death place, age, cause of death
- degree years, in-school firsts x2
- element naming origin, collaborators, full timeline
- Nobel year per prize, co-laureates, citation
- device name, deployment scale, patients treated
- notebook radioactivity, storage, access conditions
```

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

query = 'In what year was the creator of the current arrangement of the "Simpson\'s Theme" born?'
passage = 'California Breed: California Breed was an English-American hard rock band based in Los Angeles, California. Formed in 2013, the band was a supergroup composed of bassist and vocalist Glenn Hughes, guitarist Andrew Watt, and drummer Jason Bonham. Following the breakup of his previous band Black Country Communion, Hughes was introduced to Watt in 2013 and the two quickly formed California Breed, with Black Country Communion drummer Bonham completing the lineup shortly after. The band recorded its self-titled debut album with producer Dave Cobb in late 2013, which was released through Frontiers Records in May 2014 and reached number 78 on the US "Billboard" 200.'
budget = len(passage) // 2
user = CONDENSER_USER.format(
        query=query, budget=budget, text=passage)


def build_prompts() -> List[Dict[str, Any]]:
    """Build a list of Trajectory dicts (messages format) as prompts."""
    prompts = [
        {
            'messages': [
                {'role': 'system', 'content': CONDENSER_SYSTEM},
                {'role': 'user', 'content': user},
            ]
        },
    ]
    return prompts


def main():
    # ── 1. Initialize Twinkle with Ray ──────────────────────────────────
    device_groups = [
        DeviceGroup(name='sampler', ranks=list(range(SAMPLER_GPUS)), device_type='GPU', gpus_per_worker=SAMPLER_GPUS),
    ]
    sampler_mesh = DeviceMesh.from_sizes(world_size=SAMPLER_GPUS, tp_size=SAMPLER_GPUS)
    twinkle.initialize(mode='ray', nproc_per_node=SAMPLER_GPUS, groups=device_groups)

    # ── 2. Create vLLMSampler with LoRA enabled ────────────────────────
    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.7,
            'max_model_len': 4096,
            'enable_lora': True,
            'max_loras': 1,
            'max_lora_rank': 32,
            'enable_tower_connector_lora': True,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID, enable_thinking=False)
    logger.info(get_device_placement())

    # ── 3. Configure sampling parameters ────────────────────────────────
    sampling_params = SamplingParams(
        max_tokens=2018,
        temperature=0.7,
        top_p=0.9,
        num_samples=1,
    )

    # ── 4. Run inference ────────────────────────────────────────────────
    prompts = build_prompts()
    logger.info(f'Sampling {len(prompts)} prompts with model {MODEL_ID} ...')

    responses = sampler.sample(prompts, sampling_params, adapter_path=LORA_PATH)

    # ── 5. Print results ────────────────────────────────────────────────
    for i, response in enumerate(responses):
        for seq in response.sequences:
            text = seq.decoded
            logger.info(f'\n{"="*60}\nPrompt {i}: {prompts[i]["messages"][-1]["content"]}\n{"─"*60}\n{text}\n')

    logger.info('Done.')


if __name__ == '__main__':
    main()
