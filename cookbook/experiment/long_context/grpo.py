import json
import os
import random
from copy import copy
from typing import List, Dict, Any, Tuple

import numpy as np

from twinkle.reward import MathReward
from twinkle.template import Qwen3_5Template

import twinkle
import sys
from twinkle import DeviceMesh, DeviceGroup, get_device_placement, get_logger
from twinkle.advantage import GRPOAdvantage
from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.data_format import SamplingParams, Message, Trajectory, ToolCall
from twinkle.dataloader import DataLoader
from twinkle.metric import CompletionRewardMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.sampler import vLLMSampler

logger = get_logger()

MODEL_ID = os.environ.get('MODEL_ID', 'ms://Qwen/Qwen3-4B')
USE_MEGATRON = bool(int(os.environ.get('USE_MEGATRON', '0')))

MODEL_GPUS = int(os.environ.get('MODEL_GPUS', 8))
SAMPLER_GPUS = int(os.environ.get('SAMPLER_GPUS', 8))
NUM_GPUS = MODEL_GPUS + SAMPLER_GPUS

NUM_GENERATIONS = int(os.environ.get('NUM_GENERATIONS', 8))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS', 4096))
LEARNING_RATE = float(os.environ.get('LR', 1e-5))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE',
                                16))  # global prompt-level, global completion-level batch size = BATCH_SIZE * num_generations * dp_size
MINI_BATCH_SIZE = int(os.environ.get('MINI_BATCH_SIZE', 16))  # global completion-level mini-batch-size
MICRO_BATCH_SIZE = int(os.environ.get('MICRO_BATCH_SIZE',
                                      2))  # per-device-micro-batch-size (completion-level), batch_size in forward_backward
GRADIENT_ACCUMULATION_STEPS = int(os.environ.get('GRADIENT_ACCUMULATION_STEPS', 1))

import json
import math
import re
from collections import Counter
from typing import Any, Dict, List, Tuple, Optional

from twinkle.data_format import Message, Tool, ToolCall, Trajectory
from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.preprocessor import Preprocessor

SYSTEM = """You are a helpful assistant. You first thinks and outputs about the reasoning process and then provides the user with the answer. If the answer can be expressed with deterministic numbers or text, you should wrap it with \\boxed{...}.
"""


def pil_to_bytes(img):
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return {'bytes': buffer.getvalue(), 'path': None}


class StepFlashPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        raw_messages = row.get('messages', [])
        messages = [Message(role='system', content=SYSTEM)]
        tools = None

        for msg in raw_messages:
            # Convert OpenAI tool_calls to Twinkle ToolCall
            raw_tool_calls = msg.get('tool_calls')
            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        tool_name=tc['function']['name'],
                        arguments=tc['function']['arguments']
                    ) for tc in raw_tool_calls
                ]

            # Extract tools from first user message (OpenAI format)
            if msg.get('role') == 'user' and msg.get('tools') and tools is None:
                tools = [
                    Tool(
                        tool_name=t['function']['name'],
                        description=t['function']['description'],
                        parameters=json.dumps(t['function']['parameters'])
                    ) for t in msg['tools']
                ]

            message = Message(
                role=msg.get('role'),
                content=msg.get('content'),
                tool_calls=tool_calls,
                reasoning_content=msg.get('reasoning_content'),
            )
            messages.append(message)

        return Trajectory(messages=messages, tools=tools, user_data=[('loss', 'CrossEntropyLoss')])


class WorldVQAPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        # Convert base64 to PIL.Image
        import base64
        from io import BytesIO
        from PIL import Image
        base64_string = row['image']
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(base64_string)))

        return Trajectory(
            messages=[
                Message(role='system', content=SYSTEM),
                Message(role='user', content='<image>' + row['question']),
                Message(role='assistant', content=row['answer']),
            ],
            image=[image],
            user_data=[('loss', 'GRPOLoss')],
        )


class DocVQAPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='<image>' + row['question']),
            Message(role='assistant', content=row['answers'][0]),
        ], user_data=[('loss', 'GRPOLoss')])


class ArxivSummarization(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='Summarize this article: \n\n' + row['article']),
            Message(role='assistant', content=row['abstract']),
        ], user_data=[('loss', 'GRPOLoss')])


class LLaVAVideo178K(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        conversations = row.get('conversations', [])
        video = row.get('video')
        messages = [Message(role='system', content=SYSTEM)]

        role_map = {'human': 'user', 'gpt': 'assistant'}
        for conv in conversations:
            role = role_map.get(conv['from'], conv['from'])
            content = conv['value']
            videos = [video] if role == 'user' and '<image>' in content and video else None
            messages.append(Message(role=role, content=content, videos=videos))

        return Trajectory(messages=messages, user_data=[('loss', 'GRPOLoss')])


class VerifiableCoding(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        new_rows = []
        for row in rows:
            row = self.preprocess(row)
            new_rows.append(row)
        rows = self.map_row_to_col(new_rows)
        return rows

    def preprocess(self, row) -> Optional[Trajectory]:
        problem = row.get('prompt')
        answer = row.get('gold_standard_solution')
        verification_info = row.get('verification_info')
        if not problem or not answer or not verification_info:
            return Trajectory(messages=[], user_data=[])
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content=problem),
            Message(role='assistant', content=answer),
        ],
            user_data=[('verification_info', row['verification_info']), ('loss', 'GRPOLoss')])


class ComputerUse(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        problem = row.get('problem')
        answer = row.get('gold_standard_solution')
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content=problem),
            Message(role='assistant', content=answer),
        ], user_data=[('loss', 'GRPOLoss')])


class LatexOCR(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        text = row.get('text')
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='<image>Carefully read the image, and turn it into latex.'),
            Message(role='assistant', content=text),
        ], user_data=[('loss', 'GRPOLoss')])


class Condense(Preprocessor):
    THRESHOLD = 8192

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    @staticmethod
    def condense_sequence(text: str, ratio: float = 0.2, chunk_size: int = 512, start_index: int = 0) -> Tuple[
        str, List[str], int]:
        if not text or ratio >= 1.0:
            return text, [text] if text else [], 0

        # 1. 分句
        sentences = re.split(r'(?<=[.!?。！？])', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text, [text], 0

        # 2. 计算全文 TF-IDF
        all_words = []
        sentence_words = []
        for sent in sentences:
            words = re.findall(r'\b\w+\b|[\u4e00-\u9fff]', sent.lower())
            sentence_words.append(words)
            all_words.extend(words)

        if not all_words:
            return text, [text], 0

        # TF: 词频
        word_freq = Counter(all_words)
        total_words = len(all_words)
        tf = {w: freq / total_words for w, freq in word_freq.items()}

        # IDF: 包含该词的句子数
        n_sentences = len(sentences)
        doc_freq = Counter(w for words in sentence_words for w in set(words))
        idf = {w: math.log((n_sentences + 1) / (df + 1)) + 1 for w, df in doc_freq.items()}

        # TF-IDF 分数
        tfidf = {w: tf[w] * idf[w] for w in word_freq}

        # 3. 按块组织句子
        chunks = []
        current_chunk_sentences = []
        current_chunk_len = 0

        for sent in sentences:
            if current_chunk_len + len(sent) <= chunk_size:
                current_chunk_sentences.append(sent)
                current_chunk_len += len(sent)
            else:
                if current_chunk_sentences:
                    chunks.append(current_chunk_sentences)
                current_chunk_sentences = [sent]
                current_chunk_len = len(sent)
        if current_chunk_sentences:
            chunks.append(current_chunk_sentences)

        # 4. 对每个块内的句子进行压缩
        target_len = int(len(text) * ratio)
        condensed_chunks = []
        original_chunks = []

        for chunk_sents in chunks:
            original_text = ''.join(chunk_sents)
            original_chunks.append(original_text)

            # 对每个句子，保留高分词
            condensed_sents = []
            for sent in chunk_sents:
                words = re.findall(r'\b\w+\b|[\u4e00-\u9fff]', sent)
                if not words:
                    condensed_sents.append(sent)
                    continue

                # 计算每个词的分数，保留 top ratio 的词
                word_scores = [(w, tfidf.get(w.lower(), 0)) for w in words]
                word_scores.sort(key=lambda x: x[1], reverse=True)

                keep_count = max(1, int(len(words) * ratio))
                kept_words = set(w for w, _ in word_scores[:keep_count])

                # 重建句子，保留高分词和标点
                condensed = []
                for token in re.findall(r'\b\w+\b|[\u4e00-\u9fff]|[^\w\s]|\s+', sent):
                    if token in kept_words or not re.match(r'\b\w+\b|[\u4e00-\u9fff]', token):
                        condensed.append(token)
                condensed_sents.append(''.join(condensed))

            condensed_chunks.append(''.join(condensed_sents))

        # 5. 包裹标签
        result = '\n'.join(
            f'<chunk_{start_index + i}>{condensed_chunks[i]}</chunk_{start_index + i}>'
            for i in range(len(condensed_chunks))
        )

        return result, original_chunks, start_index + len(chunks)

    def preprocess(self, row: Trajectory) -> Trajectory:
        messages = row.get('messages')
        tool = Tool(
            tool_name='read_detail',
            description='Read the original uncompressed content of a specific chunk. '
                        'The user input has been compressed with key information preserved in <chunk_N> tags. '
                        'Call this tool when the compressed content is insufficient to answer accurately. '
                        'Minimize usage to avoid context length overflow.',
            parameters=json.dumps({
                'type': 'object',
                'properties': {
                    'block': {
                        'type': 'integer',
                        'description': 'The chunk number N from <chunk_N> tag to retrieve original content'
                    }
                },
                'required': ['block']
            }),
        )
        start_idx = 0
        chunks = []
        for message in messages:
            if message['role'] == 'user':
                if len(message['content']) > self.THRESHOLD and not (
                        message['images'] or message['audios'] or message['videos']):
                    condense_20, _chunks, start_idx = self.condense_sequence(message['content'], ratio=0.2,
                                                                             start_index=start_idx)
                    message['content'] = condense_20
                    chunks.extend(_chunks)
        if len(chunks) > 0:
            if not row.get('tools'):
                row['tools'] = [tool]
            else:
                row['tools'].append(tool)
            if not row.get('user_data'):
                row['user_data'] = []
            row['user_data'].append(('chunks', json.dumps(chunks)))
        return row


def create_dataset(template: str = 'Qwen3_5Template'):
    # 1. Text QA
    # 50000
    # step_flash_sft = DatasetMeta(
    #     dataset_id='/root/.cache/modelscope/hub/datasets/stepfun-ai/Step-3___5-Flash-SFT/json/general',
    #     data_slice=range(0, 50000),
    # )
    # dataset = LazyDataset(step_flash_sft)
    # dataset.map(StepFlashPreprocessor, dataset_meta=step_flash_sft)
    # 10000
    # arxiv_summarization = DatasetMeta(
    #     dataset_id='ms://ccdv/arxiv-summarization',
    #     subset_name='document',
    #     data_slice=range(0, 10000),
    # )
    # dataset = LazyDataset(arxiv_summarization)
    # dataset.add_dataset(arxiv_summarization)
    # dataset.map(ArxivSummarization, dataset_meta=arxiv_summarization)

    # 2. ImageQA
    # 3000
    # world_vqa = DatasetMeta(
    #     dataset_id='ms://moonshotai/WorldVQA',
    #     data_slice=range(0, 100),
    # )
    # dataset = LazyDataset(world_vqa)
    # dataset.map(WorldVQAPreprocessor, dataset_meta=world_vqa)
    # 5350
    doc_vqa = DatasetMeta(
        dataset_id='ms://lmms-lab/DocVQA',
        subset_name='DocVQA',
        split='validation',
        data_slice=range(0, 100),
    )
    dataset = LazyDataset(doc_vqa)
    # dataset.add_dataset(doc_vqa)
    dataset.map(DocVQAPreprocessor, dataset_meta=doc_vqa)
    # 2800
    info_graphic_vqa = DatasetMeta(
        dataset_id='ms://lmms-lab/DocVQA',
        subset_name='InfographicVQA',
        split='validation',
        data_slice=range(0, 100),
    )
    dataset.add_dataset(info_graphic_vqa)
    dataset.map(DocVQAPreprocessor, dataset_meta=info_graphic_vqa)

    # 3. VideoQA
    # 20000
    # llava_video_youtube_long = DatasetMeta(
    #     dataset_id='ms://lmms-lab/LLaVA-Video-178K',
    #     subset_name='2_3_m_youtube_v0_1',
    #     data_slice=range(0, 20000),
    # )
    # dataset.add_dataset(llava_video_youtube_long)
    # dataset.map(LLaVAVideo178K, dataset_meta=llava_video_youtube_long)
    # # 20000
    # llava_video_academic_long = DatasetMeta(
    #     dataset_id='ms://lmms-lab/LLaVA-Video-178K',
    #     subset_name='2_3_m_academic_v0_1',
    #     data_slice=range(0, 20000),
    # )
    # dataset.add_dataset(llava_video_academic_long)
    # dataset.map(LLaVAVideo178K, dataset_meta=llava_video_academic_long)
    # # 113
    # llava_video_nextqa_long = DatasetMeta(
    #     dataset_id='ms://lmms-lab/LLaVA-Video-178K',
    #     subset_name='2_3_m_nextqa',
    #     split='open_ended',
    # )
    # dataset.add_dataset(llava_video_nextqa_long)
    # dataset.map(LLaVAVideo178K, dataset_meta=llava_video_nextqa_long)

    # 4. Coding
    # 50000
    verifiable_coding = DatasetMeta(
        dataset_id='ms://PrimeIntellect/verifiable-coding-problems',
        data_slice=range(0, 100),
    )
    dataset.add_dataset(verifiable_coding)
    dataset.map(VerifiableCoding, dataset_meta=verifiable_coding)

    # 5. Computer Use
    # 50000
    # computer_use_large = DatasetMeta(
    #     dataset_id='ms://markov-ai/computer-use-large',
    #     subset_name='salesforce',
    #     data_slice=range(0, 100),
    #     # data_slice=range(0, 50000),
    # )
    # dataset.add_dataset(computer_use_large)
    # dataset.map(ComputerUse, dataset_meta=computer_use_large)

    # 6. OCR
    # 10000
    latex_ocr_default = DatasetMeta(
        dataset_id='ms://AI-ModelScope/LaTeX_OCR',
        subset_name='default',
        data_slice=range(0, 100),
        # data_slice=range(0, 10000),
    )
    dataset.add_dataset(latex_ocr_default)
    dataset.map(LatexOCR, dataset_meta=latex_ocr_default)

    # 10000
    latex_ocr_human_handwrite = DatasetMeta(
        dataset_id='ms://AI-ModelScope/LaTeX_OCR',
        subset_name='synthetic_handwrite',
        data_slice=range(0, 100),
        # data_slice=range(0, 10000),
    )
    dataset.add_dataset(latex_ocr_human_handwrite)
    dataset.map(LatexOCR, dataset_meta=latex_ocr_human_handwrite)

    # Text QA: ~60K
    # ImageQA: ~10K
    # VideoQA: ~40K
    # Coding: ~10K
    # Computer Use: 50K
    # OCR: 20K
    dataset.mix_dataset(interleave=False)
    dataset.map(Condense)
    dataset.set_template(template, model_id=MODEL_ID)
    dataset.encode()
    return dataset


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

    def construct_messages(batch):
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
                Message(role='user',
                        content=f'The Query is: {user_info}, the response is {response["content"]}, now generate your augmentation:'),
            ]
            all_trajectory.append(Trajectory(messages=_new_messages))
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


def length_reward(trajectories: List[Dict[str, Any]], scale: float = 8192.0) -> List[float]:
    rewards = []
    for trajectory in trajectories:
        length = 0
        for i, message in enumerate(trajectory['messages']):
            if message['role'] == 'assistant' and 'read_detail' in message['content'] and i < len(
                    trajectory['messages']) - 1:
                length += len(trajectory['messages'][i + 1]['content'])
        # exp(-length / scale): length=0 → 1, length→∞ → 0
        reward = np.exp(-length / scale)
        rewards.append(reward)
    return rewards


def kl_reward(hidden_topk_logprobs: List[List[List[Tuple[int, float]]]],
              ground_topk_logprobs: List[List[List[Tuple[int, float]]]],
              scale: float = 1.0) -> List[float]:
    rewards = []
    for hidden_seq, ground_seq in zip(hidden_topk_logprobs, ground_topk_logprobs):
        if not hidden_seq or not ground_seq:
            rewards.append(1.0)
            continue

        # 对齐序列长度
        min_len = min(len(hidden_seq), len(ground_seq))
        total_kl = 0.0
        valid_positions = 0

        for pos in range(min_len):
            hidden_topk = hidden_seq[pos]
            ground_topk = ground_seq[pos]

            if hidden_topk is None or ground_topk is None:
                continue

            # 构建 token -> logprob 映射
            hidden_dict = {tok: logp for tok, logp in hidden_topk}
            ground_dict = {tok: logp for tok, logp in ground_topk}

            # 计算共同 token 的 KL 散度
            # KL(P || Q) = Σ P(x) * (log P(x) - log Q(x))
            common_tokens = set(hidden_dict.keys()) & set(ground_dict.keys())
            if not common_tokens:
                continue

            # 归一化 ground truth 分布 (P) 在共同 token 上
            ground_logps = np.array([ground_dict[t] for t in common_tokens])
            ground_probs = np.exp(ground_logps - np.max(ground_logps))  # 数值稳定
            ground_probs = ground_probs / ground_probs.sum()

            # 获取 hidden 分布 (Q) 的 log 概率
            hidden_logps = np.array([hidden_dict[t] for t in common_tokens])
            hidden_probs = np.exp(hidden_logps - np.max(hidden_logps))
            hidden_probs = hidden_probs / hidden_probs.sum()

            # KL(P || Q) = Σ P * log(P / Q)
            kl = np.sum(ground_probs * (np.log(ground_probs + 1e-10) - np.log(hidden_probs + 1e-10)))
            total_kl += max(0, kl)  # KL >= 0
            valid_positions += 1

        if valid_positions > 0:
            avg_kl = total_kl / valid_positions
            # reward = exp(-kl / scale): kl=0 -> 1, kl大 -> 0
            reward = np.exp(-avg_kl / scale)
        else:
            reward = 1.0

        rewards.append(reward)

    return rewards


def compute_rewards(
        trajectories: List[Dict[str, Any]],
        hidden_topk_logprobs: List[List[List[Tuple[int, float]]]],
        ground_topk_logprobs: List[List[List[Tuple[int, float]]]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    accuracy_reward_fn = MathReward()
    accuracy_rewards = accuracy_reward_fn(trajectories, trajectories)
    length_rewards = length_reward(trajectories)
    kl_rewards = kl_reward(hidden_topk_logprobs, ground_topk_logprobs)
    total_rewards = [a + l + k for a, l, k in zip(accuracy_rewards, length_rewards, kl_rewards)]
    return total_rewards, length_rewards, accuracy_rewards, kl_rewards


def multi_round_sample(samples: List[Trajectory], sampler: vLLMSampler, sampling_params: SamplingParams,
                       num_generations, template, max_round=10) -> List[Trajectory]:
    results = samples * num_generations
    for r in results:
        r['done'] = False
    for i in range(max_round):
        responses = sampler.sample([r for r in results if not r['done']], sampling_params=sampling_params)
        for j, response in enumerate(responses):
            new_input_features = response.sequences[0].new_input_feature
            new_input_features.pop('input_ids', None)
            results[j] = new_input_features
            last_content = new_input_features['messages'][-1]['content']
            output_dict = template.tokenizer.parse_response(last_content)
            tool_calls = None
            if output_dict.get('tool_calls'):
                tool_calls = [
                    ToolCall(
                        tool_name=tc['function']['name'],
                        arguments=json.dumps(tc['function']['arguments']) if isinstance(tc['function']['arguments'],
                                                                                        dict) else tc['function'][
                            'arguments']
                    ) for tc in output_dict['tool_calls']
                ]
            if not tool_calls:
                results[j]['done'] = True
                logprobs = [logprob[1] for logprob in response.prompt_logprobs]
                logprobs += [logprob[1] for logprob in response.sequences[0].logprobs]
                results[j]['logprobs'] = logprobs
                # 存储 topk_prompt_logprobs 用于 KL 计算
                results[j]['topk_prompt_logprobs'] = response.topk_prompt_logprobs
            else:
                for tool_call in tool_calls:
                    arguments = json.loads(tool_call['arguments'])
                    if tool_call['tool_name'] == 'read_detail':
                        block = arguments['block']
                        blocks = \
                        [user_data[1] for user_data in new_input_features['user_data'] if user_data[0] == 'chunks'][0]
                        block_content = blocks[block]
                        new_input_features['messages'].append(
                            Message(role='tool', content=f'Block {block}:{block_content}'))
        if all(r['done'] for r in results):
            break
    return results


def compute_lobprobs_given_ground_truth(trajectories: List[Trajectory], sampler: vLLMSampler):
    SYSTEM = """
You are a helpful assistant to help me to solve problems. You will be given a question, and the ground truth answer will be given to you too.
You need to think about the question and give a better answer according to the question/ground truth.
"""

    sampling_params = SamplingParams(max_tokens=0, prompt_logprobs=64)
    for trajectory in trajectories:
        ground_truth = [data[1] for data in trajectory['user_data'] if data[0] == 'ground_truth'][0]
        messages = trajectory['messages']
        messages[0]['content'] = SYSTEM
        message = [m for m in messages if m['role'] == 'user'][0]
        message['content'] = message['content'] + '\n\nThe ground truth of this problem is:\n' + ground_truth

    sample_responses = sampler.sample(trajectories, sampling_params=sampling_params)

    all_topk_logprobs = []
    for response in sample_responses:
        all_topk_logprobs.append(response.topk_prompt_logprobs)
    return all_topk_logprobs


def main():
    dataloader = DataLoader(
        dataset=create_dataset,
        batch_size=BATCH_SIZE,
        # device_mesh=model_mesh,
        # remote_group='model',
    )
    # set sampler and model separate to use different gpus
    device_groups = [
        DeviceGroup(name='model', ranks=MODEL_GPUS, device_type='GPU'),
        DeviceGroup(name='sampler', ranks=SAMPLER_GPUS, device_type='GPU'),
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

    if USE_MEGATRON:
        model.set_optimizer('default', lr=LEARNING_RATE)
        model.set_lr_scheduler('default', lr_decay_steps=len(dataloader), max_lr=LEARNING_RATE)
    else:
        model.set_optimizer('AdamW', lr=LEARNING_RATE)
        model.set_lr_scheduler('CosineAnnealingLR', T_max=len(dataloader), eta_min=0)
    # model.set_loss('GRPOLoss', epsilon=0.2)
    model.set_processor(InputProcessor)
    model.set_template('Qwen3_5Template', model_id=MODEL_ID)

    sampler = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'gpu_memory_utilization': 0.8,
            'max_model_len': 32000,
        },
        device_mesh=sampler_mesh,
        remote_group='sampler',
    )
    sampler.set_template('Qwen3_5Template', model_id=MODEL_ID)

    ckpt_manager = CheckpointEngineManager(model=model, sampler=sampler)

    advantage_fn = GRPOAdvantage()
    metrics = CompletionRewardMetric()

    sampling_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, num_samples=1, logprobs=1, prompt_logprobs=64)

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
            all_input_data.append(sample_response['messages'])
            all_old_logps.append(sample_response['logprobs'])
            all_completion_lengths.append(sum(np.where(sample_response['labels'] == -100, 0, 1)))

        # 获取 hidden context 和 ground truth context 的 topk logprobs
        hidden_topk_logprobs = [r.get('topk_prompt_logprobs', []) for r in sample_responses]
        ground_topk_logprobs = compute_lobprobs_given_ground_truth(trajectories=all_input_data, sampler=sampler)

        total_rewards, length_rewards, accuracy_rewards, kl_rewards = compute_rewards(
            all_input_data, hidden_topk_logprobs, ground_topk_logprobs
        )
        metrics.accumulate(
            completion_lengths=all_completion_lengths,
            rewards={
                'total': total_rewards,
                'length': length_rewards,
                'accuracy': accuracy_rewards,
                'kl': kl_rewards,
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

            log_dict = metrics.calculate()
            log_dict.update(model.calculate_metric(is_training=True))
            metrics.reset()
            logger.info(f'[Step {optim_step}/{MAX_STEPS}] {log_dict}')

    logger.info(f'Training completed. optim_steps={optim_step}')
    model.save('grpo-gsm8k-checkpoint')


if __name__ == '__main__':
    main()
