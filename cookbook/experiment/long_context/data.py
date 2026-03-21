from twinkle.dataset import DatasetMeta, LazyDataset
from twinkle.preprocessor import Preprocessor
from typing import Any, Dict, List, Tuple
from collections import Counter
import math
import re

from twinkle.data_format import Message, Tool, ToolCall, Trajectory
import json


class StepFlashPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        raw_messages = row.get('messages', [])
        messages = []
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

        return Trajectory(messages=messages, tools=tools)


class WorldVQAPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='user', content='<image>' + row['question'], images=[row['image']]),
            Message(role='assistant', content=row['answer']),
        ])


class DocVQAPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='user', content='<image>' + row['question'], images=[row['image']]),
            Message(role='assistant', content=row['answers'][0]),
        ])


class ArxivSummarization(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='user', content='Summarize this article: \n\n' + row['article']),
            Message(role='assistant', content=row['abstract']),
        ])


class LLaVAVideo178K(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        conversations = row.get('conversations', [])
        video = row.get('video')
        messages = []

        role_map = {'human': 'user', 'gpt': 'assistant'}
        for conv in conversations:
            role = role_map.get(conv['from'], conv['from'])
            content = conv['value']
            videos = [video] if role == 'user' and '<image>' in content and video else None
            messages.append(Message(role=role, content=content, videos=videos))

        return Trajectory(messages=messages)


class VerifiableCoding(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        new_rows = []
        for row in rows:
            row = self.preprocess(row)
            if row is None:
                continue
            new_rows.append(row)
        rows = self.map_row_to_col(new_rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        problem = row.get('problem')
        answer = row.get('gold_standard_solution')
        verification_info = row.get('verification_info')
        if not answer or not verification_info:
            return None
        return Trajectory(messages=[
            Message(role='user', content=problem),
            Message(role='assistant', content=answer),
        ],
        user_data=[('verification_info', row['verification_info'])])


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
            Message(role='user', content=problem),
            Message(role='assistant', content=answer),
        ])


class LatexOCR(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        image = row.get('image')
        text = row.get('text')
        return Trajectory(messages=[
            Message(role='user', content='<image>Carefully read the image, and turn it into latex.', images=[image]),
            Message(role='assistant', content=text),
        ])


class Condense(Preprocessor):

    THRESHOLD = 8192

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    @staticmethod
    def condense_sequence(text: str, ratio: float = 0.5, chunk_size: int = 512) -> Tuple[str, List[str]]:
        if not text or ratio >= 1.0:
            return text, [text] if text else []

        sentences = re.split(r'(?<=[.!?。！？\n])', text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) <= chunk_size:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk.strip())

        chunks = [c for c in chunks if c]
        if len(chunks) <= 1:
            return text[:int(len(text) * ratio)], chunks

        chunk_words = [chunk.lower().split() for chunk in chunks]
        all_words = [w for words in chunk_words for w in words]
        word_freq = Counter(all_words)

        n_chunks = len(chunks)
        doc_freq = Counter(w for words in chunk_words for w in set(words))
        idf = {w: math.log(n_chunks / (df + 1)) for w, df in doc_freq.items()}

        scores = []
        for words in chunk_words:
            score = sum(word_freq[w] * idf.get(w, 0) for w in words) / (len(words) + 1)
            scores.append(score)

        target_len = int(len(text) * ratio)
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)

        selected = []
        current_len = 0
        for idx in ranked:
            if current_len + len(chunks[idx]) <= target_len:
                selected.append(idx)
                current_len += len(chunks[idx])

        selected.sort()
        condensed = '\n'.join(f'<chunk_{i}>{chunks[i]}</chunk_{i}>' for i in selected)
        return condensed, chunks

    def preprocess(self, row) -> Trajectory:
        messages = row.get('messages')
        tool = Tool(
            tool_name='read_detail',
            description='Use this tool to read the original text. Use it when the current compressed information is insufficient for you to accurately answer the user\'s request. '
                        'You should read as little of the original information as possible to prevent the input sequence from becoming too long.',
            parameters='[{"name":"block","type":"int","description":"The condensed block number"}]',
        )
        for message in messages:
            if message['role'] == 'user':
                if len(message['content']) > self.THRESHOLD and not (message['images'] or message['audios'] or message['videos']):
                    condense_20, chunks = self.condense_sequence(message['content'], ratio=0.2)
                    message['content'] = condense_20




def create_dataset(template: str):
    # 1. Text QA
    # 50000
    step_flash_sft = DatasetMeta(
        dataset_id='ms://stepfun-ai/Step-3.5-Flash-SFT',
        data_slice=range(0, 50000),
    )
    dataset = LazyDataset(step_flash_sft)
    dataset.map(StepFlashPreprocessor, dataset_meta=step_flash_sft)
    # 10000
    arxiv_summarization = DatasetMeta(
        dataset_id='ms://ccdv/arxiv-summarization',
        subset_name='document',
        data_slice=range(0, 10000),
    )
    dataset.add_dataset(arxiv_summarization)
    dataset.map(ArxivSummarization, dataset_meta=arxiv_summarization)

    # 2. ImageQA
    # 3000
    world_vqa = DatasetMeta(
        dataset_id='ms://moonshotai/WorldVQA',
    )
    dataset.add_dataset(world_vqa)
    dataset.map(WorldVQAPreprocessor, dataset_meta=world_vqa)
    # 5350
    doc_vqa = DatasetMeta(
        dataset_id='ms://lmms-lab/DocVQA',
        subset_name='DocVQA',
        split='validation',
    )
    dataset.add_dataset(doc_vqa)
    dataset.map(DocVQAPreprocessor, dataset_meta=doc_vqa)
    # 2800
    info_graphic_vqa = DatasetMeta(
        dataset_id='ms://lmms-lab/DocVQA',
        subset_name='InfographicVQA',
        split='validation',
    )
    dataset.add_dataset(info_graphic_vqa)
    dataset.map(DocVQAPreprocessor, dataset_meta=info_graphic_vqa)

    # 3. VideoQA
    # 20000
    llava_video_youtube_long = DatasetMeta(
        dataset_id='ms://lmms-lab/LLaVA-Video-178K',
        subset_name='2_3_m_youtube_v0_1',
        data_slice=range(0, 20000),
    )
    dataset.add_dataset(llava_video_youtube_long)
    dataset.map(LLaVAVideo178K, dataset_meta=llava_video_youtube_long)
    # 20000
    llava_video_academic_long = DatasetMeta(
        dataset_id='ms://lmms-lab/LLaVA-Video-178K',
        subset_name='2_3_m_academic_v0_1',
        data_slice=range(0, 20000),
    )
    dataset.add_dataset(llava_video_academic_long)
    dataset.map(LLaVAVideo178K, dataset_meta=llava_video_academic_long)
    # 113
    llava_video_nextqa_long = DatasetMeta(
        dataset_id='ms://lmms-lab/LLaVA-Video-178K',
        subset_name='2_3_m_nextqa',
        split='open_ended',
    )
    dataset.add_dataset(llava_video_nextqa_long)
    dataset.map(LLaVAVideo178K, dataset_meta=llava_video_nextqa_long)

    # 4. Coding
    # 50000
    verifiable_coding = DatasetMeta(
        dataset_id='ms://PrimeIntellect/verifiable-coding-problems',
    )
    dataset.add_dataset(verifiable_coding)
    dataset.map(VerifiableCoding, dataset_meta=verifiable_coding)

    # 5. Computer Use
    # 50000
    computer_use_large = DatasetMeta(
        dataset_id='ms://markov-ai/computer-use-large',
        data_slice=range(0, 50000),
    )
    dataset.add_dataset(computer_use_large)
    dataset.map(ComputerUse, dataset_meta=computer_use_large)

    # 6. OCR
    # 10000
    latex_ocr_default = DatasetMeta(
        dataset_id='ms://AI-ModelScope/LaTeX_OCR',
        subset_name='default',
        data_slice=range(0, 10000),
    )
    dataset.add_dataset(latex_ocr_default)
    dataset.map(LatexOCR, dataset_meta=latex_ocr_default)

    # 10000
    latex_ocr_human_handwrite = DatasetMeta(
        dataset_id='ms://AI-ModelScope/LaTeX_OCR',
        subset_name='synthetic_handwrite',
        data_slice=range(0, 10000),
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
    dataset.map()
    dataset.set_template(template)
    dataset.encode()
    return dataset
