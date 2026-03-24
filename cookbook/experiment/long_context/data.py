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
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='<image>' + row['question'], images=[row['image']]),
            Message(role='assistant', content=row['answer']),
        ], user_data=[('loss', 'GRPOLoss')])


class DocVQAPreprocessor(Preprocessor):

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='<image>' + row['question'], images=[row['image']]),
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
            if row is None:
                continue
            new_rows.append(row)
        rows = self.map_row_to_col(new_rows)
        return rows

    def preprocess(self, row) -> Optional[Trajectory]:
        problem = row.get('problem')
        answer = row.get('gold_standard_solution')
        verification_info = row.get('verification_info')
        if not answer or not verification_info:
            return None
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
        image = row.get('image')
        text = row.get('text')
        return Trajectory(messages=[
            Message(role='system', content=SYSTEM),
            Message(role='user', content='<image>Carefully read the image, and turn it into latex.', images=[image]),
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
    def condense_sequence(text: str, ratio: float = 0.2, chunk_size: int = 512, start_index: int = 0) -> Tuple[str, List[str], int]:
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
                if len(message['content']) > self.THRESHOLD and not (message['images'] or message['audios'] or message['videos']):
                    condense_20, _chunks, start_idx = self.condense_sequence(message['content'], ratio=0.2, start_index=start_idx)
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


def create_dataset(template: str='Qwen3_5Template'):
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
    dataset.map(Condense)
    dataset.set_template(template)
    dataset.encode()
    return dataset
