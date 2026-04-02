# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward functions for OlympiadBench math/physics problems.

Three core rewards, all normalized to [0, 1]:
- Accuracy: Answer correctness with partial credit
- Format: Answer formatting and consistency
- Quality: Reasoning structure, length, and repetition
"""
import math
import re
from typing import Any, Dict, List

from twinkle.reward.base import Reward


def _get_completion(trajectory: Dict[str, Any]) -> str:
    """Extract assistant completion from trajectory."""
    messages = trajectory.get('messages', [])
    for msg in reversed(messages):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, list):
                return ''.join(
                    block.get('text', '') for block in content
                    if isinstance(block, dict) and block.get('type') == 'text'
                )
            return content
    return ''


def _extract_boxed_answers(text: str) -> List[str]:
    """Extract all answers from \\boxed{} in text."""
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    return [m.replace(' ', '').strip() for m in matches]


def _normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    answer = re.sub(r'\s+', '', answer)
    answer = re.sub(r'(cm|mm|m|kg|g|s|°|度|米|千克|克|秒)', '', answer, flags=re.IGNORECASE)
    return answer.strip()


def _is_numeric_match(pred: str, gt: str, tolerance: float = 0.01) -> bool:
    """Check if two values match numerically."""
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        if gt_val == 0:
            return abs(pred_val) < tolerance
        return abs(pred_val - gt_val) / abs(gt_val) < tolerance
    except (ValueError, OverflowError):
        return False


def _numeric_similarity(pred: str, gt: str) -> float:
    """Return similarity score [0, 1] for numeric values."""
    try:
        pred_val = float(pred)
        gt_val = float(gt)
        if gt_val == 0:
            return 1.0 if abs(pred_val) < 0.01 else 0.0
        relative_error = abs(pred_val - gt_val) / abs(gt_val)
        return math.exp(-5 * relative_error)
    except (ValueError, OverflowError):
        return 0.0


class OlympiadBenchAccuracyReward(Reward):
    """Accuracy reward with partial credit for close answers.

    Returns:
        1.0: Exact match
        0.5-0.99: Close numeric match (partial credit)
        0.0: Wrong answer
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = _get_completion(trajectory)
            predicted = _extract_boxed_answers(completion)

            user_data = trajectory.get('user_data', [])
            gt = ''
            is_multiple = False
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = item[1]
                elif item[0] == 'is_multiple_answer':
                    is_multiple = item[1]

            if not predicted or not gt:
                rewards.append(0.0)
                continue

            gt_parts = [_normalize_answer(g.strip()) for g in gt.split(',')]

            if not is_multiple:
                pred = _normalize_answer(predicted[-1])
                gt_val = gt_parts[0] if gt_parts else ''

                if pred == gt_val or _is_numeric_match(pred, gt_val):
                    rewards.append(1.0)
                else:
                    sim = _numeric_similarity(pred, gt_val)
                    rewards.append(sim * 0.5)
            else:
                pred_normalized = [_normalize_answer(p) for p in predicted]
                correct_count = 0
                for gt_val in gt_parts:
                    for pred in pred_normalized:
                        if pred == gt_val or _is_numeric_match(pred, gt_val):
                            correct_count += 1
                            break
                rewards.append(correct_count / len(gt_parts) if gt_parts else 0.0)

        return rewards


class OlympiadBenchFormatReward(Reward):
    """Format reward: answer formatting and consistency.

    Combines:
        - Has \\boxed{} answer (0.4)
        - Answer at/near end (0.3)
        - Single consistent answer (0.3)
        - Penalize multiple conflicting answers
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = _get_completion(trajectory)
            score = 0.0

            boxed_answers = _extract_boxed_answers(completion)

            if boxed_answers:
                score += 0.4

                # Answer near end
                last_boxed_pos = completion.rfind('\\boxed{')
                if last_boxed_pos >= 0 and len(completion) - last_boxed_pos < 200:
                    score += 0.3

                # Answer consistency
                unique_answers = set(_normalize_answer(a) for a in boxed_answers)
                if len(unique_answers) == 1:
                    score += 0.3
                elif len(unique_answers) > 2:
                    score -= 0.3

            rewards.append(max(0.0, min(1.0, score)))

        return rewards


class OlympiadBenchQualityReward(Reward):
    """Quality reward: reasoning, length, and repetition combined.

    Three components (each ~0.33):
        - Reasoning: equation chains, logical structure (0.4)
        - Length: smooth curve favoring 300-1500 chars (0.3)
        - Repetition: penalize repeated content (0.3)
    """

    def _reasoning_score(self, completion: str) -> float:
        """Score reasoning quality (0-1)."""
        score = 0.0

        # Equation chains
        if len(re.findall(r'=', completion)) >= 3:
            score += 0.25

        # Mathematical operators
        if len(re.findall(r'[+\-*/^]|\\frac|\\sqrt', completion)) >= 5:
            score += 0.25

        # Logical structure
        logic_patterns = [
            r'第[一二三四五六七八九十\d]+步',
            r'Step\s*\d+',
            r'首先|其次|然后|最后|因此|所以',
            r'First|Then|Finally|Therefore|Hence',
        ]
        if any(re.search(p, completion, re.IGNORECASE) for p in logic_patterns):
            score += 0.25

        # Conclusion derivation
        last_500 = completion[-500:] if len(completion) > 500 else completion
        if re.search(r'(所以|因此|答案|故|得|Thus|Therefore|Answer).*\\boxed', last_500, re.IGNORECASE | re.DOTALL):
            score += 0.25

        return min(1.0, score)

    def _length_score(self, length: int) -> float:
        """Smooth length score (0-1)."""
        if length < 100:
            return length / 100 * 0.3
        elif length < 300:
            return 0.3 + (length - 100) / 200 * 0.7
        elif length <= 1500:
            return 1.0
        elif length <= 3000:
            t = (length - 1500) / 1500
            return 0.5 + 0.5 * math.cos(t * math.pi / 2)
        else:
            t = min((length - 3000) / 3000, 1.0)
            return 0.5 * (1 - t * 0.5)

    def _repetition_score(self, completion: str) -> float:
        """Score based on content uniqueness (0-1, higher is better)."""
        if len(completion) < 100:
            return 1.0

        # Check chunk uniqueness
        chunk_size = 50
        chunks = [completion[i:i+chunk_size] for i in range(0, len(completion) - chunk_size, chunk_size // 2)]
        if not chunks:
            return 1.0

        unique_ratio = len(set(chunks)) / len(chunks)

        # Check n-gram uniqueness
        words = completion.split()
        if len(words) >= 4:
            ngrams = [' '.join(words[i:i+4]) for i in range(len(words) - 3)]
            ngram_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
        else:
            ngram_ratio = 1.0

        return (unique_ratio + ngram_ratio) / 2

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = _get_completion(trajectory)

            reasoning = self._reasoning_score(completion)
            length = self._length_score(len(completion))
            repetition = self._repetition_score(completion)

            # Weighted combination
            score = 0.4 * reasoning + 0.3 * length + 0.3 * repetition
            rewards.append(score)

        return rewards
