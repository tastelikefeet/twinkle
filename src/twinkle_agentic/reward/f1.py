import re
import string
from typing import List, Dict, Any, Tuple, Counter

from twinkle.reward import Reward

_BOXED_RE = re.compile(r'\\boxed\{([^}]*)\}')

def _extract_final_answer(completion: str) -> str:
    matches = _BOXED_RE.findall(completion or '')
    return matches[-1].strip() if matches else ''


def _last_assistant_text(traj: Dict[str, Any]) -> str:
    for msg in reversed(traj.get('messages', [])):
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content') or ''
        if isinstance(content, str):
            return content
        return '\n'.join(
            p.get('text', '') for p in content
            if isinstance(p, dict) and p.get('type') == 'text')
    return ''


def _stem(tok: str) -> str:
    from nltk.stem import PorterStemmer
    return PorterStemmer().stem(tok) if len(tok) >= 4 and tok.isalpha() else tok


def _normalize_answer(s: str) -> str:
    s = (s or '').lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(_stem(t) for t in s.split())


def _f1_score(prediction: str, gold: str) -> Tuple[float, float]:
    filler_tokens: frozenset = frozenset([
        'long', 'tall', 'high', 'wide', 'deep', 'heavy', 'old', 'large',
        'small', 'big', 'short', 'away', 'ago', 'approximately', 'about',
        'around', 'over', 'under', 'below', 'above', 'total', 'roughly',
        'nearly', 'almost', 'exactly',
    ])
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        em = float(pred_tokens == gold_tokens)
        return em, em
    em = float(pred_tokens == gold_tokens)
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, em
    p = num_same / len(pred_tokens)
    r = num_same / len(gold_tokens)
    f1 = 2 * p * r / (p + r)

    pred_set, gold_set = set(pred_tokens), set(gold_tokens)
    if gold_set < pred_set:
        extras = pred_set - gold_set
        if all(t.isdigit() or t in filler_tokens for t in extras):
            return 1.0, em
    if pred_set < gold_set:
        missing = gold_set - pred_set
        if all(t in filler_tokens for t in missing):
            return 1.0, em
    return f1, em


class HotpotQAF1Reward(Reward):
    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for traj in trajectories:
            gold = ''
            for key, val in traj.get('user_data', []) or []:
                if key == 'ground_truth':
                    gold = val or ''
                    break
            pred = _extract_final_answer(_last_assistant_text(traj))
            f1, _ = _f1_score(pred, gold)
            rewards.append(f1)
        return rewards


class HotpotQACoTReward(Reward):
    _STEP1_RE = re.compile(r'step\s*1\b', re.IGNORECASE)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for t in trajectories:
            msgs = t.get('messages', [])

            # Concatenate all assistant turns
            all_assistant_text = ' '.join(
                m.get('content', '') or ''
                for m in msgs
                if m.get('role') == 'assistant' and isinstance(m.get('content'), str)
            )

            if self._STEP1_RE.search(all_assistant_text):
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return rewards


class HotpotQAToolExploreReward(Reward):
    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards: List[float] = []
        for t in trajectories:
            has_tool_call = any(
                m.get('role') == 'assistant' and m.get('tool_calls')
                for m in t.get('messages', []))
            rewards.append(1.0 if has_tool_call else 0.0)
        return rewards

