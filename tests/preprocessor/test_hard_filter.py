# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for HardFilter.

HardFilter drops:
  Rule 1 — Single-turn trivial query (greeting / bare wh-question).
  Rule 2 — Two-turn shallow assistant reply (< min chars, no thinking chain).

CJK and ASCII branches use different length thresholds because of the
information density gap.
"""
import pytest

from twinkle_agentic.preprocessor.hard_filter import (
    HardFilter,
    _cjk_ratio,
    _has_thinking,
    _is_simple_query,
)


def _row(messages):
    return {'messages': messages}


# ── _cjk_ratio ───────────────────────────────────────────────────────────────

class TestCjkRatio:
    def test_pure_ascii(self):
        assert _cjk_ratio('hello world') == 0.0

    def test_pure_chinese(self):
        assert _cjk_ratio('你好世界') == 1.0

    def test_mixed(self):
        # 2 CJK chars / 6 total
        assert abs(_cjk_ratio('hi你好zz') - 2 / 6) < 1e-9

    def test_japanese_hiragana(self):
        # Hiragana is in the CJK range covered by the regex.
        assert _cjk_ratio('こんにちは') == 1.0

    def test_korean_hangul(self):
        assert _cjk_ratio('안녕하세요') == 1.0

    def test_empty(self):
        # max(len, 1) → 0/1 = 0
        assert _cjk_ratio('') == 0.0


# ── _is_simple_query: ASCII / English ────────────────────────────────────────

class TestSimpleQueryEnglish:
    def test_short_text_is_simple(self):
        assert _is_simple_query('hi') is True
        assert _is_simple_query('a' * 9) is True  # default min=10

    def test_at_threshold_not_simple_unless_pattern(self):
        # 10 non-pattern chars escapes both length and pattern checks
        assert _is_simple_query('quantum xx') is False

    def test_greeting_hello(self):
        assert _is_simple_query('Hello!') is True
        assert _is_simple_query('Heeellloooo') is True

    def test_greeting_good_morning(self):
        assert _is_simple_query('Good morning') is True

    def test_greeting_how_are_you(self):
        assert _is_simple_query('How are you') is True

    def test_bare_wh_question(self):
        assert _is_simple_query('what is python') is True

    def test_imperative_short(self):
        assert _is_simple_query('tell me about it') is True
        assert _is_simple_query('explain') is True

    def test_substantive_question_not_simple(self):
        # Long, technical question should pass (not simple).
        text = ('Please explain the difference between gradient descent and '
                'momentum-based optimization in deep learning training.')
        assert _is_simple_query(text) is False


class TestSimpleQueryChinese:
    def test_short_cjk_is_simple(self):
        assert _is_simple_query('你好') is True
        assert _is_simple_query('你好啊') is True  # < 6

    def test_at_cjk_threshold(self):
        # 6 CJK chars; greeting (`你好+` matches `你好好好好好`) → simple
        assert _is_simple_query('你好好好好好') is True
        # 6 substantive CJK chars; no greeting/simple pattern → NOT simple
        assert _is_simple_query('量子计算原理') is False

    def test_greeting_zh(self):
        assert _is_simple_query('你好！') is True
        assert _is_simple_query('早上好') is True
        assert _is_simple_query('哈喽哈喽') is True

    def test_what_is_x(self):
        assert _is_simple_query('什么是机器学习？') is True
        assert _is_simple_query('梯度下降是什么？') is True

    def test_substantive_zh_not_simple(self):
        text = '请详细解释一下变换器架构中的多头自注意力机制是如何并行计算的，以及为什么需要位置编码。'
        assert _is_simple_query(text) is False


class TestSimpleQueryJapanese:
    def test_japanese_greeting(self):
        assert _is_simple_query('こんにちは') is True

    def test_japanese_what_is(self):
        assert _is_simple_query('機械学習とは何ですか') is True


class TestSimpleQueryKorean:
    def test_korean_greeting(self):
        assert _is_simple_query('안녕하세요') is True

    def test_korean_what_is(self):
        # KO_SIMPLE_RE expects "X이/가 뭐" pattern; trailing 인가요/까요 are
        # only single optional chars, so use the bare 뭐 form here.
        assert _is_simple_query('머신러닝이 뭐') is True


class TestSimpleQueryEdge:
    def test_empty(self):
        assert _is_simple_query('') is True

    def test_whitespace_only(self):
        assert _is_simple_query('   \n  ') is True

    def test_custom_thresholds(self):
        # Raise the bar so a 12-char query becomes simple.
        text = 'short query!'
        assert _is_simple_query(text, min_user_chars=20) is True
        assert _is_simple_query(text, min_user_chars=5) is False


# ── _has_thinking ────────────────────────────────────────────────────────────

class TestHasThinking:
    def test_thinking_field_long_enough(self):
        msg = {'thinking': 'a' * 250}
        assert _has_thinking(msg) is True

    def test_thinking_field_too_short(self):
        msg = {'thinking': 'short'}
        assert _has_thinking(msg) is False

    def test_reasoning_content_alias(self):
        msg = {'reasoning_content': 'a' * 250}
        assert _has_thinking(msg) is True

    def test_no_thinking(self):
        assert _has_thinking({'content': 'reply'}) is False

    def test_custom_min_chars(self):
        msg = {'thinking': 'short'}
        assert _has_thinking(msg, min_chars=3) is True

    def test_non_string_thinking_truthy(self):
        # Falls through to bool(thinking)
        assert _has_thinking({'thinking': {'a': 1}}) is True
        assert _has_thinking({'thinking': []}) is False


# ── HardFilter pipeline ──────────────────────────────────────────────────────

def _fil(rows, **kw):
    return HardFilter(**kw)(rows)


class TestRule1SimpleQuery:
    def test_drops_greeting_only(self):
        rows = [_row([
            {'role': 'user', 'content': 'hello'},
            {'role': 'assistant', 'content': 'hi there!'},
        ])]
        assert _fil(rows) == []

    def test_drops_bare_wh_question(self):
        rows = [_row([
            {'role': 'user', 'content': 'what is AI'},
            {'role': 'assistant', 'content': 'a short answer'},
        ])]
        assert _fil(rows) == []

    def test_keeps_when_substantive(self):
        rows = [_row([
            {'role': 'user', 'content':
                'Could you explain gradient descent step by step in detail?'},
            {'role': 'assistant', 'content':
                'Gradient descent is an iterative optimization algorithm... ' * 5},
        ])]
        assert len(_fil(rows)) == 1

    def test_keeps_simple_query_with_thinking(self):
        # Rule 1 rescue: thinking chain ≥200 chars saves the row.
        rows = [_row([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello',
             'reasoning_content': 'Now I need to greet politely... ' * 20},
        ])]
        assert len(_fil(rows)) == 1

    def test_simple_query_no_assistant_dropped(self):
        # No assistant turn → no thinking → dropped.
        rows = [_row([{'role': 'user', 'content': 'hi'}])]
        assert _fil(rows) == []


class TestRule2ShallowReply:
    def test_drops_short_reply(self):
        rows = [_row([
            {'role': 'user', 'content':
                'Explain the difference between A and B in detail please.'},
            {'role': 'assistant', 'content': 'A is good.'},  # < 80 chars
        ])]
        assert _fil(rows) == []

    def test_keeps_long_reply(self):
        rows = [_row([
            {'role': 'user', 'content':
                'Explain the difference between A and B in detail please.'},
            {'role': 'assistant', 'content':
                'A and B differ in several ways. ' * 5},
        ])]
        assert len(_fil(rows)) == 1

    def test_short_reply_with_thinking_kept(self):
        # Rule 2 rescue: thinking saves a short final reply.
        rows = [_row([
            {'role': 'user', 'content':
                'Explain the difference between A and B in detail please.'},
            {'role': 'assistant', 'content': 'A is good.',
             'thinking': 'Step 1: compare features... ' * 20},
        ])]
        assert len(_fil(rows)) == 1


class TestPipelineEdges:
    def test_no_user_dropped_by_default(self):
        rows = [_row([{'role': 'assistant', 'content': 'orphan reply'}])]
        assert _fil(rows) == []

    def test_no_user_kept_when_allowed(self):
        rows = [_row([{'role': 'assistant', 'content': 'orphan'}])]
        assert len(_fil(rows, allow_incomplete_role=True)) == 1

    def test_multi_user_skips_rules(self):
        # With ≥2 user turns, neither Rule 1 nor Rule 2 applies.
        rows = [_row([
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'short'},
            {'role': 'user', 'content': 'follow-up?'},
            {'role': 'assistant', 'content': 'tiny'},
        ])]
        assert len(_fil(rows)) == 1

    def test_non_list_messages(self):
        rows = [{'messages': 'not a list'}]
        assert _fil(rows) == []  # invalid → continue (skip)

    def test_missing_messages(self):
        rows = [{'id': 'x'}]
        # No user_msgs and allow_incomplete_role=False → skipped.
        assert _fil(rows) == []

    def test_empty_input(self):
        assert _fil([]) == []

    def test_custom_thresholds_applied(self):
        # Lower min_assistant_chars_2turn → keep what would normally be dropped.
        rows = [_row([
            {'role': 'user', 'content': 'tell me a real story please now'},
            {'role': 'assistant', 'content': 'A is good.'},
        ])]
        assert _fil(rows, min_assistant_chars_2turn=5) and len(rows) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
