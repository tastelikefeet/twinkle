# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for DeadLoopFilter.

Three orthogonal "stuck" signals:
  1. Hesitation density       — markers per 1000 chars > threshold
  2. Correction cascade       — ≥N markers within a sliding window
  3. High n-gram repetition   — (1 - unique/total) > threshold

A row is dropped if ANY signal trips on any assistant turn.
Rows with ``is_agent=True`` are always kept (agent rollouts have legitimate
self-correction phrasing).

When the message contains ``<think>...</think>``, the think part and the
response part are scored independently with separate (looser) think-thresholds.
"""
import pytest

from twinkle_agentic.preprocessor.dead_loop_filter import (DeadLoopFilter, _has_correction_cascade_with_threshold,
                                                           _hesitation_density, _high_repetition_with_threshold,
                                                           _is_stuck)


def _row(messages, **extra):
    return {'messages': messages, **extra}


def _fil(rows, **kw):
    return DeadLoopFilter(**kw)(rows)


# ── _hesitation_density ─────────────────────────────────────────────────────


class TestHesitationDensity:

    def test_no_markers(self):
        text = 'This is a perfectly normal explanation of gradient descent.'
        assert _hesitation_density(text) == 0.0

    def test_english_marker_counted(self):
        # "wait, wait" matches `wait[,\s]+(wait|...)` — one marker.
        text = 'wait, wait this is wrong'
        d = _hesitation_density(text)
        assert d > 0

    def test_density_per_1000(self):
        # ~5 markers in 100 chars → density ~50/1000
        text = ('hmm hmm hmm hmm hmm ' * 1).strip()  # 5 hmm tokens
        # Each "hmm" matches `hmm+[,\s]*\.{0,3}` → 5 matches
        density = _hesitation_density(text)
        assert density > 100  # very dense

    def test_chinese_marker(self):
        text = '等等，让我重新想想这个问题。'
        assert _hesitation_density(text) > 0

    def test_empty_text(self):
        assert _hesitation_density('') == 0.0

    def test_japanese_marker(self):
        text = 'ちょっと待って、もう一度考え直してみます。'
        assert _hesitation_density(text) > 0

    def test_korean_marker(self):
        text = '잠깐, 다시 생각해봐야겠어요.'
        assert _hesitation_density(text) > 0


# ── _has_correction_cascade_with_threshold ──────────────────────────────────


class TestCorrectionCascade:

    def test_below_threshold(self):
        # Only 2 cascade markers; threshold=5 → no cascade.
        text = 'wait, actually let me think.'
        assert _has_correction_cascade_with_threshold(text, threshold=5) is False

    def test_at_threshold_in_window(self):
        # 5 cascade tokens packed into <800 chars → cascade detected.
        text = 'wait wait wait wait wait'
        assert _has_correction_cascade_with_threshold(text, threshold=5, window=800) is True

    def test_threshold_outside_window(self):
        # 5 markers but spread across >800 chars → no cascade.
        spacer = ' ' * 200  # each spacer is 200 chars
        text = f'wait{spacer}wait{spacer}wait{spacer}wait{spacer}wait'  # 5*200 = 1000 chars
        assert _has_correction_cascade_with_threshold(text, threshold=5, window=800) is False

    def test_chinese_cascade(self):
        text = '等等，不对，重新想想，错了，让我再算一遍。'
        assert _has_correction_cascade_with_threshold(text, threshold=4) is True

    def test_zero_threshold_unreachable(self):
        # threshold=0 means need 0 matches in any window — len(matches) < 0 is
        # never true so this returns True even on empty.  Test the sane case.
        assert _has_correction_cascade_with_threshold('clean text', threshold=1) is False


# ── _high_repetition_with_threshold ─────────────────────────────────────────


class TestRepetition:

    def test_below_min_words(self):
        # Fewer than ngram_min_words words → False (insufficient sample).
        text = 'this is a short text'
        assert _high_repetition_with_threshold(text, threshold=0.0, ngram_min_words=30) is False

    def test_no_repetition(self):
        # 30 distinct words → unique_ratio ~ 1.0 → repetition ~ 0.
        text = ' '.join(f'word{i}' for i in range(40))
        assert _high_repetition_with_threshold(text, threshold=0.45, ngram_min_words=30) is False

    def test_high_repetition_triggers(self):
        # Same 8-gram repeated → unique_ratio low → repetition high.
        phrase = 'the quick brown fox jumps over the lazy'
        text = ' '.join([phrase] * 10)
        assert _high_repetition_with_threshold(text, threshold=0.45, ngram_size=8, ngram_min_words=30) is True

    def test_threshold_boundary(self):
        # Same text under different thresholds.
        phrase = 'a b c d e f g h '
        text = phrase * 6  # 48 words, only 8 unique
        # very low threshold → trips
        assert _high_repetition_with_threshold(text, threshold=0.1) is True
        # very high threshold → does not trip even with high duplication
        assert _high_repetition_with_threshold(text, threshold=0.99) is False


# ── _is_stuck ───────────────────────────────────────────────────────────────


class TestIsStuck:

    def test_clean_text_not_stuck(self):
        # Use diverse prose so n-gram repetition stays below threshold.
        text = ('Gradient descent is an iterative optimization algorithm used '
                'for finding the local minimum of a differentiable function. '
                'It updates parameters in the direction opposite to the '
                'gradient of the objective at the current point. Variants '
                'such as momentum and Adam improve convergence speed.')
        assert _is_stuck(text) is False

    def test_high_density_stuck(self):
        # Pack many hesitation tokens to exceed 7/1000 density.
        text = 'wait, wait this is wrong. hmm... actually no. uh, wait wait wait.'
        assert _is_stuck(text) is True

    def test_cascade_stuck(self):
        # 5 cascade tokens in tight window
        text = 'wait actually wait actually wait!'
        assert _is_stuck(
            text, hesitation_density_threshold=999.0, cascade_threshold=5, repetition_threshold=0.99) is True

    def test_repetition_stuck(self):
        phrase = 'the quick brown fox jumps over the lazy'
        text = ' '.join([phrase] * 10)
        assert _is_stuck(
            text, hesitation_density_threshold=999.0, cascade_threshold=999, repetition_threshold=0.45) is True

    def test_think_block_separate_thresholds(self):
        # Hesitation that would trip in response section is allowed inside
        # <think>...</think> because think-thresholds are looser (15.0 vs 7.0).
        # Build a think with moderate density (~10/1000) — below 15 think
        # threshold, but would exceed 7 in normal text.
        think_part = 'wait, actually let me reconsider this. ' * 3 + 'a' * 1500
        text = f'<think>{think_part}</think>The answer is 42.'
        assert _is_stuck(text) is False  # think-density well below 15

    def test_response_part_after_think_stuck(self):
        # Clean think but stuck response → still stuck.
        text = ('<think>Calculating step by step.</think>'
                'wait, wait this is wrong. hmm... actually no. uh, wait wait wait.')
        assert _is_stuck(text) is True


# ── DeadLoopFilter pipeline ─────────────────────────────────────────────────


class TestDeadLoopFilterPipeline:

    def test_drops_stuck_row(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': 'wait, wait this is wrong. hmm... actually no. '
                    'uh, wait wait wait.'
                },
            ])
        ]
        assert _fil(rows) == []

    def test_keeps_clean_row(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': 'A clear, well-formed answer goes here.'
                },
            ])
        ]
        assert len(_fil(rows)) == 1

    def test_agent_row_always_kept(self):
        # is_agent=True bypasses all stuck checks.
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': 'wait wait wait wait wait wait wait!!!'
                },
            ],
                 is_agent=True)
        ]
        assert len(_fil(rows)) == 1

    def test_no_assistant_kept(self):
        rows = [_row([{'role': 'user', 'content': 'hi'}])]
        assert len(_fil(rows)) == 1

    def test_any_assistant_stuck_drops_row(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q1'
                },
                {
                    'role': 'assistant',
                    'content': 'clean reply'
                },
                {
                    'role': 'user',
                    'content': 'q2'
                },
                {
                    'role': 'assistant',
                    'content': 'wait, wait this is wrong. hmm... actually no. '
                    'uh, wait wait wait.'
                },
            ])
        ]
        assert _fil(rows) == []

    def test_empty_input(self):
        assert _fil([]) == []

    def test_custom_thresholds(self):
        # 1 hesitation marker in a long message — density well below the
        # default 7/1000.  Tightening the threshold should drop it.
        long_msg = ('Hmm, let me think about this carefully. Gradient descent '
                    'requires a learning rate, the loss function, and an '
                    'initial parameter point. The algorithm iteratively '
                    'updates the parameters towards the negative gradient. '
                    'Momentum-based variants accumulate past gradients to '
                    'smooth the trajectory and accelerate convergence on '
                    'ill-conditioned problems. Adam additionally adapts the '
                    'per-parameter learning rate using running second-moment '
                    'estimates, which often makes it the default choice for '
                    'practitioners across many deep-learning tasks.')
        rows = [_row([
            {
                'role': 'user',
                'content': 'q'
            },
            {
                'role': 'assistant',
                'content': long_msg
            },
        ])]
        # Default 7/1000 — single marker in long text → kept
        assert len(_fil(rows)) == 1
        # Aggressive threshold drops it
        assert _fil(rows, hesitation_density_threshold=0.5) == []

    def test_chinese_stuck(self):
        rows = [
            _row([
                {
                    'role': 'user',
                    'content': 'q'
                },
                {
                    'role': 'assistant',
                    'content': '等等，不对，让我重新想想。错了，让我再来一次。'
                    '我又搞错了。等等，等等。'
                },
            ])
        ]
        assert _fil(rows) == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
