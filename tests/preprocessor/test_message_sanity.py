# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for MessageSanityFilter preprocessor."""
import pytest

from twinkle_agentic.preprocessor.message_sanity import (MessageSanityFilter, _trim_to_last_assistant,
                                                         _validate_content_integrity, _validate_role_order,
                                                         _validate_tool_call_matching)

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_rows(messages_list):
    """Wrap messages lists into row-format for the filter."""
    return [{'messages': m} for m in messages_list]


def _run_filter(messages_list, **kwargs):
    """Run MessageSanityFilter on a list of message sequences, return surviving messages."""
    f = MessageSanityFilter(**kwargs)
    rows = _make_rows(messages_list)
    result = f.message_sanity_filter(rows)
    return [r['messages'] for r in result]


# ── Role order tests ──────────────────────────────────────────────────────────


class TestRoleOrder:

    def test_valid_simple(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_role_order(msgs) is True

    def test_valid_with_system(self):
        msgs = [
            {
                'role': 'system',
                'content': 'You are helpful.'
            },
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_role_order(msgs) is True

    def test_system_not_first(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'system',
                'content': 'late system'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_role_order(msgs) is False

    def test_tool_without_tool_calls(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'let me check'
            },
            {
                'role': 'tool',
                'content': 'result',
                'tool_call_id': 'x'
            },
        ]
        assert _validate_role_order(msgs) is False

    def test_tool_after_assistant_with_tool_calls(self):
        msgs = [
            {
                'role': 'user',
                'content': 'search'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [{
                    'id': 'c1',
                    'type': 'function',
                    'function': {
                        'name': 'search',
                        'arguments': '{}'
                    }
                }]
            },
            {
                'role': 'tool',
                'content': 'found it',
                'tool_call_id': 'c1'
            },
        ]
        assert _validate_role_order(msgs) is True

    def test_tool_after_user(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'tool',
                'content': 'bad',
                'tool_call_id': 'x'
            },
        ]
        assert _validate_role_order(msgs) is False

    def test_invalid_role(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'bot',
                'content': 'hello'
            },
        ]
        assert _validate_role_order(msgs) is False

    def test_empty(self):
        assert _validate_role_order([]) is False

    def test_consecutive_tools(self):
        msgs = [
            {
                'role': 'user',
                'content': 'do things'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'a',
                            'arguments': '{}'
                        }
                    },
                    {
                        'id': 'c2',
                        'type': 'function',
                        'function': {
                            'name': 'b',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'res1',
                'tool_call_id': 'c1'
            },
            {
                'role': 'tool',
                'content': 'res2',
                'tool_call_id': 'c2'
            },
        ]
        assert _validate_role_order(msgs) is True


# ── Tool call matching tests ──────────────────────────────────────────────────


class TestToolCallMatching:

    def test_valid_matching(self):
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'fn',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'ok',
                'tool_call_id': 'c1'
            },
            {
                'role': 'assistant',
                'content': 'done'
            },
        ]
        assert _validate_tool_call_matching(msgs) is True

    def test_orphan_tool_calls(self):
        """Assistant has tool_calls but no tool response follows."""
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'fn',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'user',
                'content': 'what happened?'
            },
        ]
        assert _validate_tool_call_matching(msgs) is False

    def test_phantom_tool_response(self):
        """Tool response references an ID not in the assistant's tool_calls."""
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'fn',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'ok',
                'tool_call_id': 'WRONG_ID'
            },
        ]
        assert _validate_tool_call_matching(msgs) is False

    def test_partial_response_ok(self):
        """Only some tool_calls get responses — currently allowed."""
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'a',
                            'arguments': '{}'
                        }
                    },
                    {
                        'id': 'c2',
                        'type': 'function',
                        'function': {
                            'name': 'b',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'res1',
                'tool_call_id': 'c1'
            },
        ]
        assert _validate_tool_call_matching(msgs) is True

    def test_no_tool_calls_passes(self):
        """Conversations without tool_calls pass trivially."""
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_tool_call_matching(msgs) is True


# ── Content integrity tests ───────────────────────────────────────────────────


class TestContentIntegrity:

    def test_valid_basic(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello there'
            },
        ]
        assert _validate_content_integrity(msgs) is True

    def test_empty_assistant(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': ''
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_assistant_with_tool_calls_no_content_ok(self):
        msgs = [
            {
                'role': 'user',
                'content': 'search'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [{
                    'id': 'c1',
                    'type': 'function',
                    'function': {
                        'name': 'search_web',
                        'arguments': '{"q":"test"}'
                    }
                }]
            },
        ]
        assert _validate_content_integrity(msgs) is True

    def test_empty_system(self):
        msgs = [
            {
                'role': 'system',
                'content': ''
            },
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_too_long_message(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'x' * 60000
            },
        ]
        assert _validate_content_integrity(msgs, max_msg_chars=50000) is False

    def test_invalid_tool_call_structure(self):
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'function': 'not_a_dict'
                    },  # function must be dict
                ]
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_invalid_function_name(self):
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': '123bad',
                            'arguments': '{}'
                        }
                    },
                ]
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_invalid_arguments_json(self):
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'fn',
                            'arguments': '{invalid json'
                        }
                    },
                ]
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_dict_arguments_ok(self):
        msgs = [
            {
                'role': 'user',
                'content': 'go'
            },
            {
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'fn',
                            'arguments': {
                                'key': 'val'
                            }
                        }
                    },
                ]
            },
        ]
        assert _validate_content_integrity(msgs) is True

    def test_duplicate_user_messages(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _validate_content_integrity(msgs) is False

    def test_duplicate_tool_messages_allowed(self):
        """Two consecutive tool messages with same content should NOT be rejected."""
        msgs = [
            {
                'role': 'user',
                'content': 'search both'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'search',
                            'arguments': '{"q":"x"}'
                        }
                    },
                    {
                        'id': 'c2',
                        'type': 'function',
                        'function': {
                            'name': 'search',
                            'arguments': '{"q":"x"}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'same result',
                'tool_call_id': 'c1'
            },
            {
                'role': 'tool',
                'content': 'same result',
                'tool_call_id': 'c2'
            },
            {
                'role': 'assistant',
                'content': 'both returned same'
            },
        ]
        assert _validate_content_integrity(msgs) is True

    def test_min_turns(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        # min_turns=2 → user(1)+assistant(1)=2 >= 2 → pass
        assert _validate_content_integrity(msgs, min_turns=2) is True
        # min_turns=3 → total=2 < 3 → fail
        assert _validate_content_integrity(msgs, min_turns=3) is False


# ── Trim tests ────────────────────────────────────────────────────────────────


class TestTrimToLastAssistant:

    def test_already_ends_with_assistant(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        assert _trim_to_last_assistant(msgs) == msgs

    def test_trim_trailing_user(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
            {
                'role': 'user',
                'content': 'bye'
            },
        ]
        assert _trim_to_last_assistant(msgs) == msgs[:2]

    def test_no_assistant(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'user',
                'content': 'hello?'
            },
        ]
        assert _trim_to_last_assistant(msgs) == []


# ── Sensitive word tests ──────────────────────────────────────────────────────


class TestSensitiveWords:

    def test_english_word_boundary(self):
        msgs_clean = [
            {
                'role': 'user',
                'content': 'hello world'
            },
            {
                'role': 'assistant',
                'content': 'hi there'
            },
        ]
        msgs_bad = [
            {
                'role': 'user',
                'content': 'hello world'
            },
            {
                'role': 'assistant',
                'content': 'what the fuck'
            },
        ]
        result = _run_filter(
            [msgs_clean, msgs_bad],
            extra_sensitive_words=['fuck'],
        )
        assert len(result) == 1
        assert result[0] == msgs_clean

    def test_chinese_sensitive(self):
        msgs_bad = [
            {
                'role': 'user',
                'content': '你好'
            },
            {
                'role': 'assistant',
                'content': '操你妈'
            },
        ]
        result = _run_filter(
            [msgs_bad],
            extra_sensitive_words=['操你妈'],
        )
        assert len(result) == 0

    def test_no_sensitive_config_passes_all(self):
        msgs = [
            {
                'role': 'user',
                'content': 'fuck'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
        ]
        # No sensitive words configured → everything passes
        result = _run_filter([msgs])
        assert len(result) == 1


# ── End-to-end filter tests ───────────────────────────────────────────────────


class TestEndToEnd:

    def test_full_valid_agentic_trajectory(self):
        msgs = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': 'What is the weather?'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [
                    {
                        'id': 'call_1',
                        'type': 'function',
                        'function': {
                            'name': 'get_weather',
                            'arguments': '{"city":"Beijing"}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': '{"temp": 22, "condition": "sunny"}',
                'tool_call_id': 'call_1'
            },
            {
                'role': 'assistant',
                'content': 'It is 22°C and sunny in Beijing.'
            },
        ]
        result = _run_filter([msgs])
        assert len(result) == 1

    def test_trim_and_validate(self):
        """Trailing user message gets trimmed, result still valid."""
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'assistant',
                'content': 'hello'
            },
            {
                'role': 'user',
                'content': 'thanks'
            },
        ]
        result = _run_filter([msgs])
        assert len(result) == 1
        assert result[0][-1]['role'] == 'assistant'

    def test_no_assistant_discarded(self):
        msgs = [
            {
                'role': 'user',
                'content': 'hi'
            },
            {
                'role': 'user',
                'content': 'hello?'
            },
        ]
        result = _run_filter([msgs])
        assert len(result) == 0

    def test_multiple_tool_rounds(self):
        msgs = [
            {
                'role': 'user',
                'content': 'plan a trip'
            },
            {
                'role':
                'assistant',
                'content':
                '',
                'tool_calls': [
                    {
                        'id': 'c1',
                        'type': 'function',
                        'function': {
                            'name': 'search_flights',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'flight options...',
                'tool_call_id': 'c1'
            },
            {
                'role': 'assistant',
                'content': 'Found flights. Let me check hotels.',
                'tool_calls': [
                    {
                        'id': 'c2',
                        'type': 'function',
                        'function': {
                            'name': 'search_hotels',
                            'arguments': '{}'
                        }
                    },
                ]
            },
            {
                'role': 'tool',
                'content': 'hotel options...',
                'tool_call_id': 'c2'
            },
            {
                'role': 'assistant',
                'content': 'Here is your complete trip plan.'
            },
        ]
        result = _run_filter([msgs])
        assert len(result) == 1
