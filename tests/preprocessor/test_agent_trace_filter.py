# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for AgentTraceFilter.

AgentTraceFilter is detection-only — it tags rows with ``is_agent=True/False``
and never drops or mutates messages. Detection delegates to
``ToolCallRegistry.detect_first`` so the test surface is:

  1. Tag is set on EVERY row (uniform schema).
  2. role='tool' or non-empty ``tool_calls`` field → True.
  3. Text-embedded tool calls (Cline / Hermes / ReAct) on assistant role → True.
  4. Plain assistant content with no tool markers → False.
  5. Look-alike XML that the registry rejects (e.g. plain ``<bash>...</bash>``
     without inner params) → False.
  6. Malformed message lists never raise.
"""
import pytest

from twinkle_agentic.preprocessor.agent_trace_filter import (
    AgentTraceFilter,
    _is_agent_row,
    _msg_text,
)


def _row(messages):
    return {'messages': messages}


# ── _msg_text helper ─────────────────────────────────────────────────────────

class TestMsgText:
    def test_string_content(self):
        assert _msg_text({'role': 'user', 'content': 'hello'}) == 'hello'

    def test_list_content_concat(self):
        msg = {'content': [
            {'type': 'text', 'text': 'a'},
            {'type': 'image', 'url': '...'},  # non-text part ignored
            {'type': 'text', 'text': 'b'},
        ]}
        assert _msg_text(msg) == 'a b'

    def test_missing_content(self):
        assert _msg_text({'role': 'user'}) == ''

    def test_none_content(self):
        assert _msg_text({'role': 'user', 'content': None}) == ''

    def test_non_str_non_list_content(self):
        assert _msg_text({'role': 'user', 'content': 123}) == ''


# ── _is_agent_row detection ──────────────────────────────────────────────────

class TestIsAgentRowStructural:
    def test_role_tool_triggers(self):
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'a', 'type': 'function', 'function': {'name': 'x', 'arguments': '{}'}}
            ]},
            {'role': 'tool', 'content': 'result', 'tool_call_id': 'a'},
        ]
        assert _is_agent_row(msgs) is True

    def test_tool_calls_field_triggers(self):
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'id': 'c1', 'type': 'function', 'function': {'name': 'f', 'arguments': '{}'}}
            ]},
        ]
        assert _is_agent_row(msgs) is True

    def test_empty_tool_calls_field_does_not_trigger(self):
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'plain reply', 'tool_calls': []},
        ]
        assert _is_agent_row(msgs) is False

    def test_non_list_tool_calls_field_does_not_trigger(self):
        msgs = [
            {'role': 'assistant', 'content': 'x', 'tool_calls': None},
        ]
        assert _is_agent_row(msgs) is False


class TestIsAgentRowTextEmbedded:
    def test_cline_style_triggers(self):
        msgs = [
            {'role': 'user', 'content': 'read the file'},
            {'role': 'assistant', 'content':
                '<read_file><path>/etc/hosts</path></read_file>'},
        ]
        assert _is_agent_row(msgs) is True

    def test_hermes_qwen_style_triggers(self):
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content':
                '<tool_call>\n{"name": "search", "arguments": {"q": "x"}}\n</tool_call>'},
        ]
        assert _is_agent_row(msgs) is True

    def test_react_action_style_triggers(self):
        # ReAct parser uses bracket syntax: ``Action: name[args]``.
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content':
                'Thought: I need to search.\nAction: search[query=x]'},
        ]
        assert _is_agent_row(msgs) is True

    def test_plain_assistant_text_does_not_trigger(self):
        msgs = [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'Hello! How can I help?'},
        ]
        assert _is_agent_row(msgs) is False

    def test_lookalike_xml_without_inner_params_does_not_trigger(self):
        # ``<bash>echo hi</bash>`` has no ``<key>val</key>`` child — Cline parser
        # rejects it via inner-param requirement. Hermes/ReAct also reject.
        msgs = [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': '<bash>echo hi</bash>'},
        ]
        assert _is_agent_row(msgs) is False

    def test_denied_outer_tag_does_not_trigger(self):
        # ``<think>``/``<code>`` are in the Cline DENY frozenset.
        msgs = [
            {'role': 'assistant', 'content':
                '<think><reason>because</reason></think>'},
        ]
        assert _is_agent_row(msgs) is False

    def test_user_text_with_tool_markers_does_not_trigger(self):
        # Markers must come from the assistant — user-side embedded XML is just data.
        msgs = [
            {'role': 'user', 'content':
                '<read_file><path>x</path></read_file>'},
            {'role': 'assistant', 'content': 'I will do that.'},
        ]
        assert _is_agent_row(msgs) is False

    def test_list_content_assistant_with_tool_call(self):
        msgs = [
            {'role': 'assistant', 'content': [
                {'type': 'text', 'text': '<tool_call>'},
                {'type': 'text', 'text': '{"name":"f","arguments":{}}</tool_call>'},
            ]},
        ]
        assert _is_agent_row(msgs) is True


class TestIsAgentRowEdgeCases:
    def test_non_list_messages(self):
        assert _is_agent_row(None) is False
        assert _is_agent_row('') is False
        assert _is_agent_row({}) is False

    def test_empty_messages(self):
        assert _is_agent_row([]) is False

    def test_non_dict_message_skipped(self):
        msgs = [
            'not a dict',
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'hello'},
        ]
        assert _is_agent_row(msgs) is False

    def test_short_circuits_on_first_match(self):
        # Even if later messages are clean, an earlier tool-call hit wins.
        msgs = [
            {'role': 'tool', 'content': 'r', 'tool_call_id': 'x'},
            {'role': 'assistant', 'content': 'plain'},
        ]
        assert _is_agent_row(msgs) is True


# ── AgentTraceFilter pipeline behavior ───────────────────────────────────────

class TestAgentTraceFilterPipeline:
    def test_tags_every_row(self):
        rows = [
            _row([{'role': 'assistant', 'content': 'plain'}]),
            _row([{'role': 'tool', 'content': 'r', 'tool_call_id': 'x'}]),
            _row([{'role': 'assistant', 'content':
                   '<read_file><path>x</path></read_file>'}]),
        ]
        out = AgentTraceFilter()(rows)
        assert len(out) == 3
        # Every row must have ``is_agent`` so map_row_to_col sees a uniform schema.
        assert all('is_agent' in r for r in out)
        assert [r['is_agent'] for r in out] == [False, True, True]

    def test_never_drops_rows(self):
        rows = [_row([{'role': 'user', 'content': 'x'}])] * 5
        out = AgentTraceFilter()(rows)
        assert len(out) == 5

    def test_preserves_other_fields(self):
        rows = [
            {'messages': [{'role': 'tool', 'content': 'r', 'tool_call_id': 'x'}],
             'id': 'row-1', 'extra': {'k': 'v'}},
        ]
        out = AgentTraceFilter()(rows)
        assert out[0]['id'] == 'row-1'
        assert out[0]['extra'] == {'k': 'v'}
        assert out[0]['is_agent'] is True

    def test_does_not_mutate_input(self):
        original = _row([{'role': 'assistant', 'content': 'plain'}])
        rows = [original]
        AgentTraceFilter()(rows)
        # Filter must return new dicts, not mutate originals.
        assert 'is_agent' not in original

    def test_missing_messages_key(self):
        rows = [{'id': 'lonely'}]  # no messages
        out = AgentTraceFilter()(rows)
        assert len(out) == 1
        assert out[0]['is_agent'] is False

    def test_messages_is_none(self):
        rows = [_row(None)]
        out = AgentTraceFilter()(rows)
        assert out[0]['is_agent'] is False

    def test_empty_input(self):
        assert AgentTraceFilter()([]) == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
