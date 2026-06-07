# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure-Python tests for tool-call parsers (no model download).

Covers Hermes/Qwen, ReAct, Cline parsing, cleaning, and — most importantly
— streaming correctness via the generic state machine in
:class:`twinkle.template.base.Template`.
"""
import json

import pytest

from twinkle.template.base import Template
from twinkle.template.tools import (
    ClineParser,
    HermesQwenParser,
    ReActParser,
    ToolCallRegistry,
    trailing_prefix_of,
)


class _StubTemplate:
    """Minimal Template-shaped object exposing only stream-related members.

    Avoids loading a real tokenizer/processor (which would need network).
    """

    parse_tool_call_stream = Template.parse_tool_call_stream
    _stream_marker_blocks = Template._stream_marker_blocks
    _format_tc_delta = staticmethod(Template._format_tc_delta)

    def __init__(self, model_id: str):
        self.model_id = model_id


def _stream(model_id, chunks_with_finished):
    t = _StubTemplate(model_id)
    state = {}
    events = []
    for chunk, fin in chunks_with_finished:
        events.extend(t.parse_tool_call_stream(state, chunk, finished=fin))
    return events, state


# ---------------------------------------------------------------------------
# HermesQwenParser
# ---------------------------------------------------------------------------


class TestHermesQwenParser:

    def setup_method(self):
        self.p = HermesQwenParser()

    def test_detect(self):
        assert self.p.detect('hi <tool_call>{"name":"f","arguments":{}}</tool_call>')
        assert not self.p.detect('plain text')
        assert not self.p.detect('')

    def test_matches_model(self):
        assert self.p.matches_model('qwen2.5-7b')
        assert self.p.matches_model('qwen3-32b')
        assert not self.p.matches_model('llama-3.1-8b')

    def test_parse_json_variant(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}},
        }]

    def test_parse_function_xml_variant(self):
        text = ('<tool_call><function=add>'
                '<parameter=a>1</parameter><parameter=b>2</parameter>'
                '</function></tool_call>')
        out = self.p.parse(text)
        assert len(out) == 1
        assert out[0]['function']['name'] == 'add'
        # JSON-decoding of param values: numbers come back as int.
        assert out[0]['function']['arguments'] == {'a': 1, 'b': 2}

    def test_parse_multiple_blocks(self):
        text = ('<tool_call>{"name":"f1","arguments":{}}</tool_call>'
                'between '
                '<tool_call>{"name":"f2","arguments":{"k":"v"}}</tool_call>')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['f1', 'f2']
        assert out[1]['function']['arguments'] == {'k': 'v'}

    def test_parse_unclosed_block_at_eof(self):
        # ``\Z`` fallback in _BLOCK_RE handles truncated trailing block.
        text = '<tool_call>{"name": "f", "arguments": {}}'
        out = self.p.parse(text)
        assert out and out[0]['function']['name'] == 'f'

    def test_parse_empty_returns_empty_list(self):
        assert self.p.parse('') == []
        assert self.p.parse('plain text without markers') == []

    def test_clean_strips_blocks(self):
        text = 'hello <tool_call>{"name":"f","arguments":{}}</tool_call> world'
        assert self.p.clean(text) == 'hello  world'

    def test_clean_unclosed_at_eof(self):
        text = 'hello <tool_call>{"name":"f"'
        assert self.p.clean(text) == 'hello'

    def test_clean_empty(self):
        assert self.p.clean('') == ''

    def test_markers_declared(self):
        assert self.p.open_marker == '<tool_call>'
        assert self.p.close_marker == '</tool_call>'


class TestHermesQwenStreaming:
    """Generic open/close marker buffer state machine."""

    def test_plain_text_passthrough(self):
        events, _ = _stream('qwen2.5-7b', [('Hello world!', True)])
        assert events == [{'content': 'Hello world!'}]

    def test_holds_back_partial_open_marker(self):
        events, state = _stream('qwen2.5-7b', [
            ('Hello! ', False),
            ('<tool_', False),
        ])
        # Only the leading non-marker content emitted; '<tool_' deferred.
        assert events == [{'content': 'Hello! '}]
        assert state['pending'] == '<tool_'

    def test_emits_tool_call_after_close(self):
        events, _ = _stream('qwen2.5-7b', [
            ('Hello! ', False),
            ('<tool_', False),
            ('call>{"name":"f","arguments":{}}</tool_call>', False),
            ('done.', False),
            ('', True),
        ])
        types = [next(iter(e)) for e in events]
        assert types == ['content', 'tool_calls', 'content']
        tc = events[1]['tool_calls'][0]
        assert tc['function']['name'] == 'f'
        # OpenAI streaming spec: arguments serialised as JSON string.
        assert tc['function']['arguments'] == '{}'
        assert tc['index'] == 0
        assert tc['id'].startswith('call_')
        assert tc['type'] == 'function'

    def test_stream_chunked_inside_block(self):
        # Split the block at every char to torture-test the partial-marker
        # hold-back logic.
        full = '<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>'
        chunks = [(full[i:i + 1], False) for i in range(len(full))]
        chunks.append(('', True))
        events, state = _stream('qwen2.5-7b', chunks)
        tcs = [e['tool_calls'][0] for e in events if 'tool_calls' in e]
        assert len(tcs) == 1
        assert tcs[0]['function']['name'] == 'f'
        assert json.loads(tcs[0]['function']['arguments']) == {'x': 1}
        assert state['pending'] == ''
        # No content events should leak the markup.
        for e in events:
            if 'content' in e:
                assert '<tool_call>' not in e['content']
                assert '</tool_call>' not in e['content']

    def test_multiple_blocks_increasing_indices(self):
        events, _ = _stream('qwen2.5-7b', [
            ('<tool_call>{"name":"a","arguments":{}}</tool_call>'
             '<tool_call>{"name":"b","arguments":{}}</tool_call>', True),
        ])
        tcs = [e['tool_calls'][0] for e in events if 'tool_calls' in e]
        assert [t['function']['name'] for t in tcs] == ['a', 'b']
        assert [t['index'] for t in tcs] == [0, 1]

    def test_unclosed_block_flushed_on_finish(self):
        events, state = _stream('qwen2.5-7b', [
            ('<tool_call>{"name":"f","arguments":{}}', True),
        ])
        assert state['pending'] == ''
        tcs = [e['tool_calls'][0] for e in events if 'tool_calls' in e]
        assert tcs and tcs[0]['function']['name'] == 'f'

    def test_arguments_serialised_as_json_string(self):
        events, _ = _stream('qwen2.5-7b', [
            ('<tool_call>{"name":"f","arguments":{"k":"v","n":3}}</tool_call>', True),
        ])
        tc = next(e['tool_calls'][0] for e in events if 'tool_calls' in e)
        assert isinstance(tc['function']['arguments'], str)
        assert json.loads(tc['function']['arguments']) == {'k': 'v', 'n': 3}

    def test_content_events_lossless_for_non_block_text(self):
        # All non-tool-call text must pass through verbatim, regardless of
        # chunk boundaries.
        original_content_outside = 'aXY'
        full = ('a'
                '<tool_call>{"name":"f","arguments":{}}</tool_call>'
                'XY')
        chunks = [(full[i:i + 3], False) for i in range(0, len(full), 3)]
        chunks.append(('', True))
        events, _ = _stream('qwen2.5-7b', chunks)
        rebuilt = ''.join(e['content'] for e in events if 'content' in e)
        assert rebuilt == original_content_outside

    def test_no_emission_until_chunk_arrives(self):
        # Streaming with empty chunk and not-finished should be a no-op.
        events, _ = _stream('qwen2.5-7b', [('', False)])
        assert events == []


# ---------------------------------------------------------------------------
# ReActParser
# ---------------------------------------------------------------------------


class TestReActParser:

    def setup_method(self):
        self.p = ReActParser()

    def test_detect_action_line(self):
        assert self.p.detect('Thought: I need search.\nAction: search[python]')
        assert not self.p.detect('plain text without action keyword')
        assert not self.p.detect('')

    def test_no_block_marker(self):
        # Prose format — streaming has no marker to lock onto.
        assert self.p.open_marker is None
        assert self.p.close_marker is None

    def test_does_not_match_qwen_model(self):
        assert not self.p.matches_model('qwen2.5')
        assert not self.p.matches_model('llama-3')

    def test_parse_single_action(self):
        text = 'Thought: search the web.\nAction: search[hello world]'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {'name': 'search', 'arguments': {'input': 'hello world'}},
        }]

    def test_parse_multiple_actions(self):
        text = ('Thought: a\nAction: tool_a[x]\n'
                'Observation: ok\n'
                'Thought: b\nAction: tool_b[y z]')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['tool_a', 'tool_b']
        assert out[1]['function']['arguments'] == {'input': 'y z'}

    def test_clean_removes_action_lines(self):
        text = 'Thought: hi\nAction: search[x]\nDone'
        cleaned = self.p.clean(text)
        assert 'Action: search' not in cleaned
        assert 'Thought: hi' in cleaned
        assert 'Done' in cleaned

    def test_parse_empty(self):
        assert self.p.parse('') == []


class TestReActStreaming:
    """ReAct has no marker → falls back to plain content passthrough.

    Detection is a final-pass concern; streaming preserves content faithfully.
    """

    def test_passthrough_when_no_marker_parser(self):
        # 'react-agent' doesn't match HermesQwen ('qwen' substring) → no parser
        # cached → passthrough mode.
        events, state = _stream('react-agent', [
            ('Thought: hi\n', False),
            ('Action: foo[bar]\n', False),
            ('done', False),
            ('', True),
        ])
        rebuilt = ''.join(e['content'] for e in events if 'content' in e)
        assert rebuilt == 'Thought: hi\nAction: foo[bar]\ndone'
        assert state.get('parser') is None

    def test_no_tool_calls_event_emitted(self):
        events, _ = _stream('react-agent', [
            ('Action: foo[bar]', True),
        ])
        assert all('tool_calls' not in e for e in events)


# ---------------------------------------------------------------------------
# ClineParser
# ---------------------------------------------------------------------------


class TestClineParser:

    def setup_method(self):
        self.p = ClineParser()

    def test_detect_simple_tool(self):
        assert self.p.detect('<read_file><path>foo.py</path></read_file>')

    def test_detect_ignores_html_like_tags(self):
        # ``think`` / ``code`` are denied — even with inner content they aren't
        # treated as tool calls.
        assert not self.p.detect('<think><inner>x</inner></think>')
        assert not self.p.detect('<code><line>x</line></code>')

    def test_detect_requires_inner_param(self):
        # No inner ``<key>VAL</key>`` → not a Cline call.
        assert not self.p.detect('<read_file>just text</read_file>')

    def test_detect_ignores_hermes_block(self):
        # Hermes already owns ``<tool_call>`` — Cline must skip it.
        assert not self.p.detect('<tool_call>{"name":"f","arguments":{}}</tool_call>')

    def test_no_marker_for_streaming(self):
        # Outer tag varies per call — streaming uses passthrough, not the
        # marker state machine.
        assert self.p.open_marker is None
        assert self.p.close_marker is None

    def test_does_not_match_any_model_by_default(self):
        # Cline is an app-level prompt protocol, not a model-family format.
        assert not self.p.matches_model('qwen2.5')
        assert not self.p.matches_model('claude-3')

    def test_parse_single_arg(self):
        text = '<read_file><path>src/foo.py</path></read_file>'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {'name': 'read_file', 'arguments': {'path': 'src/foo.py'}},
        }]

    def test_parse_multi_arg_with_whitespace(self):
        text = ('<execute_command>\n'
                '  <command>ls -la</command>\n'
                '  <requires_approval>false</requires_approval>\n'
                '</execute_command>')
        out = self.p.parse(text)
        fn = out[0]['function']
        assert fn['name'] == 'execute_command'
        assert fn['arguments'] == {'command': 'ls -la', 'requires_approval': 'false'}

    def test_parse_multiple_blocks(self):
        text = ('<read_file><path>a</path></read_file>'
                ' between '
                '<list_files><path>b</path><recursive>true</recursive></list_files>')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['read_file', 'list_files']
        assert out[1]['function']['arguments'] == {'path': 'b', 'recursive': 'true'}

    def test_parse_skips_hermes_block(self):
        text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
        assert self.p.parse(text) == []

    def test_clean_strips_tool_blocks(self):
        text = 'before <read_file><path>x</path></read_file> after'
        assert self.p.clean(text) == 'before  after'

    def test_clean_preserves_non_tool_xml(self):
        text = '<think>reasoning</think> <read_file><path>x</path></read_file> tail'
        cleaned = self.p.clean(text)
        assert '<think>reasoning</think>' in cleaned
        assert '<read_file>' not in cleaned
        assert 'tail' in cleaned

    def test_clean_empty(self):
        assert self.p.clean('') == ''


class TestClineStreaming:
    """Cline streams as plain content (no fixed open marker)."""

    def test_content_passthrough_lossless_across_chunk_boundaries(self):
        full = ('intro <read_file><path>src/foo.py</path></read_file> outro'
                ' next <list_files><path>x</path></list_files>')
        # Chunk every 4 chars — boundaries fall inside tags, args, etc.
        chunks = [(full[i:i + 4], False) for i in range(0, len(full), 4)]
        chunks.append(('', True))
        events, _ = _stream('cline-bot', chunks)
        rebuilt = ''.join(e['content'] for e in events if 'content' in e)
        assert rebuilt == full
        # No tool_calls events because no parser was selected by model_id.
        assert all('tool_calls' not in e for e in events)


# ---------------------------------------------------------------------------
# Registry round-robin & helpers
# ---------------------------------------------------------------------------


class TestRegistryRoundRobin:

    def test_first_match_wins_no_nested_reparse(self):
        # Hermes block must take ownership; ReAct/Cline shouldn't see it.
        text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'hermes_qwen'

    def test_cline_wins_for_xml_tools(self):
        text = '<read_file><path>x</path></read_file>'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'cline'

    def test_react_wins_for_action_keyword(self):
        text = 'Thought: hi\nAction: search[x]'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'react'

    def test_no_parser_for_plain_text(self):
        assert ToolCallRegistry.detect_first('just some plain text') is None
        assert ToolCallRegistry.detect_first('') is None

    def test_select_for_qwen_picks_hermes(self):
        parser = ToolCallRegistry.select_for_model('qwen2.5-7b')
        assert parser is not None and parser.name == 'hermes_qwen'

    def test_select_for_unknown_returns_none(self):
        assert ToolCallRegistry.select_for_model('llama-3.1-8b') is None
        assert ToolCallRegistry.select_for_model(None) is None


class TestTrailingPrefixOf:
    """Holdback length helper used by the marker state machine."""

    def test_no_prefix(self):
        assert trailing_prefix_of('hello world', '<tool_call>') == 0

    def test_partial_prefix_4_chars(self):
        # buf ends with '<too' — prefix of '<tool_call>' length 4.
        assert trailing_prefix_of('hello <too', '<tool_call>') == 4

    def test_partial_prefix_1_char(self):
        assert trailing_prefix_of('hello <', '<tool_call>') == 1

    def test_full_marker_returns_zero(self):
        # Full marker at end is NOT a strict prefix (search range is 1..len-1),
        # so the helper returns 0 — block code path will see the marker via
        # ``find()`` rather than holdback.
        assert trailing_prefix_of('text<tool_call>', '<tool_call>') == 0

    def test_empty_buf(self):
        assert trailing_prefix_of('', '<tool_call>') == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
