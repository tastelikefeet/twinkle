# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import logging
import re
from typing import Any, Dict, List

from twinkle import remote_class
from twinkle.template import Template

logger = logging.getLogger(__name__)


@remote_class()
class QwenTemplate(Template):

    _BLOCK_RE = re.compile(r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
    _FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
    _PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
    _STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')

    _TOOL_CALL_OPEN = '<tool_call>'
    _TOOL_CALL_CLOSE = '</tool_call>'

    def parse(self, decoded: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block_m in self._BLOCK_RE.finditer(decoded or ''):
            block = block_m.group(1)
            func_m = self._FUNCTION_RE.search(block)
            if func_m:
                args: Dict[str, Any] = {}
                for pm in self._PARAMETER_RE.finditer(func_m.group(2)):
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        args[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[key] = val
                calls.append({
                    'type': 'function',
                    'function': {
                        'name': func_m.group(1).strip(),
                        'arguments': args,
                    },
                })
                continue
            # JSON fallback: ``{"name": ..., "arguments": ...}`` inside the block.
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            name = data.get('name') or data.get('tool_name', '')
            if not name:
                continue
            args = data.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': args if isinstance(args, dict) else {},
                },
            })
        return calls

    def clean(self, decoded: str) -> str:
        return self._STRIP_RE.sub('', decoded or '').rstrip()

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        """Parse tool calls from the assistant's decoded output.

        Dispatches by model family on ``self.model_id``; the actual
        wire-format logic lives in :mod:`.tool_call_parser`.
        """
        mid = (self.model_id or '').lower()
        if 'qwen' in mid:
            return self.parse(decoded)
        # TODO: Other models (Llama3, OpenAI JSON, …) — add a parser in
        # ``tool_call_parser.py`` and extend this dispatch.
        return []

    def clean_tool_call(self, decoded: str) -> str:
        """Strip family-specific tool-call markup from assistant text."""
        mid = (self.model_id or '').lower()
        if 'qwen' in mid:
            return self.clean(decoded)
        # TODO: Other models
        return (decoded or '').rstrip()

    @staticmethod
    def _trailing_prefix_of(buf: str, marker: str) -> int:
        """Length of trailing chars of ``buf`` that form a strict prefix of ``marker``.

        Used to hold back the last ``k`` chars when they could be the start of an
        incoming tool-call open tag — prevents splitting ``<tool_call>`` mid-stream.
        """
        upper = min(len(marker) - 1, len(buf))
        for k in range(upper, 0, -1):
            if buf.endswith(marker[:k]):
                return k
        return 0

    def _format_tc_delta(self, state: Dict[str, Any], tc: Dict[str, Any]) -> Dict[str, Any]:
        fn = dict(tc.get('function') or {})
        args = fn.get('arguments')
        if isinstance(args, dict):
            fn['arguments'] = json.dumps(args, ensure_ascii=False)
        delta = {
            'index': state['tc_count'],
            'id': tc.get('id') or f'call_{state["tc_count"]}',
            'type': tc.get('type') or 'function',
            'function': fn,
        }
        state['tc_count'] += 1
        return delta

    def parse_tool_call_stream(
        self,
        state: Dict[str, Any],
        new_text: str,
        finished: bool = False,
    ) -> List[Dict[str, Any]]:
        """Hermes-style ``<tool_call>...</tool_call>`` streaming state machine.

        Buffers partial markup until a closing tag, then parses the block and
        emits a single ``tool_calls`` delta. Plain text is forwarded as
        ``content`` deltas, with the suffix held back when it could be the
        beginning of an incoming open tag.
        """
        state.setdefault('pending', '')
        state.setdefault('tc_count', 0)
        if new_text:
            state['pending'] += new_text

        events: List[Dict[str, Any]] = []
        while True:
            buf = state['pending']
            if not buf:
                break
            open_idx = buf.find(self._TOOL_CALL_OPEN)
            if open_idx == -1:
                # No open tag yet; defer trailing chars that could start one,
                # unless the stream is finished.
                partial = 0 if finished else self._trailing_prefix_of(buf, self._TOOL_CALL_OPEN)
                emit = buf[:-partial] if partial else buf
                state['pending'] = buf[-partial:] if partial else ''
                if emit:
                    events.append({'content': emit})
                break
            if open_idx > 0:
                events.append({'content': buf[:open_idx]})
                state['pending'] = buf[open_idx:]
                continue
            close_idx = buf.find(self._TOOL_CALL_CLOSE)
            if close_idx == -1:
                if finished:
                    # EOF with unclosed block: rely on _BLOCK_RE's \Z fallback.
                    try:
                        parsed = self.parse(buf) or []
                    except Exception:
                        logger.exception(
                            'parse_tool_call failed for unclosed streamed block; emitting as raw content')
                        events.append({'content': buf})
                        state['pending'] = ''
                        break
                    if parsed:
                        for tc in parsed:
                            events.append({'tool_calls': [self._format_tc_delta(state, tc)]})
                    else:
                        events.append({'content': buf})
                    state['pending'] = ''
                break
            block = buf[:close_idx + len(self._TOOL_CALL_CLOSE)]
            try:
                parsed = self.parse(block) or []
            except Exception:
                logger.exception(
                    'parse_tool_call failed for streamed block; emitting as raw content')
                events.append({'content': block})
                state['pending'] = buf[close_idx + len(self._TOOL_CALL_CLOSE):]
                continue
            for tc in parsed:
                events.append({'tool_calls': [self._format_tc_delta(state, tc)]})
            state['pending'] = buf[close_idx + len(self._TOOL_CALL_CLOSE):]
        return events
