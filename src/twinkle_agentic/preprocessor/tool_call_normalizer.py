# Copyright (c) ModelScope Contributors. All rights reserved.
"""Normalize non-standard tool-call formats to OpenAI-native schema.

Multiple agent frameworks embed tool calls in assistant ``content`` using
proprietary markup instead of the standard ``tool_calls`` field:

- **Cline**: XML tags (``<read_file><path>...</path></read_file>``)
- **ReAct**: ``Action: tool_name[args]``
- **VCP**: ``<<<[TOOL_REQUEST]>>>...<<<[END_TOOL_REQUEST]>>>``
- **Hermes/Qwen**: ``<tool_call>{...}</tool_call>``

This normalizer uses :class:`ToolCallRegistry.detect_first` to identify ANY
registered format in assistant content and rewrites the conversation to
standard ``tool_calls`` + ``role=tool`` so downstream template rendering
and intent classification work correctly.

Tool-result heuristic: the user message immediately following a converted
assistant is treated as a tool result if it matches common patterns
(``[tool for '...'] Result:``, ``Tool output:``, bare content after tool call).
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.template.tools import ToolCallRegistry
from twinkle.template.tools.base import ToolCallParser


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rewrite: embedded tool calls → tool_calls field; following user tool-results → role=tool."""
    out: List[Dict[str, Any]] = []
    call_counter = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get('role', '')
        content = msg.get('content') or ''

        if role == 'assistant' and not msg.get('tool_calls'):
            parser: Optional[ToolCallParser] = ToolCallRegistry.detect_first(content)
            if parser:
                parsed = parser.parse(content)
                if parsed:
                    cleaned = parser.clean(content)
                    tc_list = []
                    for tc in parsed:
                        call_counter += 1
                        args = tc['function']['arguments']
                        tc_list.append({
                            'id': f'call_norm_{call_counter:04d}',
                            'type': 'function',
                            'function': {
                                'name': tc['function']['name'],
                                'arguments': json.dumps(args, ensure_ascii=False)
                                             if isinstance(args, dict) else str(args),
                            },
                        })
                    out.append({
                        'role': 'assistant',
                        'content': cleaned,
                        'tool_calls': json.dumps(tc_list, ensure_ascii=False),
                        'tool_call_id': '',
                    })
                    # Consume following user messages that are tool results.
                    j = i + 1
                    tc_idx = 0
                    while j < len(messages) and tc_idx < len(tc_list):
                        nxt = messages[j]
                        if nxt.get('role') != 'user':
                            break
                        nxt_content = nxt.get('content') or ''
                        if parser.detect_result(nxt_content):
                            result_body = parser.parse_result(nxt_content)
                        elif tc_idx == 0 and len(tc_list) == 1:
                            # Single tool call, next user msg is the result.
                            result_body = nxt_content
                        else:
                            break
                        out.append({
                            'role': 'tool',
                            'content': result_body,
                            'tool_calls': '',
                            'tool_call_id': tc_list[tc_idx]['id'],
                        })
                        tc_idx += 1
                        j += 1
                    i = j
                    continue

        out.append(msg)
        i += 1

    return out


class ToolCallNormalizer(Preprocessor):
    """Normalize any registered non-standard tool-call format to OpenAI schema.

    Uses ToolCallRegistry to detect embedded tool calls in assistant content
    (Cline XML, ReAct, VCP, Hermes, etc.) and rewrites:
    - assistant.tool_calls: populated with JSON string of OpenAI-format list
    - assistant.content: cleaned (markup removed)
    - following user messages carrying tool results: converted to role=tool

    Non-matching rows pass through unchanged. This is a mapper (never drops rows).
    """

    def __call__(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out = []
        for row in rows:
            msgs = row.get('messages')
            if not msgs or not isinstance(msgs, list):
                out.append(row)
                continue
            has_embedded = any(
                m.get('role') == 'assistant'
                and not m.get('tool_calls')
                and ToolCallRegistry.detect_first(m.get('content') or '')
                for m in msgs
            )
            if not has_embedded:
                out.append(row)
                continue
            row = dict(row)
            row['messages'] = _normalize_messages(msgs)
            out.append(row)
        return out, []
