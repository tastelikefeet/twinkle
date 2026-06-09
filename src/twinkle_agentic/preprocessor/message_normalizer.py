# Copyright (c) ModelScope Contributors. All rights reserved.
"""Normalize message sequences to standard OpenAI multi-turn schema.

Two passes:
1. **Tool-call normalization** — rewrite embedded tool calls (Cline XML, ReAct,
   VCP, Hermes) to standard ``tool_calls`` field + ``role=tool``.
2. **Consecutive-role merging** — merge adjacent messages with the same role
   into a single message (newline-joined content). Handles:
   - Consecutive system (multi-part system prompts, skills injections)
   - Consecutive user (heartbeat, system-reminder injections, empty pings)
   - Consecutive assistant (chunked responses)
   - Consecutive tool (parallel tool results already handled by schema, but
     tolerate duplicates)

Empty messages within a consecutive run are silently dropped.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.template.tools import ToolCallRegistry
from twinkle.template.tools.base import ToolCallParser

_HEARTBEAT_USER_RE = re.compile(
    r'Read HEARTBEAT\.md|heartbeat|HEARTBEAT_OK|keep.?alive',
    re.IGNORECASE,
)
_HEARTBEAT_ASST_RE = re.compile(
    r'heartbeat|HEARTBEAT_OK|HEARTBEAT_|duplicate heartbeat',
    re.IGNORECASE,
)


def _strip_heartbeat(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove heartbeat polling rounds (user + assistant)."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get('role', '')
        content = (m.get('content') or '')[:300]
        if role == 'user' and _HEARTBEAT_USER_RE.search(content):
            continue
        if role == 'assistant' and not m.get('tool_calls') and _HEARTBEAT_ASST_RE.search(content):
            continue
        out.append(m)
    # Trim leading non-user/non-system to preserve role_order invariant
    while out and out[0].get('role') not in ('user', 'system'):
        out.pop(0)
    return out


# ── Tool-call normalization ──────────────────────────────────────────────────

def _normalize_tool_calls(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


# ── Consecutive-role merging ─────────────────────────────────────────────────

def _merge_consecutive(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge adjacent messages with the same role into one, joining content with newline."""
    if not messages:
        return messages

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get('role', '')

        # tool messages: don't merge — each has its own tool_call_id
        if role == 'tool':
            if (msg.get('content') or '').strip():
                out.append(msg)
            i += 1
            continue

        # assistant with tool_calls: don't merge — tool_calls are position-sensitive
        if role == 'assistant' and msg.get('tool_calls'):
            out.append(msg)
            i += 1
            continue

        # Collect consecutive run of same role (non-tool, non-tool_calls-assistant)
        run = [msg]
        j = i + 1
        while j < len(messages):
            nxt = messages[j]
            nxt_role = nxt.get('role', '')
            if nxt_role != role:
                break
            if nxt_role == 'assistant' and nxt.get('tool_calls'):
                break
            run.append(nxt)
            j += 1

        # Merge the run
        parts = []
        for m in run:
            c = (m.get('content') or '').strip()
            if c:
                parts.append(c)

        if parts:
            merged = dict(run[0])
            merged['content'] = '\n'.join(parts)
            out.append(merged)
        # If all empty, drop the entire run

        i = j

    return out


# ── Combined normalizer ──────────────────────────────────────────────────────

class MessageNormalizer(Preprocessor):
    """Two-pass message normalizer: tool-call extraction + consecutive-role merge.

    Pass 1: detect embedded tool calls (Cline XML, ReAct, VCP, Hermes) and
    rewrite to OpenAI schema (tool_calls field + role=tool).

    Pass 2: merge consecutive same-role messages into one (content joined by
    newline). Empty messages are dropped. tool messages and assistant messages
    with tool_calls are never merged.

    This is a mapper — never drops rows.
    """

    def __call__(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        out = []
        for row in rows:
            msgs = row.get('messages')
            if not msgs or not isinstance(msgs, list):
                out.append(row)
                continue

            # Pass 0: strip heartbeat polling rounds
            msgs = _strip_heartbeat(msgs)

            # Pass 1: tool-call normalization
            has_embedded = any(
                isinstance(m, dict)
                and m.get('role') == 'assistant'
                and not m.get('tool_calls')
                and ToolCallRegistry.detect_first(m.get('content') or '')
                for m in msgs
            )
            if has_embedded:
                msgs = _normalize_tool_calls(msgs)

            # Pass 2: merge consecutive same-role
            msgs = _merge_consecutive(msgs)

            row = dict(row)
            row['messages'] = msgs
            out.append(row)
        return out, []

