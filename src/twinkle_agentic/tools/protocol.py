# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tool-call protocol abstraction.

Different model families emit tool calls in different wire formats:
Qwen3.5 uses XML ``<tool_call><function=...>...</function></tool_call>``,
OpenAI / Llama3 emit JSON function-call objects, etc. The rollout loop
takes a :class:`ToolCallProtocol` instance so the parser + output
sanitiser pair can be swapped per model without touching the
orchestration code.

Add a new family by subclassing :class:`ToolCallProtocol` and passing
your instance as ``tool_protocol=`` to :func:`run_agentic_rollouts`.
"""
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ToolCallProtocol(ABC):
    """Pair of (parser, output sanitiser) for one tool-call wire format."""

    @abstractmethod
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Return a list of ``{'tool_name': str, 'arguments': dict}``."""
        raise NotImplementedError

    @abstractmethod
    def clean(self, text: str) -> str:
        """Strip protocol-specific tool-call markup from assistant text."""
        raise NotImplementedError


# ─── Qwen3.5 native XML format ────────────────────────────────────────────────
_QWEN_TOOL_CALL_BLOCK_RE = re.compile(
    r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
_QWEN_FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
_QWEN_PARAMETER_RE = re.compile(
    r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
_QWEN_TOOL_CALL_STRIP_RE = re.compile(
    r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')


class Qwen35ToolCallProtocol(ToolCallProtocol):
    """Qwen3.5 native XML tool-call format (with JSON fallback).

    Recognised shape::

        <tool_call>
        <function=TOOL_NAME>
        <parameter=ARG_NAME>ARG_VALUE</parameter>
        </function>
        </tool_call>

    The ``\\Z`` branch in the block regex handles cases where the
    sampler stripped the closing ``</tool_call>`` stop token. If no
    ``<function>`` is found inside a block, parsing falls back to
    treating the block body as raw JSON (``{"name": ..., "arguments":
    ...}``) so malformed outputs still have a chance of parsing.
    """

    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block_m in _QWEN_TOOL_CALL_BLOCK_RE.finditer(text or ''):
            block = block_m.group(1)
            func_m = _QWEN_FUNCTION_RE.search(block)
            if func_m:
                args: Dict[str, Any] = {}
                for pm in _QWEN_PARAMETER_RE.finditer(func_m.group(2)):
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        args[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[key] = val
                calls.append({
                    'tool_name': func_m.group(1).strip(),
                    'arguments': args,
                })
                continue
            # JSON fallback
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
                'tool_name': name,
                'arguments': args if isinstance(args, dict) else {},
            })
        return calls

    def clean(self, text: str) -> str:
        return _QWEN_TOOL_CALL_STRIP_RE.sub('', text or '').rstrip()
