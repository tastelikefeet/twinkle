# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ToolCallParser(ABC):
    """Abstract base for a model family's tool-call wire format."""

    @abstractmethod
    def parse(self, decoded: str) -> List[Dict[str, Any]]:
        """Return Twinkle-shape tool calls: ``[{'tool_name', 'arguments'}]``.

        ``arguments`` is a ``dict`` (will be JSON-serialised downstream).
        Implementations must be tolerant of truncated / stop-token-stripped
        output \u2014 e.g. a missing closing ``</tool_call>`` at end-of-stream.
        """

    @abstractmethod
    def clean(self, decoded: str) -> str:
        """Strip family-specific tool-call markup from the assistant text."""


class QwenToolCallParser(ToolCallParser):
    """Qwen3.5 native XML tool-call format.

    ``<tool_call><function=NAME><parameter=KEY>VAL</parameter></function></tool_call>``

    ``\\Z`` branches handle sampler-stripped closing tokens (the stop
    token ``</tool_call>`` can be swallowed by some vLLM configs).
    Falls back to JSON ``{"name", "arguments"}`` inside the block for
    older Qwen tool-call dumps that still used the JSON form.
    """

    _BLOCK_RE = re.compile(
        r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
    _FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
    _PARAMETER_RE = re.compile(
        r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
    _STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')

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
                    'tool_name': func_m.group(1).strip(),
                    'arguments': args,
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
                'tool_name': name,
                'arguments': args if isinstance(args, dict) else {},
            })
        return calls

    def clean(self, decoded: str) -> str:
        return self._STRIP_RE.sub('', decoded or '').rstrip()


# Module-level singletons: parsers are stateless, so one instance is enough.
QWEN_TOOL_CALL_PARSER = QwenToolCallParser()
