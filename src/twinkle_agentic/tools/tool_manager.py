# Copyright (c) ModelScope Contributors. All rights reserved.
"""``ToolManager``: name-indexed registry and dispatcher for RL-time tools."""
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from twinkle.data_format.message import Tool as ToolInfo
from twinkle.data_format.message import ToolCall

from .base import Tool


class ToolManager:
    """Registry + dispatcher for :class:`Tool` instances used during RL.

    A thin container that:

    * stores named :class:`Tool` instances (keyed by their
      ``tool_info()['tool_name']``),
    * advertises their schemas via :meth:`tool_infos` for prompt injection,
    * dispatches :class:`ToolCall` payloads to the matching tool by name,
      always returning a string suitable for a ``role='tool'`` message.

    The manager does not own tool state: each tool is responsible for its
    own lifecycle (e.g. :class:`ExtractCompressed` captures a reference to
    the pre-compression chunks).  :meth:`register` is idempotent per name
    -- the latest registration wins -- so tools can be swapped per-rollout.

    Args:
        tools: Optional iterable of :class:`Tool` instances to register on
            construction.  Equivalent to calling :meth:`register` for each.

    Example:
        >>> mgr = ToolManager()
        >>> mgr.register(ExtractCompressed(original_chunks=full_chunks))
        >>> # 1. advertise to the LLM (goes into the system prompt / tool field)
        >>> mgr.tool_infos()
        [{'tool_name': 'extract_compressed', 'description': '...',
          'parameters': '...'}]
        >>> # 2. the LLM emits a tool_call; dispatch returns the tool result
        >>> result = mgr.dispatch({
        ...     'tool_name': 'extract_compressed',
        ...     'arguments': '{"blocks": [1, 3]}',
        ... })
        >>> # feed it back as a tool message
        >>> {'role': 'tool', 'content': result, 'tool_call_id': '...'}
    """

    def __init__(self, tools: Optional[Iterable[Tool]] = None) -> None:
        self._tools: Dict[str, Tool] = {}
        for t in tools or ():
            self.register(t)

    # -- Registration ---------------------------------------------------------

    def register(self, tool: Tool) -> Tool:
        """Register (or replace) a tool keyed by ``tool_info()['tool_name']``."""
        if not isinstance(tool, Tool):
            raise TypeError(f'expected a Tool instance, got {type(tool).__name__}')
        info = tool.tool_info()
        name = info.get('tool_name') if isinstance(info, dict) else None
        if not name:
            raise ValueError(
                f'tool {type(tool).__name__} must expose a non-empty '
                f'tool_info()["tool_name"]')
        self._tools[name] = tool
        return tool

    def unregister(self, name: str) -> Optional[Tool]:
        """Remove and return the named tool, or ``None`` if it was not registered."""
        return self._tools.pop(name, None)

    # -- Introspection --------------------------------------------------------

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> List[str]:
        return list(self._tools)

    def tool_infos(self) -> List[ToolInfo]:
        """Return the list of :class:`ToolInfo` schemas for prompt injection."""
        return [t.tool_info() for t in self._tools.values()]

    # -- Dispatch -------------------------------------------------------------

    def dispatch(self, tool_call: Union[ToolCall, Dict[str, Any]]) -> str:
        """Dispatch a :class:`ToolCall` to its registered tool.

        The return value is always a string so it can be dropped straight
        into a ``role='tool'`` message.  Unknown or malformed calls produce
        an ``Error: ...`` string rather than raising, so a misbehaving LLM
        cannot crash the rollout -- it just gets a descriptive error
        message back as the tool result.
        """
        if not isinstance(tool_call, dict):
            return f'Error: tool_call must be an object, got {type(tool_call).__name__}.'
        name = tool_call.get('tool_name')
        if not name:
            return 'Error: tool_call missing "tool_name".'
        if (tool := self._tools.get(name)) is None:
            available = ', '.join(sorted(self._tools)) or '(none)'
            return f'Error: unknown tool {name!r}. Available: {available}.'

        raw_args = tool_call.get('arguments')
        if raw_args is None:
            args: Dict[str, Any] = {}
        elif isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError as e:
                return f'Error: invalid JSON in arguments: {e}'
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            return (f'Error: "arguments" must be a JSON string or object, '
                    f'got {type(raw_args).__name__}.')

        try:
            # ``str(x)`` is a no-op when ``x`` is already a string, so this also
            # normalises accidental non-string returns from misbehaving tools.
            return str(tool(name, args))
        except Exception as e:  # noqa: BLE001 -- surface tool failures as tool output
            return f'Error: tool {name!r} raised {type(e).__name__}: {e}'
