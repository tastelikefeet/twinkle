import json
from typing import List, Optional, Dict, Union, Any

from fastmcp.utilities.inspect import ToolInfo

from twinkle.data_format import ToolCall
from twinkle_agentic.tools.base import Tool


class ToolManager:

    def __init__(self, tools: Dict[str, Tool]):
        self._tools = tools

    def register(self, tool: Tool):
        info = tool.tool_info()
        name = info.get('tool_name') if isinstance(info, dict) else None
        if not name:
            raise ValueError(
                f'tool {type(tool).__name__} must expose a non-empty '
                f'tool_info()["tool_name"]')
        self._tools[name] = tool

    def unregister(self, name: str) -> Optional[Tool]:
        return self._tools.pop(name, None)

    def names(self) -> List[str]:
        return list(self._tools)

    def tool_infos(self) -> List[ToolInfo]:
        return [t.tool_info() for t in self._tools.values()]

    def __call__(self, tool_call: Union[ToolCall, Dict[str, Any]]) -> str:
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
            return str(tool(name, args))
        except Exception as e: # noqa
            return f'Error: tool {name!r} raised {type(e).__name__}: {e}'