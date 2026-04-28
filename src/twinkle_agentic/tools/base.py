# Copyright (c) ModelScope Contributors. All rights reserved.
"""Abstract base class for RL-time tools exposed to the LLM."""
from abc import ABC, abstractmethod
from typing import Any, Dict

from twinkle.data_format.message import Tool as ToolInfo


class Tool(ABC):
    """Abstract base for RL-time tools.

    A :class:`Tool` is a callable unit the LLM can invoke during a rollout.
    Concrete subclasses must implement:

    * :meth:`__call__` -- receives the LLM-provided ``arguments`` dict (already
      parsed from JSON) and returns a string result to be dropped into a
      ``role='tool'`` message.
    * :meth:`tool_info` -- advertises the tool's schema (name / description /
      JSON-schema parameters) to the prompt.

    Any exception raised from :meth:`__call__` should be surfaced as a
    ``role='tool'`` error string rather than propagating, so a misbehaving
    LLM does not crash the rollout.  The convention is enforced by
    :class:`ToolManager.dispatch`.
    """

    @abstractmethod
    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def tool_info(self) -> ToolInfo:
        raise NotImplementedError
