# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ToolCallParser(ABC):
    """Single-format tool-call parser."""

    name: str = ''
    open_marker: Optional[str] = None
    close_marker: Optional[str] = None

    def matches_model(self, model_id: str) -> bool:
        """Return True if this parser is the canonical choice for ``model_id``.

        Used for streaming where we must commit to a parser before any text
        has arrived. Default False — parser is text-detection-only.
        """
        return False

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Cheap pre-check: does ``text`` carry this format's markup?"""

    @abstractmethod
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Return OpenAI-shape tool_calls. ``arguments`` is a dict (jinja-friendly)."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Strip parser-specific markup; return plain content text."""


class ToolCallRegistry:
    """Global ordered registry of :class:`ToolCallParser` instances."""

    _parsers: List[ToolCallParser] = []

    @classmethod
    def register(cls, parser: ToolCallParser) -> ToolCallParser:
        for p in cls._parsers:
            if p.name == parser.name:
                return p
        cls._parsers.append(parser)
        return parser

    @classmethod
    def parsers(cls) -> List[ToolCallParser]:
        return list(cls._parsers)

    @classmethod
    def select_for_model(cls, model_id: Optional[str]) -> Optional[ToolCallParser]:
        mid = (model_id or '').lower()
        for p in cls._parsers:
            if p.matches_model(mid):
                return p
        return None

    @classmethod
    def detect_first(cls, text: str) -> Optional[ToolCallParser]:
        if not text:
            return None
        for p in cls._parsers:
            if p.detect(text):
                return p
        return None


def trailing_prefix_of(buf: str, marker: str) -> int:
    """Length of trailing chars of ``buf`` that form a strict prefix of ``marker``.

    Used by streaming protocols to hold back the tail when it could be the
    start of an upcoming open tag, preventing mid-marker splits.
    """
    upper = min(len(marker) - 1, len(buf))
    for k in range(upper, 0, -1):
        if buf.endswith(marker[:k]):
            return k
    return 0
