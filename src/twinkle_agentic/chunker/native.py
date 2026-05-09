# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rule-based trajectory chunker: splits Trajectory into Chunks."""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Message, ToolCall
from twinkle.template import Template
from twinkle_agentic.data_format import Chunks
from twinkle_agentic.data_format import Chunk

from .base import Chunker



class NativeChunker(Chunker):

    def __call__(self, trajectory: Trajectory) -> Chunks:
        pass