# Copyright (c) Twinkle Contributors. All rights reserved.
"""Base class for skill providers.

A SkillProvider is responsible for fetching skill files (typically markdown)
from a remote source (Git repo, API, local directory, etc.) and making their
content available for injection into the agent's system prompt.

To create a new provider, subclass SkillProvider and implement:
- `name` property: human-readable provider name
- `fetch()`: download/update skill files to local cache
- `load_skills()`: read cached files and return list of Skill objects
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path


@dataclasses.dataclass
class Skill:
    """A single skill loaded from a provider.

    Attributes:
        name: Short identifier for the skill (typically filename without extension).
        content: Full markdown content of the skill.
        source: Provider name + relative path for traceability.
    """

    name: str
    content: str
    source: str


class SkillProvider(ABC):
    """Abstract base class for skill providers.

    Subclass this to add new skill sources (e.g., HuggingFace, local files).
    """

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the provider.

        Args:
            cache_dir: Local directory to cache fetched skill files.
                       If None, uses ~/.cache/twinkle_tui/skills/<provider_name>
        """
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'twinkle_tui' / 'skills' / self.name
        self.cache_dir = cache_dir

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this provider (used as cache subdirectory)."""
        ...

    @abstractmethod
    async def fetch(self) -> None:
        """Fetch or update skill files from the remote source.

        Should clone/pull a git repo, download files, etc.
        Results should be stored in self.cache_dir.
        """
        ...

    @abstractmethod
    async def load_skills(self) -> list[Skill]:
        """Load all skills from the local cache.

        Returns:
            List of Skill objects with their content loaded.
        """
        ...

    async def get_skills(self) -> list[Skill]:
        """Convenience method: fetch then load.

        Can be overridden if a provider wants custom logic.
        """
        await self.fetch()
        return await self.load_skills()
