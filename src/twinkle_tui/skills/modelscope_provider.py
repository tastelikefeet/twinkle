# Copyright (c) Twinkle Contributors. All rights reserved.
"""ModelScope skill provider - fetches skills from modelscope-skills GitHub repo."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from twinkle_tui.skills.base import Skill, SkillProvider

logger = logging.getLogger(__name__)

# Default repo URL; can be overridden via constructor
_DEFAULT_REPO_URL = 'https://github.com/modelscope/modelscope-skills.git'
_DEFAULT_BRANCH = 'main'


class ModelScopeSkillProvider(SkillProvider):
    """Fetches skill markdown files from the modelscope-skills GitHub repository.

    Skills are cloned to a local cache directory. On subsequent calls,
    the repo is pulled to get updates.

    Usage:
        provider = ModelScopeSkillProvider()
        skills = await provider.get_skills()
    """

    def __init__(
        self,
        repo_url: str = _DEFAULT_REPO_URL,
        branch: str = _DEFAULT_BRANCH,
        cache_dir: Path | None = None,
    ):
        self._repo_url = repo_url
        self._branch = branch
        super().__init__(cache_dir=cache_dir)

    @property
    def name(self) -> str:
        return 'modelscope'

    async def fetch(self) -> None:
        """Clone or pull the modelscope-skills repository."""
        repo_dir = self.cache_dir / 'repo'

        if (repo_dir / '.git').exists():
            # Already cloned — pull latest
            logger.info('Updating modelscope-skills repo...')
            proc = await asyncio.create_subprocess_exec(
                'git', '-C', str(repo_dir), 'pull', '--ff-only',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(f'git pull failed: {stderr.decode().strip()}')
        else:
            # First time — clone
            logger.info(f'Cloning modelscope-skills from {self._repo_url}...')
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            proc = await asyncio.create_subprocess_exec(
                'git', 'clone', '--depth', '1', '--branch', self._branch,
                self._repo_url, str(repo_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error(f'git clone failed: {stderr.decode().strip()}')

    async def load_skills(self) -> list[Skill]:
        """Load all .md files from the cloned repository as skills.

        Scans recursively, skipping LICENSE, README, and hidden files.
        """
        repo_dir = self.cache_dir / 'repo'
        if not repo_dir.exists():
            logger.warning('modelscope-skills repo not found. Call fetch() first.')
            return []

        skills: list[Skill] = []
        skip_names = {'license', 'readme', 'contributing', 'changelog'}

        for md_file in sorted(repo_dir.rglob('*.md')):
            # Skip hidden dirs (e.g., .github)
            if any(part.startswith('.') for part in md_file.relative_to(repo_dir).parts):
                continue
            # Skip common non-skill files
            if md_file.stem.lower() in skip_names:
                continue

            try:
                content = md_file.read_text(encoding='utf-8')
                rel_path = md_file.relative_to(repo_dir)
                skills.append(Skill(
                    name=md_file.stem,
                    content=content,
                    source=f'modelscope-skills/{rel_path}',
                ))
            except Exception as e:
                logger.warning(f'Failed to read skill file {md_file}: {e}')

        logger.info(f'Loaded {len(skills)} skills from modelscope-skills')
        return skills
