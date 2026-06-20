# Copyright (c) Twinkle Contributors. All rights reserved.
"""Skills loading framework - extensible providers for agent skill injection."""

from twinkle_tui.skills.base import SkillProvider
from twinkle_tui.skills.manager import SkillManager
from twinkle_tui.skills.modelscope_provider import ModelScopeSkillProvider

__all__ = ['SkillProvider', 'SkillManager', 'ModelScopeSkillProvider']
