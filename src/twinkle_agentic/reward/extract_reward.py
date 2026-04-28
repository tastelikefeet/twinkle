# Copyright (c) ModelScope Contributors. All rights reserved.
"""``ExtractReward``: penalise excessive use of the extract-compressed tool.

Pairs with :class:`twinkle_agentic.tools.extract.ExtractCompressed`.  The
extract tool lets the LLM recall pre-compression text of ``<block_N>`` markers
during a rollout; without a counter-pressure, a lazy policy would simply
expand every block every turn and defeat the purpose of condensation.  This
reward decays monotonically with the number of extract invocations, so the
policy learns to request recalls only when they actually help.
"""
import math
from typing import List, Optional

from twinkle.data_format import Trajectory
from twinkle.reward.base import Reward


class ExtractReward(Reward):
    """Reverse-sigmoid (logistic) reward in the number of extract-tool calls.

    For each trajectory, counts how many times the model invoked the extract
    tool (identified by ``tool_name``) across all assistant ``tool_calls``
    entries, then returns a logistic-decay score::

        reward(n) = max(min_reward,
                        max_reward / (1 + exp(steepness * (n - midpoint))))

    This shape has three regions:

    * **Plateau** (``n`` much smaller than ``midpoint``) -- the first few
      calls barely change the reward, so the policy is not punished for
      occasional legitimate recalls.
    * **Cliff** (``n`` near ``midpoint``) -- the reward drops sharply; this
      is the zone where the policy feels strong gradient pressure.
    * **Tail** (``n`` much larger than ``midpoint``) -- the reward is already
      close to zero and each extra call costs little; the floor
      ``min_reward`` bounds it from below.

    ``ground_truths`` is accepted for signature compatibility with
    :class:`Reward` but unused -- this reward is purely a behavioural
    regulariser.

    Args:
        tool_name: Name of the tool to count.  Defaults to
            ``'extract_compressed'``, matching
            :class:`twinkle_agentic.tools.extract.ExtractCompressed.name`.
        midpoint: Call count at which the reward crosses
            ``(max_reward + min_reward) / 2``.  Higher values push the
            cliff further right (more tolerant).  Defaults to ``3.0``.
        steepness: Slope of the transition.  Higher values make the cliff
            sharper (closer to a step function); lower values make the
            decay gentler.  Must be positive.  Defaults to ``1.5``.
        max_reward: Asymptotic upper bound (reward when ``n = 0`` is
            slightly below this; at ``n << midpoint`` it is indistinguishable
            from ``max_reward``).  Defaults to ``1.0``.
        min_reward: Floor for the decayed reward.  Defaults to ``0.0``.

    Example:
        >>> rf = ExtractReward(midpoint=3.0, steepness=1.5)
        >>> # Typical curve:
        >>> #   n=0 -> 0.989   (plateau)
        >>> #   n=1 -> 0.953
        >>> #   n=2 -> 0.818
        >>> #   n=3 -> 0.500   (midpoint)
        >>> #   n=4 -> 0.182
        >>> #   n=5 -> 0.047
        >>> #   n=8 -> 0.000   (tail -> floor)
    """

    def __init__(
        self,
        tool_name: str = 'extract_compressed',
        midpoint: float = 3.0,
        steepness: float = 1.5,
        max_reward: float = 1.0,
        min_reward: float = 0.0,
    ) -> None:
        if steepness <= 0:
            raise ValueError(f'steepness must be > 0, got {steepness}')
        if midpoint < 0:
            raise ValueError(f'midpoint must be >= 0, got {midpoint}')
        if max_reward < min_reward:
            raise ValueError(
                f'max_reward ({max_reward}) must be >= min_reward ({min_reward})')
        self.tool_name = tool_name
        self.midpoint = midpoint
        self.steepness = steepness
        self.max_reward = max_reward
        self.min_reward = min_reward

    def __call__(
        self,
        trajectories: List[Trajectory],
        ground_truths: Optional[List[Trajectory]] = None,
        **kwargs,
    ) -> List[float]:
        # ``ground_truths`` and ``kwargs`` accepted for signature compatibility
        # with the broader ``Reward`` family; this reward ignores both.
        del ground_truths, kwargs
        return [self._score_trajectory(t) for t in trajectories]

    # -- Internals ------------------------------------------------------------

    def _score_trajectory(self, trajectory: Trajectory) -> float:
        n = self._count_extract_calls(trajectory)
        # Logistic decay: starts flat near ``max_reward``, crosses
        # ``max_reward/2`` at ``n == midpoint``, asymptotes to 0.
        # ``math.exp`` is clamped on huge positive exponents to avoid overflow.
        exponent = self.steepness * (n - self.midpoint)
        reward = self.max_reward / (1.0 + math.exp(exponent)) if exponent < 700 else 0.0
        return max(self.min_reward, reward)

    def _count_extract_calls(self, trajectory: Trajectory) -> int:
        """Count ``tool_calls`` entries whose ``tool_name`` matches our target."""
        if not isinstance(trajectory, dict):
            return 0
        count = 0
        for msg in trajectory.get('messages') or ():
            # Only assistant turns carry tool_calls; skip everything else so a
            # ``role='tool'`` echo doesn't get double-counted.
            if not isinstance(msg, dict) or msg.get('role') != 'assistant':
                continue
            for tc in msg.get('tool_calls') or ():
                if isinstance(tc, dict) and tc.get('tool_name') == self.tool_name:
                    count += 1
        return count
