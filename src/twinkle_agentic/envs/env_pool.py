# Copyright (c) ModelScope Contributors. All rights reserved.
"""EnvPool: distributed environment pool with @remote_class integration.

Decorated with ``@remote_class(execute='all')``:
- With ``ranks=1``: single worker manages all environments (same as before).
- With ``ranks=N``: N workers each manage pool_size/N environments (sharded).
- Without ``remote_group``: runs locally in the current process.

The pool manages N environment instances sharded across workers. Each slot is
accessed by global index. :class:`EnvPoolAdapter` wraps a single slot as a
standard :class:`Env` so it can be used with :class:`EnvTool` / :class:`ToolManager`.

Usage (local)::

    pool = EnvPool(env_cls='blackjack_env:BlackjackEnv', pool_size=32)
    adapters = pool.get_adapters(tool_schema=TOOL_SCHEMA)

Usage (remote, single worker)::

    pool = EnvPool(
        env_cls='coding_env:CodingEnv', pool_size=8,
        remote_group='env', device_mesh=DeviceMesh.from_sizes(world_size=1, dp_size=1),
    )

Usage (remote, multi-worker distributed)::

    pool = EnvPool(
        env_cls='coding_env:CodingEnv', pool_size=32,
        remote_group='env', device_mesh=DeviceMesh.from_sizes(world_size=4, dp_size=4),
    )
    # 4 workers, each manages 8 environments
"""
import importlib
import json
import math
import os
from typing import Any, Callable, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.utils import get_logger
from .base import Env, StepResult

logger = get_logger()


def _collect_single(results, device_mesh=None):
    """Collect single-item results: pick the non-None result from workers."""
    for r in results:
        if r is not None:
            return r
    return None


def _collect_batch(results, device_mesh=None):
    """Collect batch results: merge (idx, result) pairs, return sorted by idx."""
    merged = []
    for worker_results in results:
        if worker_results:
            merged.extend(worker_results)
    merged.sort(key=lambda x: x[0])
    return [r[1] for r in merged]


def _import_env_class(path: str):
    """Import an environment class from a dotted or colon-separated path.

    Supports:
      - ``'module:ClassName'`` (entry-point style)
      - ``'module.ClassName'`` (dotted style)
    """
    if ':' in path:
        module_path, class_name = path.rsplit(':', 1)
    elif '.' in path:
        module_path, class_name = path.rsplit('.', 1)
    else:
        raise ValueError(
            f"env_cls must be 'module.ClassName' or 'module:ClassName', got {path!r}"
        )
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ImportError(f"Cannot find class {class_name!r} in module {module_path!r}")
    return cls


def _format_observation(obs) -> str:
    """Normalize observation to string."""
    if obs is None:
        return ''
    if isinstance(obs, str):
        return obs
    if isinstance(obs, dict):
        for key in ('result', 'output', 'content', 'text', 'message'):
            if key in obs:
                return str(obs[key])
        try:
            return json.dumps(obs, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(obs)
    return str(obs)


def _normalize_result(result) -> Dict[str, Any]:
    """Normalize an environment result to a standard dict."""
    # Already a dict
    if isinstance(result, dict):
        return {
            'observation': _format_observation(result.get('observation', '')),
            'reward': float(result.get('reward', 0.0)),
            'done': bool(result.get('done', False)),
        }
    # Has .observation attribute (StepResult, OpenEnv result, etc.)
    if hasattr(result, 'observation'):
        obs = result.observation
        return {
            'observation': _format_observation(obs),
            'reward': float(getattr(result, 'reward', 0.0) or 0.0),
            'done': bool(getattr(result, 'done', False)),
        }
    # Fallback
    return {
        'observation': str(result) if result is not None else '',
        'reward': 0.0,
        'done': False,
    }


def _accepts_two_positional(method) -> bool:
    """Check if method accepts >= 2 positional args (besides self)."""
    import inspect
    try:
        sig = inspect.signature(method)
        params = [
            p for p in sig.parameters.values()
            if p.name != 'self' and p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(params) >= 2
    except (ValueError, TypeError):
        return False


@remote_class(execute='all')
class EnvPool:
    """Distributed pool of environment instances managed as a Twinkle remote_class.

    When deployed with multiple workers (ranks > 1), environments are sharded
    across workers. Each worker manages pool_size // num_workers environments.

    Args:
        env_cls: Import path to the environment class (e.g.
            ``'blackjack_env:BlackjackEnv'``), or the class itself.
        pool_size: Total number of environment instances across all workers.
        device_mesh: Optional DeviceMesh for distributed deployment.
        env_kwargs: Extra keyword arguments for environment construction.
    """

    def __init__(
        self,
        env_cls: Union[str, Type],
        pool_size: int = 32,
        device_mesh: Optional[DeviceMesh] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        if isinstance(env_cls, str):
            self._env_cls = _import_env_class(env_cls)
        else:
            self._env_cls = env_cls

        self._pool_size = pool_size
        self._env_kwargs = env_kwargs or {}

        # Shard: each worker owns [_start, _end)
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        shard_size = math.ceil(pool_size / world_size)
        self._start = rank * shard_size
        self._end = min(self._start + shard_size, pool_size)
        local_size = self._end - self._start

        self._episode_rewards: List[float] = [0.0] * local_size
        self._envs: List[Any] = [
            self._env_cls(**self._env_kwargs) for _ in range(local_size)
        ]

        logger.info(f'EnvPool initialized: env_cls={env_cls}, pool_size={pool_size}, '
                    f'shard=[{self._start}, {self._end}), rank={rank}/{world_size}')

    def _owns(self, idx: int) -> bool:
        return self._start <= idx < self._end

    def _do_reset(self, idx: int) -> Dict[str, Any]:
        local_idx = idx - self._start
        env = self._envs[local_idx]
        self._episode_rewards[local_idx] = 0.0
        result = env.reset()
        normalized = _normalize_result(result)
        normalized['episode_reward'] = 0.0
        return normalized

    def _do_step(self, idx: int, action: Dict[str, Any]) -> Dict[str, Any]:
        local_idx = idx - self._start
        env = self._envs[local_idx]

        if 'tool_name' in action and 'arguments' in action:
            if hasattr(env, 'step') and _accepts_two_positional(env.step):
                result = env.step(action['tool_name'], action['arguments'])
            else:
                result = env.step(action)
        else:
            result = env.step(action)

        normalized = _normalize_result(result)
        self._episode_rewards[local_idx] += normalized['reward']
        normalized['episode_reward'] = self._episode_rewards[local_idx]
        return normalized

    @remote_function(dispatch='all', collect=_collect_single)
    def reset(self, idx: int) -> Dict[str, Any]:
        """Reset environment instance at global slot ``idx``."""
        if not self._owns(idx):
            return None
        return self._do_reset(idx)

    @remote_function(dispatch='all', collect=_collect_single)
    def step(self, idx: int, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step on environment at global slot ``idx``."""
        if not self._owns(idx):
            return None
        return self._do_step(idx, action)

    @remote_function(dispatch='all', collect=_collect_batch)
    def reset_batch(self, indices: List[int]) -> List:
        """Batch reset multiple environments (only processes owned slots)."""
        results = []
        for i in indices:
            if self._owns(i):
                results.append((i, self._do_reset(i)))
        return results

    @remote_function(dispatch='all', collect=_collect_batch)
    def step_batch(self, indices: List[int], actions: List[Dict[str, Any]]) -> List:
        """Batch step multiple environments (only processes owned slots)."""
        results = []
        for i, a in zip(indices, actions):
            if self._owns(i):
                results.append((i, self._do_step(i, a)))
        return results

    @remote_function(dispatch='all', collect='first')
    def close(self) -> None:
        """Release all environment resources."""
        for env in self._envs:
            if hasattr(env, 'close'):
                try:
                    env.close()
                except Exception:
                    pass
        self._envs.clear()
        logger.info('EnvPool closed.')

    @property
    def pool_size(self) -> int:
        return self._pool_size

    def get_adapters(
        self,
        n: Optional[int] = None,
        tool_schema: Optional[List] = None,
        action_mapper: Optional[Callable] = None,
    ) -> List['EnvPoolAdapter']:
        """Create EnvPoolAdapter instances for use with EnvTool/ToolManager.

        Args:
            n: Number of adapters to create (default: pool_size).
            tool_schema: Tool definitions for the LLM.
            action_mapper: Optional callable to transform (tool_name, arguments)
                before passing to the environment.

        Returns:
            List of EnvPoolAdapter instances (indices 0..n-1).
        """
        pool_size = getattr(self, '_pool_size', None)
        if n is None:
            if pool_size is None:
                raise ValueError('n must be specified when calling get_adapters from driver')
            n = pool_size
        if pool_size is not None and n > pool_size:
            raise ValueError(
                f'Requested {n} adapters but pool only has {pool_size} slots.'
            )
        return [
            EnvPoolAdapter(pool=self, idx=i, tool_schema=tool_schema, action_mapper=action_mapper)
            for i in range(n)
        ]


class EnvPoolAdapter(Env):
    """Wraps a single slot in an :class:`EnvPool` as a standard :class:`Env`.

    This adapter allows a pool slot to be used transparently with
    :class:`EnvTool`, :class:`ToolManager`, and :class:`MultiTurnRollout`.

    Args:
        pool: The EnvPool instance.
        idx: Slot index in the pool.
        tool_schema: Tool definitions for the environment.
        action_mapper: Optional callable to transform (tool_name, arguments).
    """

    def __init__(
        self,
        pool: EnvPool,
        idx: int,
        tool_schema: Optional[List] = None,
        action_mapper: Optional[Callable] = None,
    ):
        self._pool = pool
        self._idx = idx
        self._tool_schema = tool_schema
        self._action_mapper = action_mapper
        self._episode_reward: float = 0.0

    def reset(self, trajectory=None) -> StepResult:
        """Reset the environment slot."""
        self._episode_reward = 0.0
        result = self._pool.reset(self._idx)
        return StepResult(
            observation=result.get('observation', ''),
            reward=0.0,
            done=False,
            info=result,
        )

    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        """Execute a tool call on the environment slot."""
        try:
            if self._action_mapper is not None:
                action = self._action_mapper(tool_name, arguments)
            else:
                action = {'tool_name': tool_name, 'arguments': arguments}

            result = self._pool.step(self._idx, action)
            obs = result.get('observation', '')
            reward = float(result.get('reward', 0.0))
            done = bool(result.get('done', False))
            self._episode_reward = float(result.get('episode_reward', 0.0))

            return StepResult(
                observation=obs,
                reward=reward,
                done=done,
                info={'raw_result': result, 'episode_reward': self._episode_reward},
            )
        except Exception as e:
            logger.warning(f'EnvPoolAdapter step error (idx={self._idx}): {e}')
            return StepResult(
                observation=f'Error: {e}',
                reward=0.0,
                done=True,
                info={'error': str(e)},
            )

    def tools(self) -> List:
        """Return tool definitions."""
        if self._tool_schema is not None:
            return self._tool_schema
        return []

    def evaluate(self, trajectories, **kwargs) -> List[float]:
        """No-op: rewards are accumulated per-step."""
        return [0.0] * len(trajectories)

    @property
    def episode_reward(self) -> float:
        """Cumulative reward for the current episode."""
        return self._episode_reward

    def close(self) -> None:
        """No-op: lifecycle managed by the pool."""
        pass
