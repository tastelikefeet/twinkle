# Copyright (c) ModelScope Contributors. All rights reserved.
"""EnvPool: environment pool with @remote_class integration.

Follows the same pattern as :class:`twinkle.dataloader.DataLoader`:
- Decorated with ``@remote_class(execute='first')``
- When instantiated **without** ``remote_group``, runs locally in the
  current process (driver or worker) with zero RPC overhead.
- When instantiated **with** ``remote_group='env'``, gets deployed to a
  dedicated Ray Worker for process-level isolation.

The pool manages N environment instances internally. Each slot is accessed
by index. :class:`EnvPoolAdapter` wraps a single slot as a standard
:class:`Env` so it can be used with :class:`EnvTool` / :class:`ToolManager`.

Usage (local, inside MultiTurnRollout worker)::

    pool = EnvPool(env_cls='blackjack_env:BlackjackEnv', pool_size=32)
    adapters = pool.get_adapters(tool_schema=TOOL_SCHEMA)
    # adapters[i] is a standard Env

Usage (remote, on a dedicated DeviceGroup)::

    pool = EnvPool(
        env_cls='coding_env:CodingEnv',
        pool_size=8,
        remote_group='env',
        device_mesh=env_mesh,
    )
"""
import importlib
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union

from twinkle import DeviceMesh, remote_class, remote_function
from twinkle.utils import get_logger
from .base import Env, StepResult

logger = get_logger()


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


@remote_class(execute='first')
class EnvPool:
    """Pool of environment instances managed as a Twinkle remote_class.

    Args:
        env_cls: Import path to the environment class (e.g.
            ``'blackjack_env:BlackjackEnv'``), or the class itself.
        pool_size: Number of environment instances to create.
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
        # Resolve env class
        if isinstance(env_cls, str):
            self._env_cls = _import_env_class(env_cls)
        else:
            self._env_cls = env_cls

        self._pool_size = pool_size
        self._env_kwargs = env_kwargs or {}
        self._episode_rewards: List[float] = [0.0] * pool_size

        # Instantiate all environments
        self._envs: List[Any] = []
        for _ in range(pool_size):
            self._envs.append(self._env_cls(**self._env_kwargs))

        logger.info(f'EnvPool initialized: env_cls={env_cls}, pool_size={pool_size}')

    @remote_function()
    def reset(self, idx: int) -> Dict[str, Any]:
        """Reset environment instance at slot ``idx``.

        Returns:
            Dict with keys: observation, reward, done.
        """
        env = self._envs[idx]
        self._episode_rewards[idx] = 0.0
        result = env.reset()
        normalized = _normalize_result(result)
        normalized['episode_reward'] = 0.0
        return normalized

    @remote_function()
    def step(self, idx: int, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one step on environment at slot ``idx``.

        Args:
            idx: Environment slot index.
            action: Action dict. If it contains 'tool_name' and 'arguments',
                dispatches as ``env.step(tool_name, arguments)`` for Twinkle
                Env protocol. Otherwise passes the dict directly.

        Returns:
            Dict with keys: observation, reward, done, episode_reward.
        """
        env = self._envs[idx]

        # Dispatch based on env interface
        if 'tool_name' in action and 'arguments' in action:
            if hasattr(env, 'step') and _accepts_two_positional(env.step):
                result = env.step(action['tool_name'], action['arguments'])
            else:
                result = env.step(action)
        else:
            result = env.step(action)

        normalized = _normalize_result(result)
        self._episode_rewards[idx] += normalized['reward']
        normalized['episode_reward'] = self._episode_rewards[idx]
        return normalized

    @remote_function()
    def reset_batch(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Batch reset multiple environments.

        Args:
            indices: List of slot indices to reset.

        Returns:
            List of result dicts, one per index.
        """
        return [self.reset(i) for i in indices]

    @remote_function()
    def step_batch(self, indices: List[int], actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch step multiple environments.

        Args:
            indices: List of slot indices.
            actions: List of action dicts, aligned with indices.

        Returns:
            List of result dicts, one per index.
        """
        assert len(indices) == len(actions)
        return [self.step(i, a) for i, a in zip(indices, actions)]

    @remote_function()
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
        if n is None:
            n = self._pool_size
        if n > self._pool_size:
            raise ValueError(
                f'Requested {n} adapters but pool only has {self._pool_size} slots.'
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
