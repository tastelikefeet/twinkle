# Copyright (c) Twinkle Contributors. All rights reserved.
"""Agent tool definitions for training control, search, and metrics."""

from __future__ import annotations

import json
from typing import Any, Callable

from twinkle_tui.connection import LocalConnection

# --- Tool Registry ---

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'list_training_runs',
            'description': 'List all active and historical training runs.',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'get_training_status',
            'description': 'Get detailed status and recent metrics for a training run.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'pause_training',
            'description': 'Pause a running training job. Training will pause after the current step completes.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to pause.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'resume_training',
            'description': 'Resume a paused training job.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to resume.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'stop_training',
            'description': 'Stop a training job permanently. The training script will exit after the current step.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID to stop.'},
                },
                'required': ['run_id'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_datasets',
            'description': 'Search ModelScope for datasets matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query for datasets.'},
                    'limit': {'type': 'integer', 'description': 'Max results.', 'default': 5},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_models',
            'description': 'Search ModelScope for models matching a query.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'Search query for models.'},
                    'limit': {'type': 'integer', 'description': 'Max results.', 'default': 5},
                },
                'required': ['query'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'zoom_metrics',
            'description': 'Adjust the metrics chart view. Use to zoom in/out on specific step ranges or reset.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'action': {
                        'type': 'string',
                        'enum': ['zoom', 'reset'],
                        'description': '"zoom" to set range, "reset" to show all.',
                    },
                    'x_start': {'type': 'integer', 'description': 'Start step for x-axis.'},
                    'x_end': {'type': 'integer', 'description': 'End step for x-axis.'},
                    'y_min': {'type': 'number', 'description': 'Min value for y-axis.'},
                    'y_max': {'type': 'number', 'description': 'Max value for y-axis.'},
                },
                'required': ['action'],
            },
        },
    },
]


class ToolExecutor:
    """Executes agent tool calls against the local connection."""

    def __init__(self, connection: LocalConnection, metrics_callback: Callable | None = None):
        self.connection = connection
        self.metrics_callback = metrics_callback

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as a string."""
        handler = getattr(self, f'_tool_{name}', None)
        if handler is None:
            return f'Unknown tool: {name}'
        try:
            result = await handler(**arguments)
            return json.dumps(result, ensure_ascii=False, default=str)
        except Exception as e:
            return f'Error executing {name}: {e}'

    async def _tool_list_training_runs(self) -> list[dict]:
        return self.connection.list_training_runs()

    async def _tool_get_training_status(self, run_id: str) -> dict:
        metrics = self.connection.get_metrics(run_id, last_n=10)
        is_paused = self.connection.is_paused(run_id)
        is_stopped = self.connection.is_stopped(run_id)
        state = 'stopped' if is_stopped else ('paused' if is_paused else 'running')
        return {'run_id': run_id, 'state': state, 'recent_metrics': metrics}

    async def _tool_pause_training(self, run_id: str) -> dict:
        return self.connection.pause_training(run_id)

    async def _tool_resume_training(self, run_id: str) -> dict:
        return self.connection.resume_training(run_id)

    async def _tool_stop_training(self, run_id: str) -> dict:
        return self.connection.stop_training(run_id)

    async def _tool_search_datasets(self, query: str, limit: int = 5) -> list[dict]:
        """Search ModelScope for datasets (requires modelscope SDK)."""
        import asyncio

        def _search():
            from modelscope.hub.api import HubApi
            api = HubApi()
            results = api.list_datasets(query=query, limit=limit)
            return [{'id': d.id, 'name': d.name} for d in results]

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _search)
        except ImportError:
            return [{'error': 'modelscope SDK not installed. Install with: pip install modelscope'}]
        except Exception as e:
            return [{'error': str(e)}]

    async def _tool_search_models(self, query: str, limit: int = 5) -> list[dict]:
        """Search ModelScope for models (requires modelscope SDK)."""
        import asyncio

        def _search():
            from modelscope.hub.api import HubApi
            api = HubApi()
            results = api.list_models(query=query, limit=limit)
            return [{'id': m.id, 'name': m.name} for m in results]

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _search)
        except ImportError:
            return [{'error': 'modelscope SDK not installed. Install with: pip install modelscope'}]
        except Exception as e:
            return [{'error': str(e)}]

    async def _tool_zoom_metrics(
        self,
        action: str,
        x_start: int | None = None,
        x_end: int | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
    ) -> dict:
        """Control the metrics chart zoom level."""
        if self.metrics_callback:
            if action == 'reset':
                self.metrics_callback('reset')
            else:
                self.metrics_callback('zoom', x_start=x_start, x_end=x_end, y_min=y_min, y_max=y_max)
        return {'action': action, 'status': 'applied'}
