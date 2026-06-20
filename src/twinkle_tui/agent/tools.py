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
            'description': 'Pause training by killing the client process. In Server Mode, the server retains all state (LoRA weights, optimizer, LR scheduler) in GPU memory. Restart the script to continue.',
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
            'description': 'Resume a paused training job by restarting the client script with the same adapter_name. Server state is preserved.',
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
            'description': 'Stop training permanently. Kills the client process and marks the run as stopped. The adapter will be cleaned up after adapter_timeout on the server.',
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
            'name': 'list_supported_models',
            'description': 'Query the Twinkle server for its list of supported base models. Use this to discover which models are available for training before writing scripts.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'base_url': {
                        'type': 'string',
                        'description': 'Server base URL. Defaults to http://localhost:8000. Use http://www.modelscope.cn/twinkle for cloud.',
                    },
                    'api_key': {
                        'type': 'string',
                        'description': 'API key for authentication. Defaults to MODELSCOPE_TOKEN env var or EMPTY_API_KEY.',
                    },
                },
                'required': [],
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
    {
        'type': 'function',
        'function': {
            'name': 'update_script',
            'description': 'Update the training script for a run. Archives the current train.py as train_v{N}.py and writes the new version. Use this after diagnosing a script error to deploy a fixed version, then call resume_training to re-execute.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'run_id': {'type': 'string', 'description': 'Training run ID.'},
                    'script_content': {'type': 'string', 'description': 'Full Python source code of the new training script.'},
                },
                'required': ['run_id', 'script_content'],
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

    async def _tool_update_script(self, run_id: str, script_content: str) -> dict:
        """Update the training script for a run with version archiving."""
        return self.connection.update_script(run_id, script_content)

    async def _tool_list_supported_models(
        self, base_url: str | None = None, api_key: str | None = None
    ) -> dict:
        """Query the Twinkle server for supported models via /get_server_capabilities."""
        import asyncio
        import os

        url = base_url or os.environ.get('TWINKLE_SERVER_URL', 'http://localhost:8000')
        key = api_key or os.environ.get('MODELSCOPE_TOKEN') or os.environ.get('TWINKLE_SERVER_TOKEN') or 'EMPTY_API_KEY'

        def _query():
            from twinkle_client import init_twinkle_client
            client = init_twinkle_client(base_url=url, api_key=key)
            caps = client.get_server_capabilities()
            return {
                'base_url': url,
                'supported_models': [m.model_name for m in caps.supported_models],
            }

        try:
            return await asyncio.get_event_loop().run_in_executor(None, _query)
        except ImportError:
            return {'error': 'twinkle_client not installed. Install with: pip install twinkle-kit'}
        except Exception as e:
            return {'error': f'Failed to query server at {url}: {e}'}

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
