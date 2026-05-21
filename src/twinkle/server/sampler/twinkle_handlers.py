# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native sampler handler mixin.

Provides /twinkle/* sampler endpoints that call the sampler directly (no queue needed).
"""
from __future__ import annotations

import json
import time
import traceback
import uuid
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

from twinkle_client.common.serialize import deserialize_object

if TYPE_CHECKING:
    from .app import SamplerManagement

import numpy as np

import twinkle_client.types as types
from twinkle.data_format import InputFeature, SamplingParams, Trajectory
from twinkle.utils.logger import get_logger

logger = get_logger()


def _serialize_input_feature(feature: dict) -> dict:
    """Convert numpy arrays / torch tensors in an InputFeature to plain Python lists."""
    result = {}
    for k, v in feature.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    result[k] = v.tolist()
                    continue
            except ImportError:
                pass
            result[k] = v
    return result


def _get_twinkle_sampler_adapter_name(request: Request, adapter_name: str | None) -> str | None:
    """Prefix the adapter name with the request ID for per-request isolation."""
    if adapter_name is None or adapter_name == '':
        return None
    return request.state.request_id + '-' + adapter_name


def _openai_body_to_trajectory_and_params(
        body: Dict[str, Any]) -> Tuple[Trajectory, SamplingParams]:
    """Map an OpenAI ``/v1/chat/completions`` body to (Trajectory, SamplingParams).

    Trajectory.messages / .tools are already OpenAI-shaped TypedDicts, so they
    pass through verbatim — no field renaming needed.
    """
    messages = body.get('messages')
    if not messages:
        raise HTTPException(status_code=400, detail='messages is required')
    trajectory: Trajectory = {'messages': list(messages)}
    if body.get('tools'):
        trajectory['tools'] = list(body['tools'])

    sp_kwargs: Dict[str, Any] = {}
    if body.get('temperature') is not None:
        sp_kwargs['temperature'] = float(body['temperature'])
    if body.get('top_p') is not None:
        sp_kwargs['top_p'] = float(body['top_p'])
    # max_completion_tokens supersedes max_tokens per the newer OpenAI spec
    if body.get('max_completion_tokens') is not None:
        sp_kwargs['max_tokens'] = int(body['max_completion_tokens'])
    elif body.get('max_tokens') is not None:
        sp_kwargs['max_tokens'] = int(body['max_tokens'])
    if body.get('seed') is not None:
        sp_kwargs['seed'] = int(body['seed'])
    if body.get('n') is not None:
        sp_kwargs['num_samples'] = int(body['n'])
    if body.get('stop'):
        sp_kwargs['stop'] = body['stop']
    if body.get('logprobs'):
        sp_kwargs['logprobs'] = int(body.get('top_logprobs') or 0)
    fp = body.get('frequency_penalty')
    if fp is not None and fp != 0:
        # OpenAI frequency_penalty (-2..2, 0 == no penalty) -> repetition_penalty
        sp_kwargs['repetition_penalty'] = 1.0 + float(fp)
    return trajectory, SamplingParams(**sp_kwargs)


def _format_openai_choice(seq: Any, idx: int, template: Any) -> Dict[str, Any]:
    """Build one ``choices[]`` entry from a SampledSequence."""
    decoded = seq.decoded or ''
    tool_calls: List[Dict[str, Any]] = []
    if template is not None:
        try:
            parsed = template.parse_tool_call(decoded)
        except Exception:
            parsed = []
        for j, tc in enumerate(parsed or []):
            fn = dict(tc.get('function') or {})
            args = fn.get('arguments')
            # OpenAI wire format demands arguments as a JSON string, not a dict
            if isinstance(args, dict):
                fn['arguments'] = json.dumps(args, ensure_ascii=False)
            tool_calls.append({
                'id': tc.get('id') or f'call_{idx}_{j}',
                'type': tc.get('type') or 'function',
                'function': fn,
            })
        if tool_calls:
            try:
                decoded = template.clean_tool_call(decoded)
            except Exception:
                pass

    finish_reason = 'length' if seq.stop_reason == 'length' else (
        'tool_calls' if tool_calls else 'stop')
    message: Dict[str, Any] = {'role': 'assistant', 'content': decoded}
    if tool_calls:
        message['tool_calls'] = tool_calls
    return {'index': idx, 'message': message, 'finish_reason': finish_reason}


def _build_openai_completion(
        response: Any, model_id: str, template: Any) -> Dict[str, Any]:
    """Wrap a SampleResponse as an OpenAI ChatCompletion object."""
    choices = [
        _format_openai_choice(seq, i, template)
        for i, seq in enumerate(response.sequences)
    ]
    completion_tokens = sum(len(seq.tokens) for seq in response.sequences)
    return {
        'id': f'chatcmpl-{uuid.uuid4().hex}',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': model_id,
        'choices': choices,
        'usage': {
            'prompt_tokens': 0,
            'completion_tokens': completion_tokens,
            'total_tokens': completion_tokens,
        },
    }


def _build_openai_chunk(
        delta_event: Dict[str, Any], completion_id: str, created: int,
        model_id: str) -> Dict[str, Any]:
    """Wrap a sampler delta dict as an OpenAI ``chat.completion.chunk`` object.

    ``delta_event`` is one item yielded by ``Sampler.astream_one``, with keys
    ``index``, ``delta``, ``finish_reason``.
    """
    return {
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model_id,
        'choices': [{
            'index': delta_event.get('index', 0),
            'delta': delta_event.get('delta') or {},
            'finish_reason': delta_event.get('finish_reason'),
        }],
    }


def _register_twinkle_sampler_routes(app: FastAPI, self_fn: Callable[[], SamplerManagement]) -> None:
    """Register all /twinkle/* sampler routes on the given FastAPI app.

    self_fn is a zero-argument callable returning the current SamplerManagement replica instance.
    It is wired in via Depends so it is resolved lazily at request time.
    """

    async def run_task(coro):
        """Await a schedule_task_and_wait coroutine and surface any exception as a
        structured HTTP 500 response so the client receives the full traceback instead
        of an opaque connection-level error.

        Note: HTTPException is re-raised directly to preserve its status code and detail.
        """
        try:
            return await coro
        except HTTPException:
            raise
        except Exception:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post('/twinkle/create', response_model=types.CreateResponse)
    async def create(request: Request, self: SamplerManagement = Depends(self_fn)) -> types.CreateResponse:
        """Health check / session creation endpoint."""
        return types.CreateResponse()

    @app.post('/twinkle/sample', response_model=types.SampleResponseModelList)
    async def sample(
        request: Request, body: types.SampleRequest,
        self: SamplerManagement = Depends(self_fn)) -> types.SampleResponseModelList:
        """Sample completions from the model.

        Supports Trajectory or InputFeature inputs, with optional LoRA adapter.
        """
        token = await self._on_request_start(request)

        async def _task():
            # Resolve adapter
            adapter_path = None
            adapter_name = body.adapter_name or ''
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''

            if body.adapter_uri:
                from twinkle.server.common.checkpoint_factory import create_checkpoint_manager
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, adapter_path = checkpoint_manager.parse_adapter_uri(body.adapter_uri)
                # Reset prefix cache only when new weights are loaded
                self.sampler.reset_prefix_cache()

            # Parse inputs
            inputs = body.inputs
            if isinstance(inputs, list) and inputs:
                first = inputs[0]
                if isinstance(first, dict) and 'input_ids' in first:
                    inputs = [InputFeature(**item) for item in inputs]
                else:
                    inputs = [Trajectory(**item) for item in inputs]
            elif isinstance(inputs, dict):
                if 'input_ids' in inputs:
                    inputs = [InputFeature(**inputs)]
                else:
                    inputs = [Trajectory(**inputs)]

            # Build sampling params
            params = None
            if body.sampling_params:
                params = SamplingParams.from_dict(body.sampling_params)

            # Sample
            responses = self.sampler.sample(
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )

            sample_models = []
            for response in responses:
                sequences = [
                    types.SampledSequenceModel(
                        stop_reason=seq.stop_reason,
                        tokens=list(seq.tokens),
                        logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
                        decoded=seq.decoded,
                        new_input_feature=_serialize_input_feature(seq.new_input_feature)
                        if seq.new_input_feature is not None else None,
                    ) for seq in response.sequences
                ]
                sample_models.append(
                    types.SampleResponseModel(
                        sequences=sequences,
                        prompt_logprobs=response.prompt_logprobs,
                        topk_prompt_logprobs=response.topk_prompt_logprobs,
                    ))
            return types.SampleResponseModelList(samples=sample_models)

        # Calculate metrics for queue scheduling
        inputs_list = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                token=token,
                input_tokens=input_tokens,
                task_type='sample',
            ))

    @app.post('/v1/chat/completions')
    async def chat_completions(
            request: Request,
            body: Dict[str, Any],
            self: SamplerManagement = Depends(self_fn),
    ):
        """OpenAI-compatible chat completions endpoint.

        Accepts the standard ``/v1/chat/completions`` body (messages, tools,
        temperature, top_p, max_tokens, n, seed, stop, frequency_penalty,
        logprobs/top_logprobs, ...) and returns an OpenAI ``chat.completion``
        response. Twinkle-specific extensions: ``adapter_name`` and
        ``adapter_uri`` for LoRA inference. When ``stream=true`` is set the
        response is an SSE stream of ``chat.completion.chunk`` objects.
        """
        # Flatten extra_body so Twinkle extras (adapter_name/adapter_uri/...) are
        # accessible regardless of whether the OpenAI SDK already inlined them.
        extra = body.pop('extra_body', None)
        if isinstance(extra, dict):
            for k, v in extra.items():
                body.setdefault(k, v)

        token = await self._on_request_start(request)

        # Resolve adapter (shared by stream / non-stream paths)
        async def _resolve_adapter() -> Tuple[str, Any]:
            adapter_path = None
            adapter_name = body.get('adapter_name') or ''
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''
            adapter_uri = body.get('adapter_uri')
            if adapter_uri:
                from twinkle.server.common.checkpoint_factory import create_checkpoint_manager
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, adapter_path = checkpoint_manager.parse_adapter_uri(adapter_uri)
                self.sampler.reset_prefix_cache()
            return full_adapter_name, adapter_path

        if body.get('stream'):
            # Streaming path: bypass the GPU serial queue entirely. Each request
            # opens a single async generator on a balanced DP actor and pipes
            # chat.completion.chunk events back as SSE.
            full_adapter_name, adapter_path = await _resolve_adapter()
            trajectory, params = _openai_body_to_trajectory_and_params(body)
            model_id = body.get('model') or getattr(self, 'model_id', '') or ''
            completion_id = f'chatcmpl-{uuid.uuid4().hex}'
            created = int(time.time())

            async def _sse_generator():
                try:
                    async for event in self.sampler.astream_one(
                            trajectory,
                            params,
                            adapter_name=full_adapter_name,
                            adapter_path=adapter_path,
                    ):
                        chunk = _build_openai_chunk(event, completion_id, created, model_id)
                        yield f'data: {json.dumps(chunk, ensure_ascii=False)}\n\n'
                    yield 'data: [DONE]\n\n'
                except HTTPException:
                    raise
                except Exception:
                    err_tb = traceback.format_exc()
                    logger.error(err_tb)
                    err_chunk = {
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model_id,
                        'error': {'message': err_tb, 'type': 'internal_error'},
                    }
                    yield f'data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n'
                    yield 'data: [DONE]\n\n'

            return StreamingResponse(
                _sse_generator(),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                },
            )

        async def _task():
            full_adapter_name, adapter_path = await _resolve_adapter()
            trajectory, params = _openai_body_to_trajectory_and_params(body)

            responses = self.sampler.sample(
                [trajectory],
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )

            return _build_openai_completion(
                responses[0],
                model_id=body.get('model') or getattr(self, 'model_id', '') or '',
                template=getattr(self.sampler, 'template', None),
            )

        # Rough char-based estimate for queue scheduling; trajectory tokens are unknown pre-encode
        rough_tokens = sum(
            len(m.get('content') or '') if isinstance(m.get('content'), str) else 0
            for m in (body.get('messages') or [])
        ) // 4
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                token=token,
                input_tokens=rough_tokens,
                task_type='sample',
            ))

    @app.post('/twinkle/set_template', response_model=types.SetTemplateResponse)
    async def set_template(
            request: Request,
            body: types.SetTemplateRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.SetTemplateResponse:
        """Set the chat template for encoding Trajectory inputs."""
        extra_kwargs = body.model_extra or {}
        self.sampler.set_template(body.template_cls, **extra_kwargs)
        return types.SetTemplateResponse()

    @app.post('/twinkle/add_adapter_to_sampler', response_model=types.AddAdapterResponse)
    async def add_adapter_to_sampler(
            request: Request,
            body: types.AddAdapterRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.AddAdapterResponse:
        """Add a LoRA adapter to the sampler."""
        assert body.adapter_name, 'You need to specify a valid `adapter_name`'
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)

        from peft import LoraConfig
        config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

        self.sampler.add_adapter_to_sampler(full_adapter_name, config)

        return types.AddAdapterResponse(adapter_name=full_adapter_name)

    @app.post('/twinkle/apply_patch')
    async def apply_patch(
            request: Request,
            body: types.ApplyPatchRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> None:
        extra_kwargs = body.model_extra or {}
        patch_cls = deserialize_object(body.patch_cls)
        self.sampler.apply_patch(patch_cls, **extra_kwargs)
