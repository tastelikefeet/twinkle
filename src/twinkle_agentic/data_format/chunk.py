import sys
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

_MULTIMODAL_TYPES = ('image', 'video', 'audio')
_MEDIA_BUCKETS = (('images', 'image'), ('videos', 'video'), ('audios', 'audio'))


class Chunk(TypedDict, total=False):

    type: Literal['text', 'image', 'video', 'audio']
    content: Union[str, Any]
    raw: Union[str, Any]
    role: str


@dataclass
class Chunks:

    chunks: List[Chunk]

    def to_trajectory(
        self,
        block_wrapper: Optional[Tuple[str, str]] = ('<block_{n}>', '</block_{n}>'),
    ) -> Dict[str, Any]:
        """Reassemble chunks back into a :class:`Trajectory`-shaped dict.

        This is the inverse of :class:`NativeChunker.chunk`.  The algorithm
        walks the chunks in order and merges **consecutive chunks sharing the
        same role** into a single :class:`Message`, then dispatches each chunk
        inside a group based on its ``raw['kind']`` (for text chunks) or its
        ``type`` (for multi-modal chunks):

        * ``raw.kind == 'reasoning_content'`` -> ``message['reasoning_content']``
          (multiple such chunks inside one group are joined by blank lines).
        * ``raw.kind == 'content'`` -> ``message['content']`` as plain text,
          or as an OpenAI-style list when mixed with multi-modal parts.
        * ``raw.kind == 'tool_call'`` -> appended to ``message['tool_calls']``;
          the original :class:`ToolCall` dict stored in ``raw['tool_call']``
          is restored verbatim.
        * ``type in {'image', 'video', 'audio'}``:
            - ``raw`` is a ``dict`` (structured-content origin) -> embedded
              in the message's list-form ``content`` in its original position.
            - ``raw`` is a scalar (trajectory-level origin) -> routed to the
              trajectory-level ``images`` / ``videos`` / ``audios`` buckets.

        Order is preserved throughout: chunks keep their emission order inside
        each group, so the structured ``content`` list (e.g. image-then-text)
        matches the original message layout.

        Args:
            block_wrapper: Optional ``(prefix, suffix)`` format-string pair
                wrapping each text chunk's ``content`` with block markers;
                ``{n}`` is substituted with the chunk's 0-based index in
                ``self.chunks``.  Pass ``None`` to disable wrapping.  Only
                **condensed** text chunks (those carrying
                ``raw['condensed'] = True``) are wrapped -- unchanged text
                chunks, multi-modal URLs / paths and tool-call structural
                payloads are left untouched, since ``<block_N>`` markers
                only make sense for text whose original form can be recalled
                via :class:`~twinkle_agentic.tools.extract.ExtractCompressed`.
                Defaults to ``('<block_{n}>', '</block_{n}>')``.

        Returns:
            A dict with keys ``messages`` (always present) and optionally
            ``images`` / ``videos`` / ``audios`` for trajectory-level media.

        Example:
            >>> chunks = Chunks(chunks=[
            ...     {'role': 'user', 'type': 'image',
            ...      'content': '/tmp/plot.png',
            ...      'raw': {'type': 'image', 'image': '/tmp/plot.png'}},
            ...     {'role': 'user', 'type': 'text',
            ...      'content': 'Sort the list.',
            ...      'raw': {'kind': 'content', 'text': 'Sort the list.'}},
            ...     {'role': 'assistant', 'type': 'text',
            ...      'content': 'I will call python.',
            ...      'raw': {'kind': 'reasoning_content', 'text': 'I will call python.'}},
            ...     {'role': 'assistant', 'type': 'text',
            ...      'content': '[tool_call:python]\n{}',
            ...      'raw': {'kind': 'tool_call',
            ...              'tool_call': {'tool_name': 'python', 'arguments': '{}'}}},
            ... ])
            >>> chunks.to_trajectory()  # default block markers are applied
            {'messages': [
                {'role': 'user',
                 'content': [{'type': 'image', 'image': '/tmp/plot.png'},
                             {'type': 'text', 'text': '<block_1>Sort the list.</block_1>'}]},
                {'role': 'assistant',
                 'reasoning_content': '<block_2>I will call python.</block_2>',
                 'tool_calls': [{'tool_name': 'python', 'arguments': '{}'}]},
            ]}
            >>> chunks.to_trajectory(block_wrapper=None)  # raw round-trip
            {'messages': [
                {'role': 'user',
                 'content': [{'type': 'image', 'image': '/tmp/plot.png'},
                             {'type': 'text', 'text': 'Sort the list.'}]},
                {'role': 'assistant',
                 'reasoning_content': 'I will call python.',
                 'tool_calls': [{'tool_name': 'python', 'arguments': '{}'}]},
            ]}
        """
        media: Dict[str, List[Any]] = {t: [] for t in _MULTIMODAL_TYPES}
        bound: List[Chunk] = []

        # Route trajectory-level media aside; wrap text content with block markers.
        # Block numbers are a **1-based sequential counter over wrapped chunks
        # only** (i.e. condensed text chunks, role != 'tool'). This gives the
        # model a compact, predictable index space (1, 2, 3, ...) that is
        # independent of how many non-wrapped chunks (system, question,
        # short-text, tool responses) appear in between. The mapping from
        # displayed number to the 0-based position in ``self.chunks`` is
        # what :class:`~twinkle_agentic.tools.extract.ExtractCompressed`
        # needs to recall the original passage -- use
        # :meth:`displayed_block_mapping` to build that lookup.
        wrap_counter = 0
        for c in self.chunks:
            if c.get('type') in _MULTIMODAL_TYPES and not isinstance(c.get('raw'), dict):
                media[c['type']].append(c.get('content'))
                continue
            if block_wrapper and c.get('type') == 'text':
                # Only wrap chunks that were actually shortened by the
                # condenser (``raw['condensed'] = True``). Unchanged chunks
                # need no ``<block_N>`` marker because there is no original
                # text to recall via ``ExtractCompressed`` -- wrapping them
                # would invite wasted tool calls against pristine content.
                #
                # ``role='tool'`` chunks are never wrapped, regardless of
                # condensation: a tool message typically IS the response
                # produced by ``ExtractCompressed`` and already carries its
                # own ``<block_N>...</block_N>`` markers for the passages
                # it recalled.
                raw = c.get('raw')
                is_condensed = isinstance(raw, dict) and raw.get('condensed')
                content = c.get('content')
                if (is_condensed and isinstance(content, str) and content
                        and c.get('role') != 'tool'):
                    wrap_counter += 1
                    prefix = block_wrapper[0].format(n=wrap_counter)
                    suffix = block_wrapper[1].format(n=wrap_counter)
                    c = {**c, 'content': f'{prefix}{content}{suffix}'}
            bound.append(c)

        # Merge consecutive same-role chunks into one message via groupby.
        messages = [
            self._group_to_message(role, list(grp))
            for role, grp in groupby(bound, key=lambda c: c.get('role') or 'user')
        ]

        trajectory: Dict[str, Any] = {'messages': messages}
        for plural, singular in _MEDIA_BUCKETS:
            if media[singular]:
                trajectory[plural] = media[singular]
        return trajectory

    def displayed_block_mapping(self) -> Dict[int, int]:
        """Return ``{displayed_number -> full_chunk_idx}`` for wrapped chunks.

        ``displayed_number`` is the 1-based counter used by
        :meth:`to_trajectory` when wrapping condensed text chunks in
        ``<block_N>...</block_N>``. ``full_chunk_idx`` is the 0-based index
        into ``self.chunks``. A tool like
        :class:`~twinkle_agentic.tools.extract.ExtractCompressed` can use
        this mapping to translate a block number the model emitted (1, 2,
        ...) back into the chunk that holds the original passage.

        The wrap condition is kept exactly in sync with :meth:`to_trajectory`:
        ``type == 'text'`` AND ``role != 'tool'`` AND ``raw['condensed']``
        truthy AND ``content`` is a non-empty string.
        """
        mapping: Dict[int, int] = {}
        counter = 0
        for idx, c in enumerate(self.chunks):
            if c.get('type') != 'text':
                continue
            if c.get('role') == 'tool':
                continue
            raw = c.get('raw')
            if not (isinstance(raw, dict) and raw.get('condensed')):
                continue
            content = c.get('content')
            if not (isinstance(content, str) and content):
                continue
            counter += 1
            mapping[counter] = idx
        return mapping

    @staticmethod
    def _group_to_message(role: str, group: List[Chunk]) -> Dict[str, Any]:
        """Fold a same-role run of chunks into one :class:`Message`.

        Preserves the intra-group order so mixed text / image / video / audio
        parts round-trip back into OpenAI-style structured ``content``.
        """
        reasoning: List[str] = []
        parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_call_id: Optional[str] = None
        has_media = False

        for c in group:
            t, raw, content = c.get('type'), c.get('raw'), c.get('content')
            kind = raw.get('kind') if isinstance(raw, dict) else None
            # Any chunk in the group may carry the shared ``tool_call_id``.
            if isinstance(raw, dict) and raw.get('tool_call_id') and tool_call_id is None:
                tool_call_id = raw['tool_call_id']

            if t == 'text' and kind == 'reasoning_content' and content:
                reasoning.append(content)
            elif t == 'text' and kind == 'tool_call' and isinstance(raw.get('tool_call'), dict):
                tool_calls.append(dict(raw['tool_call']))
            elif t == 'text' and content:
                parts.append({'type': 'text', 'text': content})
            elif t in _MULTIMODAL_TYPES and isinstance(raw, dict):
                has_media = True
                # Drop condenser-only markers, keep the original part shape.
                parts.append({k: v for k, v in raw.items() if k != 'condensed'}
                             or {'type': t, t: content})

        msg: Dict[str, Any] = {'role': role}
        if reasoning:
            msg['reasoning_content'] = '\n\n'.join(reasoning)
        if parts:
            msg['content'] = parts if has_media else '\n\n'.join(p['text'] for p in parts)
        if tool_calls:
            msg['tool_calls'] = tool_calls
        if tool_call_id is not None:
            msg['tool_call_id'] = tool_call_id
        return msg
