"""Verify the freeze-and-append rollout loop's cross-round invariants.

This test exercises the ``_FrozenContext`` machinery from
``short_math_grpo_with_tools.py`` without booting vLLM, the model, or Ray --
it drives the chunker + condenser + frozen cache manually across three
simulated rounds and checks:

  Invariant A: ``<block_N>`` numbering is stable across rounds.
               ``full_chunks[N].content`` at round 1 == round 3.
  Invariant B: previously-compressed chunks are never re-compressed.
               ``compressed_chunks[N]`` byte-identical across rounds.
  Invariant C: ``to_trajectory`` emits balanced ``<block_N>`` / ``</block_N>``
               tag pairs on the accumulated chunks.
  Invariant D: ``_strip_block_tags`` removes echoed ``<block_N>`` wrappers
               and ``[[#N]]`` fake citations before the assistant content
               enters the trajectory (so they do not survive into the next
               freeze as naked-text block tags waiting to be shredded).

Re-defining ``_FrozenContext`` / ``_strip_block_tags`` inline here rather
than importing the cookbook script, because the cookbook module pulls in
heavy ``twinkle.*`` + Ray dependencies at import time that have nothing to
do with the freeze-and-append contract under test.  Any drift between this
mirror and the cookbook will surface as identical-looking but differently
behaving regex/constants, so the cookbook file is the single source of
truth -- update both together.
"""
import re
from typing import Any, Dict, List


# --- Mirror of the helpers under test (keep in lockstep with cookbook) ---
_TOOL_CALL_STRIP_RE = re.compile(r'<tool_call>.*?(?:</tool_call>|\Z)', re.DOTALL)
_BLOCK_TAG_STRIP_RE = re.compile(r'<block_(\d+)>\s*([\s\S]*?)\s*</block_\1>')
_FAKE_CITE_RE = re.compile(r'\[\[#\d+\]\]')


def _strip_tool_call_spans(text: str) -> str:
    return _TOOL_CALL_STRIP_RE.sub('', text or '').rstrip()


def _strip_block_tags(text: str) -> str:
    text = _BLOCK_TAG_STRIP_RE.sub(lambda m: m.group(2), text or '')
    text = _FAKE_CITE_RE.sub('', text)
    return text.rstrip()


class _FrozenContext:
    __slots__ = ('frozen_msg_count', 'full_chunks',
                 'compressed_chunks', 'media_frozen')

    def __init__(self) -> None:
        self.frozen_msg_count: int = 0
        self.full_chunks: List[Dict[str, Any]] = []
        self.compressed_chunks: List[Dict[str, Any]] = []
        self.media_frozen: bool = False


class _Rollout:
    __slots__ = ('trajectory', 'done', 'frozen')

    def __init__(self, prompt_trajectory: Dict[str, Any]) -> None:
        self.trajectory: Dict[str, Any] = {
            'messages': list(prompt_trajectory.get('messages', [])),
            'user_data': prompt_trajectory.get('user_data', []),
        }
        for _k in ('images', 'videos', 'audios'):
            if prompt_trajectory.get(_k):
                self.trajectory[_k] = list(prompt_trajectory[_k])
        self.done = False
        self.frozen = _FrozenContext()


def main() -> None:
    from twinkle_agentic.chunker.native import NativeChunker
    from twinkle_agentic.condenser.native import NativeCondenser
    from twinkle_agentic.data_format.chunk import Chunks

    # --- Test D: _strip_block_tags + _FAKE_CITE_RE ---
    raw = ('Based on <block_11> Milhouse Van Houten is a fictional character '
           'featured in The Simpsons </block_11> and [[#1]][[#2]], '
           'the answer is foo.')
    cleaned = _strip_block_tags(raw)
    assert '<block_11>' not in cleaned and '</block_11>' not in cleaned, cleaned
    assert '[[#1]]' not in cleaned and '[[#2]]' not in cleaned, cleaned
    assert 'Milhouse Van Houten is a fictional character' in cleaned, cleaned
    assert 'the answer is foo.' in cleaned, cleaned
    print('[D] _strip_block_tags OK:', cleaned[:80], '...')

    # --- Build a small rollout with 4 HotpotQA-like passages ---
    # Use any tokenizer available; the exact model does not affect chunking
    # boundaries we care about here (passage_boundary_re is the decisive one).
    MODEL_ID = 'Qwen/Qwen2.5-1.5B-Instruct'
    chunker = NativeChunker(model_id=MODEL_ID, chunk_size=512,
                             chunk_overlap=50,
                             passage_boundary_re=r'^\[\d+\]\s+')
    condenser = NativeCondenser(keep_ratio=(0.2, 0.5), skip_system=True)

    passages = '\n\n'.join([
        '[1] Radio City: Radio City is India\'s first private FM radio '
        'station, started on 3 July 2001. It plays Hindi, English and '
        'regional songs.',
        '[2] History of Albania: Albania is a Balkan country. Its capital '
        'is Tirana. The modern Albanian state was proclaimed in 1912 after '
        'the Balkan Wars.',
        '[3] Echosmith: Echosmith is an American indie pop band formed in '
        'February 2009 in Chino, California. They are best known for their '
        'hit song Cool Kids, which reached number 13 on the Billboard Hot 100.',
        '[4] Milhouse Van Houten: Milhouse Mussolini van Houten is a '
        'fictional character in The Simpsons. He is the best friend of Bart '
        'Simpson and named after Richard Nixon\'s middle name.',
    ])
    prompt = {
        'messages': [
            {'role': 'system', 'content': 'You are a HotpotQA assistant.'},
            {'role': 'user', 'content':
                f'{passages}\n\nQuestion: Who was Milhouse named after?'},
        ],
        'user_data': [],
    }
    r = _Rollout(prompt)

    def freeze_round(r: _Rollout) -> None:
        """Run one round of the freeze-and-append loop.

        Keep this body in sync with ``run_agentic_rollouts``'s per-rollout
        block in ``short_math_grpo_with_tools.py``.
        """
        frozen = r.frozen
        total_msgs = r.trajectory['messages']
        new_msgs = total_msgs[frozen.frozen_msg_count:]
        needs_media_freeze = (not frozen.media_frozen) and any(
            r.trajectory.get(k) for k in ('images', 'videos', 'audios'))
        if not (new_msgs or needs_media_freeze):
            return
        delta_traj: Dict[str, Any] = {
            'messages': list(new_msgs),
            'user_data': r.trajectory.get('user_data', []),
        }
        if needs_media_freeze:
            for k in ('images', 'videos', 'audios'):
                v = r.trajectory.get(k)
                if v:
                    delta_traj[k] = v
            frozen.media_frozen = True
        new_full = chunker.chunk(delta_traj)
        if frozen.frozen_msg_count == 0:
            new_compressed = condenser.condense(new_full)
        else:
            new_compressed = condenser.condense(
                new_full, keep_ratio=condenser.max_keep_ratio)
        frozen.full_chunks.extend(new_full.chunks)
        frozen.compressed_chunks.extend(new_compressed.chunks)
        frozen.frozen_msg_count = len(total_msgs)

    # --- Round 0: initial freeze (full passages) ---
    freeze_round(r)
    snap0_full = [c['content'] for c in r.frozen.full_chunks]
    snap0_compressed = [c['content'] for c in r.frozen.compressed_chunks]
    n0 = len(snap0_full)
    print(f'[R0] freeze produced {n0} chunks')
    assert n0 >= 4, f'Expected >=4 chunks (4 passages + system), got {n0}'

    # --- Round 1: simulate assistant tool call + echoed <block_3> ---
    decoded_r0 = (
        'Let me check. <block_3> Echosmith is an American indie pop band. '
        '</block_3> Hmm, that\'s not the answer. I will call extract_compressed '
        'on block 4.'
        '<tool_call>{"tool_name": "extract_compressed", "arguments": '
        '{"block_index": 4}}</tool_call>'
    )
    content_cleaned = _strip_block_tags(_strip_tool_call_spans(decoded_r0))
    assert '<block_3>' not in content_cleaned
    assert '<tool_call>' not in content_cleaned
    r.trajectory['messages'].append({
        'role': 'assistant',
        'content': content_cleaned,
        'tool_calls': [{'tool_name': 'extract_compressed',
                        'arguments': '{"block_index": 4}'}],
    })
    r.trajectory['messages'].append({
        'role': 'tool',
        'content': '[4] Milhouse Van Houten: Milhouse Mussolini van Houten is '
                   'a fictional character in The Simpsons. He is the best '
                   'friend of Bart Simpson and named after Richard Nixon\'s '
                   'middle name.',
        'tool_call_id': 'call_t0_i0',
    })
    freeze_round(r)
    snap1_full = [c['content'] for c in r.frozen.full_chunks]
    snap1_compressed = [c['content'] for c in r.frozen.compressed_chunks]
    n1 = len(snap1_full)
    print(f'[R1] freeze produced {n1} chunks (delta = {n1 - n0})')
    assert n1 > n0, 'Round 1 should have added chunks'

    # --- Invariant A: block numbering stable (full chunks) ---
    for i in range(n0):
        assert snap1_full[i] == snap0_full[i], (
            f'[A] full_chunks[{i}] changed across rounds:\n'
            f'  R0: {snap0_full[i][:80]!r}\n  R1: {snap1_full[i][:80]!r}')
    print(f'[A] full_chunks stable across R0 -> R1 for all {n0} frozen indices')

    # --- Invariant B: compressed chunks byte-identical ---
    for i in range(n0):
        assert snap1_compressed[i] == snap0_compressed[i], (
            f'[B] compressed_chunks[{i}] was re-compressed across rounds:\n'
            f'  R0: {snap0_compressed[i][:80]!r}\n'
            f'  R1: {snap1_compressed[i][:80]!r}')
    print(f'[B] compressed_chunks byte-identical for all {n0} frozen indices')

    # --- Round 2: terminal answer ---
    decoded_r1 = ('Based on [4], Milhouse was named after [[#1]] '
                  'Richard Nixon. \\boxed{Richard Nixon}')
    r.trajectory['messages'].append(
        {'role': 'assistant', 'content': _strip_block_tags(decoded_r1)})
    r.done = True
    freeze_round(r)
    snap2_full = [c['content'] for c in r.frozen.full_chunks]
    snap2_compressed = [c['content'] for c in r.frozen.compressed_chunks]
    n2 = len(snap2_full)
    print(f'[R2] freeze produced {n2} chunks (delta = {n2 - n1})')

    for i in range(n1):
        assert snap2_full[i] == snap1_full[i], f'[A] drift at R1 -> R2, idx={i}'
        assert snap2_compressed[i] == snap1_compressed[i], (
            f'[B] re-compression at R1 -> R2, idx={i}')
    print(f'[A/B] stability holds across R1 -> R2 for all {n1} frozen indices')

    # --- Invariant C: to_trajectory block markers are balanced ---
    rendered = Chunks(chunks=list(r.frozen.compressed_chunks)).to_trajectory()
    rendered_text = str(rendered)
    condensed_count = sum(
        1 for c in r.frozen.compressed_chunks
        if isinstance(c.get('raw'), dict) and c['raw'].get('condensed'))
    open_tags = set(re.findall(r'<block_(\d+)>', rendered_text))
    close_tags = set(re.findall(r'</block_(\d+)>', rendered_text))
    assert open_tags == close_tags, (
        f'[C] unbalanced block tags: open={open_tags} close={close_tags}')
    print(f'[C] {len(open_tags)} unique <block_N> indices, all balanced '
          f'(condensed chunks: {condensed_count})')

    # --- Final sanity: [[#1]] did not leak into the terminal message ---
    final_msg = r.trajectory['messages'][-1]['content']
    assert '[[#1]]' not in final_msg, f'[D2] fake cite leaked: {final_msg!r}'
    assert '\\boxed{Richard Nixon}' in final_msg
    print('[D2] fake [[#N]] citations stripped from terminal message')

    print('\nAll invariants satisfied. Frozen-and-append rollout is correct.')


if __name__ == '__main__':
    main()
