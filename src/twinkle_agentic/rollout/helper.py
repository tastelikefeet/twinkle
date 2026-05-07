import json
import time
from typing import List


def make_dump_rollout_trace(trace_path: str):
    if not trace_path:
        return None

    def _hook(turn, active, displays, responses):
        if not active:
            return
        try:
            records: List[str] = []
            for idx, r in enumerate(active):
                try:
                    resp = responses[idx] if idx < len(responses) else None
                    tcc = sum(
                        len(m.get('tool_calls') or [])
                        for m in r.trajectory.get('messages', [])
                        if m.get('role') == 'assistant')
                    last_decoded = ''
                    if resp and getattr(resp, 'sequences', None):
                        last_decoded = resp.sequences[0].decoded or ''
                    final_answer = _extract_final_answer(
                        _last_assistant_text(r.trajectory)) if r.done else ''
                    record = {
                        'ts': time.time(), 'turn': turn,
                        'group_size': len(active), 'picked_idx': idx,
                        'rollout_id': id(r), 'tool_call_count': tcc,
                        'done': bool(r.done),
                        'compressed': displays[idx] if idx < len(displays) else None,
                        'last_decoded': last_decoded, 'final_answer': final_answer,
                    }
                    records.append(json.dumps(record, ensure_ascii=False, default=str))
                except Exception:
                    pass
            if records:
                with open(trace_path, 'a', encoding='utf-8') as f:
                    f.write('\n'.join(records) + '\n')
        except Exception:
            pass

    return _hook