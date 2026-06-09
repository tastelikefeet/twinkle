import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from twinkle.preprocessor import Preprocessor
from twinkle.utils.parallel import PosixFileLock

_SYSTEM_INJECTION_RE = re.compile(r'^<(?:system-reminder|system_reminder|context|user_info|attached_files)[ >]',
                                  re.IGNORECASE)


def _is_injected_user(content: str) -> bool:
    """True if user message is system-injected metadata rather than real user input."""
    return bool(_SYSTEM_INJECTION_RE.match(content.strip()))


def _head_tail(text: str, n: int = 100) -> str:
    """First n chars + last n chars for stronger fingerprint."""
    if len(text) <= n * 2:
        return text
    return text[:n] + text[-n:]


def _conversation_sig(row: Dict[str, Any], prefix_chars: int = 100, asst_chars: int = 100) -> str:
    """Hash of first real user (head+tail) + following assistant (head+tail).

    Skips system messages and system-injected user messages (e.g. <system-reminder>)
    to avoid template-dominated signatures in agent frameworks.
    Fallback: if no real user found, use first two non-empty assistant contents.
    """
    msgs = row.get('messages') or []
    user_text = ''
    asst_text = ''
    for m in msgs:
        role = m.get('role', '')
        content = m.get('content') or ''
        if role == 'user' and not user_text:
            if _is_injected_user(content):
                continue
            user_text = _head_tail(content, prefix_chars)
        elif role == 'assistant' and user_text and not asst_text:
            asst_text = _head_tail(content, asst_chars)
            break
    # Fallback for agent-autonomous convos where all user msgs are injected
    if not user_text:
        asst_parts = []
        for m in msgs:
            if m.get('role') == 'assistant':
                c = (m.get('content') or '').strip()
                if c:
                    asst_parts.append(_head_tail(c, prefix_chars))
                    if len(asst_parts) >= 2:
                        break
        raw = '||'.join(asst_parts) if asst_parts else json.dumps(msgs[:3], ensure_ascii=False)[:400]
    else:
        raw = f'{user_text}||{asst_text}'
    return hashlib.md5(raw.encode()).hexdigest()


class DedupFilter(Preprocessor):
    """Conversation-level near-dedup: keep at most max_per_sig rows per signature.

    Uses file-backed state + PosixFileLock for multi-process safety.
    """

    def __init__(self,
                 max_per_sig: int = 1,
                 prefix_chars: int = 100,
                 asst_chars: int = 100,
                 state_file: Optional[str] = None):
        self._max = max_per_sig
        self._prefix = prefix_chars
        self._asst_chars = asst_chars
        # Deterministic path derived from init params — keeps HF Datasets fingerprint stable across runs
        sig = hashlib.md5(f'{max_per_sig}_{prefix_chars}_{asst_chars}'.encode()).hexdigest()[:12]
        self._state_file = state_file or f'/tmp/dedup_filter_{sig}.json'
        self._lock_file = self._state_file + '.lock'
        # Wipe stale state from prior runs (deterministic path means no auto-isolation)
        for p in (self._state_file, self._lock_file):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

    def __call__(self, rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        rows = self.map_col_to_row(rows)
        with PosixFileLock(self._lock_file):
            # state: {sig: {"n": msg_count, "fh": full_hash}}
            seen: Dict[str, Dict[str, Any]] = {}
            if os.path.exists(self._state_file):
                with open(self._state_file) as f:
                    seen = json.load(f)

            out: List[Dict[str, Any]] = []
            dropped: List[Dict[str, Any]] = []
            batch_idx: Dict[str, int] = {}

            for r in rows:
                sig = _conversation_sig(r, self._prefix, self._asst_chars)
                msgs = r.get('messages') or []
                n = len(msgs)
                fh = hashlib.md5(json.dumps(msgs, ensure_ascii=False).encode()).hexdigest()

                prev = seen.get(sig)
                if prev is None:
                    seen[sig] = {'n': n, 'fh': fh}
                    batch_idx[sig] = len(out)
                    out.append(r)
                elif fh == prev['fh']:
                    dropped.append(dict(r, drop_reason='duplicate'))
                elif n > prev['n']:
                    # Longer version wins
                    seen[sig] = {'n': n, 'fh': fh}
                    if sig in batch_idx:
                        old_idx = batch_idx[sig]
                        dropped.append(dict(out[old_idx], drop_reason='duplicate'))
                        out[old_idx] = r
                    else:
                        batch_idx[sig] = len(out)
                        out.append(r)
                else:
                    dropped.append(dict(r, drop_reason='duplicate'))

            with open(self._state_file, 'w') as f:
                json.dump(seen, f)
        return out, dropped
