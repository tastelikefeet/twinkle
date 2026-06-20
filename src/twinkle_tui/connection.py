# Copyright (c) Twinkle Contributors. All rights reserved.
"""Local file-based connection layer for TUI.

Reads metrics/logs from JSONL files written by the training script.

In Server Mode, training control is done by killing/restarting the client
process. The server retains all model/optimizer state in GPU memory.
- "Pause" = kill client process
- "Resume" = start a new client with same adapter_name
- "Modify" = kill → edit script → restart

File layout under run_dir (~/.cache/twinkle/{run_id}/):
    metrics.jsonl  — one JSON object per line, written after each step
    logs.jsonl     — one JSON object per line (ts + msg)
    meta.json      — run metadata (model_id, config, status)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any


# Default base directory for all training run data
DEFAULT_BASE_DIR = Path.home() / '.cache' / 'twinkle'


class LocalConnection:
    """File-based connection between TUI and training process.

    All monitoring happens through the local filesystem:
    - Metrics and logs are read from JSONL files (tail-style)
    - Training control is via process management (kill/restart)
    """

    def __init__(self, base_dir: Path | str | None = None):
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.current_run_id: str | None = None
        self._metrics_offset: int = 0  # file position for incremental read
        self._logs_offset: int = 0

    @property
    def run_dir(self) -> Path | None:
        """Current run's data directory."""
        if self.current_run_id:
            return self.base_dir / self.current_run_id
        return None

    def list_training_runs(self) -> list[dict[str, Any]]:
        """List all training runs by scanning base directory."""
        if not self.base_dir.exists():
            return []
        runs = []
        for entry in sorted(self.base_dir.iterdir(), reverse=True):
            if entry.is_dir() and (entry / 'metrics.jsonl').exists():
                # Read first line to get metadata
                meta = {'run_id': entry.name, 'dir': str(entry)}
                meta_file = entry / 'meta.json'
                if meta_file.exists():
                    try:
                        meta.update(json.loads(meta_file.read_text()))
                    except Exception:
                        pass
                runs.append(meta)
        return runs

    def get_metrics(self, run_id: str, last_n: int = 200) -> list[dict[str, Any]]:
        """Read metrics from JSONL file (tail last_n entries)."""
        metrics_file = self.base_dir / run_id / 'metrics.jsonl'
        if not metrics_file.exists():
            return []
        try:
            lines = metrics_file.read_text().strip().splitlines()
            recent = lines[-last_n:] if len(lines) > last_n else lines
            return [json.loads(line) for line in recent if line.strip()]
        except Exception:
            return []

    def get_new_metrics(self, run_id: str) -> list[dict[str, Any]]:
        """Read only new metrics since last read (incremental)."""
        metrics_file = self.base_dir / run_id / 'metrics.jsonl'
        if not metrics_file.exists():
            return []
        try:
            with open(metrics_file, 'r') as f:
                f.seek(self._metrics_offset)
                new_data = f.read()
                self._metrics_offset = f.tell()
            if not new_data.strip():
                return []
            return [json.loads(line) for line in new_data.strip().splitlines() if line.strip()]
        except Exception:
            return []

    def get_logs(self, run_id: str, since: float = 0, limit: int = 200) -> list[dict[str, Any]]:
        """Read log entries from JSONL file."""
        logs_file = self.base_dir / run_id / 'logs.jsonl'
        if not logs_file.exists():
            return []
        try:
            lines = logs_file.read_text().strip().splitlines()
            entries = [json.loads(line) for line in lines if line.strip()]
            if since > 0:
                entries = [e for e in entries if e.get('ts', 0) > since]
            return entries[-limit:]
        except Exception:
            return []

    def get_new_logs(self, run_id: str) -> list[dict[str, Any]]:
        """Read only new logs since last read (incremental)."""
        logs_file = self.base_dir / run_id / 'logs.jsonl'
        if not logs_file.exists():
            return []
        try:
            with open(logs_file, 'r') as f:
                f.seek(self._logs_offset)
                new_data = f.read()
                self._logs_offset = f.tell()
            if not new_data.strip():
                return []
            return [json.loads(line) for line in new_data.strip().splitlines() if line.strip()]
        except Exception:
            return []

    def pause_training(self, run_id: str) -> dict[str, Any]:
        """Pause training by killing the client process (SIGKILL).

        In Server Mode, the server retains all state. Killing the client
        is a zero-cost "pause" — restart the script to continue.
        """
        meta_file = self.base_dir / run_id / 'meta.json'
        pid = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                pid = meta.get('pid')
                meta['status'] = 'paused'
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        # Kill the client process if PID is known
        if pid:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass  # Already dead

        return {'status': 'paused', 'run_id': run_id, 'action': 'kill_client', 'pid': pid}

    def resume_training(self, run_id: str) -> dict[str, Any]:
        """Resume training by re-executing the stored training script.

        In Server Mode, simply restart the same script — server state
        (LoRA weights, optimizer, LR scheduler) is preserved in GPU memory.
        The script is stored as ``train.py`` inside the run directory.
        """
        meta_file = self.base_dir / run_id / 'meta.json'
        script_path = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                script_path = meta.get('script_path')
                meta['status'] = 'running'
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        # Launch the training script as a background subprocess
        new_pid = None
        if script_path and Path(script_path).exists():
            proc = subprocess.Popen(
                ['python', script_path],
                cwd=str(self.base_dir / run_id),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            new_pid = proc.pid
            # Update PID in meta
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text())
                    meta['pid'] = new_pid
                    meta_file.write_text(json.dumps(meta, indent=2))
                except Exception:
                    pass

        return {
            'status': 'resumed',
            'run_id': run_id,
            'action': 'restart_client',
            'script_path': script_path,
            'pid': new_pid,
        }

    def stop_training(self, run_id: str) -> dict[str, Any]:
        """Stop training gracefully via SIGTERM.

        Sends SIGTERM to the client process, which triggers the training script's
        graceful shutdown handler to save a checkpoint (model + dataloader state)
        before exiting. This ensures training can be resumed from the exact point
        using model.resume_from_checkpoint() + dataloader.resume_from_checkpoint().

        The adapter will be cleaned up after adapter_timeout on the server.
        """
        meta_file = self.base_dir / run_id / 'meta.json'
        pid = None
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                pid = meta.get('pid')
                meta['status'] = 'stopping'
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        # Send SIGTERM for graceful shutdown
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass  # Already dead

        return {'status': 'stopping', 'run_id': run_id, 'action': 'sigterm_client', 'pid': pid}

    def is_paused(self, run_id: str) -> bool:
        """Check if training is currently paused (client not running)."""
        meta_file = self.base_dir / run_id / 'meta.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                return meta.get('status') == 'paused'
            except Exception:
                pass
        return False

    def is_stopped(self, run_id: str) -> bool:
        """Check if training has been permanently stopped."""
        meta_file = self.base_dir / run_id / 'meta.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                return meta.get('status') in ('stopped', 'completed', 'error')
            except Exception:
                pass
        return False

    def close(self) -> None:
        """No-op for local connection (no resources to release)."""
        pass

    def update_script(self, run_id: str, new_script_content: str) -> dict[str, Any]:
        """Update the training script for a run (version management).

        Archives the current ``train.py`` as ``train_v{N}.py`` and writes the
        new content as the active ``train.py``. Updates ``script_version`` in
        meta.json.

        This is used when the LLM Agent rewrites a script to fix errors or
        adjust hyperparameters, while keeping the same run_id (same adapter
        state on the server).

        Args:
            run_id: The training run ID.
            new_script_content: Full Python source code of the new script.

        Returns:
            Dict with version info and file paths.
        """
        import shutil

        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        train_py = run_dir / 'train.py'
        meta_file = run_dir / 'meta.json'

        # Archive existing train.py
        version = 1
        if train_py.exists():
            existing_versions = list(run_dir.glob('train_v*.py'))
            version = len(existing_versions) + 2
            archive_name = f'train_v{version - 1}.py'
            shutil.copy2(train_py, run_dir / archive_name)

        # Write new script
        train_py.write_text(new_script_content)

        # Update meta.json
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta['script_version'] = version
                meta['script_path'] = str(train_py)
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        return {
            'run_id': run_id,
            'script_version': version,
            'script_path': str(train_py),
            'archived_versions': [f.name for f in sorted(run_dir.glob('train_v*.py'))],
        }
