# Copyright (c) Twinkle Contributors. All rights reserved.
"""Local file-based connection layer for TUI.

Reads metrics/logs from JSONL files written by the training script,
and controls training via file-based signals (pause/stop).

File layout under run_dir (~/.cache/twinkle/{run_id}/):
    metrics.jsonl  — one JSON object per line, written after each step
    logs.jsonl     — one JSON object per line (ts + msg)
    pause          — existence = training paused
    stop           — existence = training should stop
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


# Default base directory for all training run data
DEFAULT_BASE_DIR = Path.home() / '.cache' / 'twinkle'


class LocalConnection:
    """File-based connection between TUI and training process.

    All communication happens through the local filesystem:
    - Metrics and logs are read from JSONL files (tail-style)
    - Pause/stop signals are file-based (create/delete)
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
        """Pause training by creating a signal file."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        pause_file = run_dir / 'pause'
        pause_file.touch()
        return {'status': 'paused', 'run_id': run_id}

    def resume_training(self, run_id: str) -> dict[str, Any]:
        """Resume training by removing the pause signal file."""
        pause_file = self.base_dir / run_id / 'pause'
        if pause_file.exists():
            pause_file.unlink()
        return {'status': 'resumed', 'run_id': run_id}

    def stop_training(self, run_id: str) -> dict[str, Any]:
        """Stop training by creating a stop signal file."""
        run_dir = self.base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        stop_file = run_dir / 'stop'
        stop_file.touch()
        return {'status': 'stopped', 'run_id': run_id}

    def is_paused(self, run_id: str) -> bool:
        """Check if training is currently paused."""
        return (self.base_dir / run_id / 'pause').exists()

    def is_stopped(self, run_id: str) -> bool:
        """Check if training has been signaled to stop."""
        return (self.base_dir / run_id / 'stop').exists()

    def close(self) -> None:
        """No-op for local connection (no resources to release)."""
        pass
