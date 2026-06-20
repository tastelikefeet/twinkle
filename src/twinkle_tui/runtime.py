# Copyright (c) Twinkle Contributors. All rights reserved.
"""Training runtime utilities for TUI integration.

This module provides helpers that training scripts import to:
1. Write structured metrics and logs to local JSONL files
2. Check for pause/stop signals from TUI
3. Manage run lifecycle (start/end)

Usage in training scripts:
    from twinkle_tui.runtime import TrainingRuntime

    rt = TrainingRuntime(run_id='my-grpo-run')
    rt.start(model_id='Qwen/Qwen3.5-4B', config={...})

    for step, batch in enumerate(dataloader):
        rt.check_signals()  # blocks if paused, raises if stopped

        # ... training logic ...
        rt.log_metrics(step=step, loss=loss, reward=reward, grad_norm=gn, lr=lr)
        rt.log('Completed step {step}, loss={loss:.4f}')

    rt.finish()
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_BASE_DIR = Path.home() / '.cache' / 'twinkle'


class TrainingStoppedError(Exception):
    """Raised when a stop signal is detected."""
    pass


class TrainingRuntime:
    """Runtime helper for training scripts to integrate with TUI.

    Manages:
    - Writing metrics.jsonl (structured step data)
    - Writing logs.jsonl (timestamped log messages)
    - Checking pause/stop signal files
    - Run metadata (meta.json)
    """

    def __init__(self, run_id: str, base_dir: Path | str | None = None, poll_interval: float = 1.0):
        """Initialize the training runtime.

        Args:
            run_id: Unique identifier for this training run.
            base_dir: Base directory for run data. Defaults to ~/.cache/twinkle/
            poll_interval: Seconds between checks when paused.
        """
        self.run_id = run_id
        self.base_dir = Path(base_dir) if base_dir else DEFAULT_BASE_DIR
        self.run_dir = self.base_dir / run_id
        self.poll_interval = poll_interval

        self._metrics_file: Any = None
        self._logs_file: Any = None
        self._started = False

    def start(self, model_id: str = '', config: dict[str, Any] | None = None) -> None:
        """Initialize the run directory and write metadata.

        Call this once at the beginning of training.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Write run metadata
        meta = {
            'run_id': self.run_id,
            'model_id': model_id,
            'config': config or {},
            'start_time': time.time(),
            'status': 'running',
        }
        (self.run_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

        # Open files for append
        self._metrics_file = open(self.run_dir / 'metrics.jsonl', 'a', buffering=1)
        self._logs_file = open(self.run_dir / 'logs.jsonl', 'a', buffering=1)

        # Clear any stale signals
        for sig in ('pause', 'stop'):
            sig_file = self.run_dir / sig
            if sig_file.exists():
                sig_file.unlink()

        self._started = True
        self.log('Training started')

    def log_metrics(self, **kwargs) -> None:
        """Write a metrics entry to metrics.jsonl.

        All keyword arguments are written as a single JSON line.
        A timestamp is automatically added.

        Example:
            rt.log_metrics(step=10, loss=0.5, reward=1.2, grad_norm=0.8, lr=1e-5)
        """
        if not self._metrics_file:
            return
        entry = {'ts': time.time(), **kwargs}
        self._metrics_file.write(json.dumps(entry, default=str) + '\n')

    def log(self, message: str) -> None:
        """Write a log message to logs.jsonl.

        Args:
            message: Human-readable log message.
        """
        if not self._logs_file:
            return
        entry = {'ts': time.time(), 'msg': message}
        self._logs_file.write(json.dumps(entry) + '\n')

    def check_signals(self) -> None:
        """Check for pause/stop signals. Call between training steps.

        - If stop signal exists: raises TrainingStoppedError
        - If pause signal exists: blocks until signal is removed

        Raises:
            TrainingStoppedError: When stop signal is detected.
        """
        # Check stop first
        if (self.run_dir / 'stop').exists():
            self.log('Stop signal received, terminating training')
            raise TrainingStoppedError('Stop signal received')

        # Check pause
        if (self.run_dir / 'pause').exists():
            self.log('Pause signal received, waiting for resume...')
            while (self.run_dir / 'pause').exists():
                # Also check stop while paused
                if (self.run_dir / 'stop').exists():
                    self.log('Stop signal received while paused')
                    raise TrainingStoppedError('Stop signal received while paused')
                time.sleep(self.poll_interval)
            self.log('Resumed')

    def finish(self, status: str = 'completed') -> None:
        """Mark training as finished and close files.

        Args:
            status: Final status ('completed', 'stopped', 'error').
        """
        self.log(f'Training finished with status: {status}')

        # Update metadata
        meta_file = self.run_dir / 'meta.json'
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text())
                meta['status'] = status
                meta['end_time'] = time.time()
                meta_file.write_text(json.dumps(meta, indent=2))
            except Exception:
                pass

        # Close files
        if self._metrics_file:
            self._metrics_file.close()
            self._metrics_file = None
        if self._logs_file:
            self._logs_file.close()
            self._logs_file = None

        self._started = False

    @property
    def is_paused(self) -> bool:
        """Check if currently paused (non-blocking)."""
        return (self.run_dir / 'pause').exists()

    @property
    def is_stopped(self) -> bool:
        """Check if stop signal exists (non-blocking)."""
        return (self.run_dir / 'stop').exists()
