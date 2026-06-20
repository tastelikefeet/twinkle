# Copyright (c) Twinkle Contributors. All rights reserved.
"""Training monitor - LLM-driven periodic analysis of metrics and logs."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable

from openai import AsyncOpenAI

from twinkle_tui.agent.prompts import MONITOR_SYSTEM_PROMPT
from twinkle_tui.connection import LocalConnection


class TrainingMonitor:
    """Background task that periodically feeds metrics and logs to LLM for analysis.

    Instead of hard-coded rules, the LLM reasons about all training signals:
    loss trends, reward dynamics, gradient norms, KL divergence, throughput,
    entropy collapse, overfitting, and anything else visible in the data.

    When the LLM identifies an issue or has a suggestion, it proactively
    messages the user through the TUI chat panel.
    """

    def __init__(
        self,
        connection: LocalConnection,
        on_message: Callable[[str], None],
        llm_base_url: str = 'http://localhost:11434/v1',
        llm_model: str = 'qwen3.5',
        llm_api_key: str = 'not-needed',
        poll_interval: float = 30.0,
    ):
        self.connection = connection
        self.on_message = on_message
        self.llm_model = llm_model
        self.poll_interval = poll_interval
        self._client = AsyncOpenAI(base_url=llm_base_url, api_key=llm_api_key)
        self._running = True
        self._last_reported_step: int = -1

    async def run(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._analyze()
            except Exception:
                pass  # Don't crash monitor on transient errors
            await asyncio.sleep(self.poll_interval)

    def stop(self) -> None:
        """Stop the monitor."""
        self._running = False

    async def _analyze(self) -> None:
        """Collect metrics + logs, ask LLM to analyze, report if needed."""
        if not self.connection.current_run_id:
            return

        run_id = self.connection.current_run_id

        # Gather data (LocalConnection methods are synchronous)
        metrics = self.connection.get_metrics(run_id, last_n=50)
        logs = self.connection.get_logs(run_id, since=0, limit=30)

        if not metrics:
            return

        # Skip if no new steps since last report
        latest_step = metrics[-1].get('step', 0)
        if latest_step <= self._last_reported_step:
            return

        # Format data for LLM
        data_summary = self._format_for_llm(metrics, logs)

        # Ask LLM to analyze
        response = await self._call_llm(data_summary)

        if response and response.strip():
            # LLM returned something — it found an issue or has advice
            self.on_message(f'[Monitor] {response}')
            self._last_reported_step = latest_step

    def _format_for_llm(self, metrics: list[dict[str, Any]], logs: list[dict[str, Any]]) -> str:
        """Format metrics and logs into a concise text block for LLM analysis."""
        parts = []

        # Metrics summary
        parts.append('## Recent Training Metrics (last 50 steps)')
        if metrics:
            # Show all keys from first entry
            keys = [k for k in metrics[0].keys() if k != 'ts']
            parts.append(f'Fields: {", ".join(keys)}')
            parts.append('')

            # Show last 10 entries in detail
            parts.append('Last 10 entries:')
            for m in metrics[-10:]:
                row = {k: v for k, v in m.items() if k != 'ts'}
                parts.append(f'  {json.dumps(row, default=str)}')

            # Show trend summary for key metrics
            if len(metrics) >= 20:
                parts.append('')
                parts.append('Trend (first 10 vs last 10):')
                for key in keys:
                    if key in ('step', 'epoch'):
                        continue
                    first_vals = [m.get(key) for m in metrics[:10] if m.get(key) is not None]
                    last_vals = [m.get(key) for m in metrics[-10:] if m.get(key) is not None]
                    if first_vals and last_vals:
                        avg_first = sum(first_vals) / len(first_vals)
                        avg_last = sum(last_vals) / len(last_vals)
                        parts.append(f'  {key}: {avg_first:.6g} → {avg_last:.6g}')

        # Recent logs
        if logs:
            parts.append('')
            parts.append('## Recent Logs (last 30 entries)')
            for entry in logs[-30:]:
                parts.append(f'  {entry.get("msg", "")}')

        return '\n'.join(parts)

    async def _call_llm(self, data_summary: str) -> str | None:
        """Call LLM with training data for analysis.

        Returns analysis text if issues found, empty string if all normal.
        """
        messages = [
            {'role': 'system', 'content': MONITOR_SYSTEM_PROMPT},
            {'role': 'user', 'content': data_summary},
        ]

        try:
            response = await self._client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                max_tokens=300,
            )
            content = response.choices[0].message.content or ''
            return content
        except Exception:
            return None
