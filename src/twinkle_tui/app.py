# Copyright (c) Twinkle Contributors. All rights reserved.
"""Twinkle TUI main application."""

from __future__ import annotations

import asyncio
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical

from twinkle_tui.agent.core import AgentLoop
from twinkle_tui.agent.monitor import TrainingMonitor
from twinkle_tui.connection import LocalConnection
from twinkle_tui.skills import ModelScopeSkillProvider, SkillManager
from twinkle_tui.widgets.chat import ChatPanel
from twinkle_tui.widgets.logs import LogPanel
from twinkle_tui.widgets.metrics import MetricsPanel
from twinkle_tui.widgets.status_bar import StatusBar


class TwinkleTUI(App):
    """Main Textual application for Twinkle training control."""

    TITLE = 'Twinkle TUI'
    SUB_TITLE = 'ML Training Control'

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: auto 2fr 3fr;
        grid-columns: 2fr 1fr;
    }

    #status-bar {
        column-span: 2;
        height: 3;
    }

    #metrics {
        height: 100%;
    }

    #logs {
        height: 100%;
        row-span: 2;
    }

    #chat {
        height: 100%;
    }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding('q', 'quit', 'Quit'),
        Binding('ctrl+p', 'toggle_metrics', 'Toggle Metrics'),
        Binding('ctrl+l', 'clear_logs', 'Clear Logs'),
    ]

    def __init__(
        self,
        run_id: str | None = None,
        llm_base_url: str = 'http://localhost:11434/v1',
        llm_model: str = 'qwen3.5',
        llm_api_key: str = 'not-needed',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.run_id = run_id
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self._connection: LocalConnection | None = None
        self._agent: AgentLoop | None = None
        self._monitor: TrainingMonitor | None = None
        self._monitor_task: asyncio.Task | None = None
        self._log_poll_task: asyncio.Task | None = None
        self._metrics_poll_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        yield StatusBar(id='status-bar')
        yield MetricsPanel(id='metrics')
        yield LogPanel(id='logs')
        yield ChatPanel(id='chat')

    async def on_mount(self) -> None:
        """Initialize connection, agent, skills, and background tasks."""
        self._connection = LocalConnection()
        if self.run_id:
            self._connection.current_run_id = self.run_id

        # Load skills from all registered providers
        self._skill_manager = SkillManager()
        self._skill_manager.register(ModelScopeSkillProvider())
        await self._skill_manager.load_all()
        skills_prompt = self._skill_manager.format_for_prompt()

        self._agent = AgentLoop(
            connection=self._connection,
            on_message=self._on_agent_message,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
            skills_prompt=skills_prompt,
        )
        self._monitor = TrainingMonitor(
            connection=self._connection,
            on_message=self._on_agent_message,
            llm_base_url=self.llm_base_url,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
        )
        # Connect metrics callback for zoom control
        self._agent.set_metrics_callback(self._handle_metrics_zoom)

        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor.run())
        self._log_poll_task = asyncio.create_task(self._poll_logs())
        self._metrics_poll_task = asyncio.create_task(self._poll_metrics())

    async def _poll_metrics(self, interval: float = 3.0) -> None:
        """Poll local metrics file and update MetricsPanel + StatusBar."""
        while True:
            try:
                if self._connection and self._connection.current_run_id:
                    run_id = self._connection.current_run_id
                    metrics = self._connection.get_metrics(run_id, last_n=200)
                    if metrics:
                        metrics_panel = self.query_one('#metrics', MetricsPanel)
                        metrics_panel.update_metrics(metrics)
                        latest = metrics[-1]
                        status_bar = self.query_one('#status-bar', StatusBar)
                        state = 'paused' if self._connection.is_paused(run_id) else 'training'
                        status_bar.update_status(
                            state=state,
                            step=latest.get('step'),
                            total_steps=latest.get('total_steps'),
                        )
            except Exception:
                pass
            await asyncio.sleep(interval)

    async def _poll_logs(self, interval: float = 2.0) -> None:
        """Poll local logs file and push new entries to LogPanel."""
        while True:
            try:
                if self._connection and self._connection.current_run_id:
                    new_logs = self._connection.get_new_logs(self._connection.current_run_id)
                    if new_logs:
                        log_panel = self.query_one('#logs', LogPanel)
                        for entry in new_logs:
                            log_panel.append_log(entry.get('msg', ''))
            except Exception:
                pass
            await asyncio.sleep(interval)

    def _on_agent_message(self, message: str) -> None:
        """Callback for agent to push messages to chat panel."""
        chat = self.query_one('#chat', ChatPanel)
        chat.add_assistant_message(message)

    def _handle_metrics_zoom(self, action: str, **kwargs) -> None:
        """Handle zoom/reset requests from agent tools."""
        metrics_panel = self.query_one('#metrics', MetricsPanel)
        if action == 'reset':
            metrics_panel.reset_zoom()
        else:
            metrics_panel.zoom(**kwargs)

    async def on_chat_panel_user_submitted(self, event: ChatPanel.UserSubmitted) -> None:
        """Handle user input from chat panel."""
        if self._agent:
            chat = self.query_one('#chat', ChatPanel)
            chat.set_thinking(True)
            try:
                response = await self._agent.send(event.text)
                chat.add_assistant_message(response)
            except Exception as e:
                chat.add_assistant_message(f'[Error] {e}')
            finally:
                chat.set_thinking(False)

    def action_toggle_metrics(self) -> None:
        """Toggle metrics panel visibility."""
        metrics = self.query_one('#metrics', MetricsPanel)
        metrics.toggle_class('hidden')

    def action_clear_logs(self) -> None:
        """Clear the log panel."""
        log_panel = self.query_one('#logs', LogPanel)
        log_panel.clear()

    async def on_unmount(self) -> None:
        """Cleanup background tasks."""
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._log_poll_task:
            self._log_poll_task.cancel()
        if self._metrics_poll_task:
            self._metrics_poll_task.cancel()
