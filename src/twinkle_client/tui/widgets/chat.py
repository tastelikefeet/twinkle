# Copyright (c) Twinkle Contributors. All rights reserved.
"""Chat panel widget - handles user/agent conversation display and input."""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.message import Message as TextualMessage
from textual.widgets import Input, RichLog, Static
from textual.widget import Widget


class ChatPanel(Widget):
    """Interactive chat panel for user <-> agent conversation."""

    DEFAULT_CSS = """
    ChatPanel {
        layout: vertical;
        border: solid $primary;
        padding: 0;
    }

    ChatPanel > #chat-title {
        dock: top;
        height: 1;
        background: $primary;
        color: $text;
        text-align: center;
    }

    ChatPanel > #chat-log {
        height: 1fr;
        padding: 0 1;
    }

    ChatPanel > #chat-input {
        dock: bottom;
        height: 3;
        margin: 0 1;
    }

    ChatPanel > #streaming-text {
        dock: bottom;
        max-height: 6;
        padding: 0 1;
        color: $text;
        display: none;
    }
    """

    class UserSubmitted(TextualMessage):
        """Event emitted when user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    # Minimum interval (seconds) between UI updates during streaming
    _STREAM_THROTTLE = 0.05  # 50ms → ~20fps

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._streaming_buffer = ''
        self._streaming_widget: Static | None = None
        self._last_render_time: float = 0.0
        self._dirty = False

    def compose(self) -> ComposeResult:
        yield Static('Chat', id='chat-title')
        yield RichLog(id='chat-log', wrap=True, markup=True, max_lines=200)
        yield Static('', id='streaming-text')
        yield Input(placeholder='Ask the agent anything...', id='chat-input')

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        text = event.value.strip()
        if not text:
            return
        event.input.value = ''
        self.add_user_message(text)
        self.post_message(self.UserSubmitted(text))

    def add_user_message(self, text: str) -> None:
        """Add a user message to the chat log."""
        self.query_one('#chat-log', RichLog).write(f'[bold green]You:[/] {text}')

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the chat log."""
        self.query_one('#chat-log', RichLog).write(f'[bold cyan]Agent:[/] {text}')

    def start_streaming(self) -> None:
        """Begin a streaming assistant response."""
        self._streaming_buffer = ''
        self._dirty = False
        self._last_render_time = 0.0
        self._streaming_widget = self.query_one('#streaming-text', Static)
        self._streaming_widget.update('[bold cyan]Agent:[/] █')
        self._streaming_widget.styles.display = 'block'

    def reset_stream(self) -> None:
        """Discard buffered streaming content (called when tool-calls detected).

        Resets to the initial streaming state so the next round starts fresh.
        """
        self._streaming_buffer = ''
        self._dirty = False
        if self._streaming_widget is not None:
            self._streaming_widget.update('[bold cyan]Agent:[/] [dim]calling tools...[/]')

    def append_stream(self, chunk: str) -> None:
        """Append a chunk to the streaming display (throttled)."""
        self._streaming_buffer += chunk
        self._dirty = True
        now = time.monotonic()
        if now - self._last_render_time >= self._STREAM_THROTTLE:
            self._flush_stream()

    def _flush_stream(self) -> None:
        """Actually update the streaming widget text."""
        if not self._dirty or self._streaming_widget is None:
            return
        display_text = self._streaming_buffer[-300:]
        self._streaming_widget.update(f'[bold cyan]Agent:[/] {display_text}█')
        self._last_render_time = time.monotonic()
        self._dirty = False

    def finish_streaming(self) -> str:
        """End streaming: move buffer to chat log, hide streaming widget.

        Returns the full accumulated text.
        """
        self._flush_stream()
        if self._streaming_widget is not None:
            self._streaming_widget.update('')
            self._streaming_widget.styles.display = 'none'
            self._streaming_widget = None
        full_text = self._streaming_buffer
        if full_text:
            self.add_assistant_message(full_text)
        self._streaming_buffer = ''
        return full_text
