# Copyright (c) Twinkle Contributors. All rights reserved.
"""Chat panel widget - handles user/agent conversation display and input."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message as TextualMessage
from textual.widgets import Input, Markdown, Static
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

    ChatPanel > #chat-scroll {
        height: 1fr;
        padding: 0 1;
    }

    ChatPanel > #chat-input {
        dock: bottom;
        height: 3;
        margin: 0 1;
    }

    ChatPanel > #thinking-indicator {
        dock: bottom;
        height: 1;
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }

    .chat-user {
        color: $success;
        margin: 1 0 0 0;
    }

    .chat-assistant {
        color: $secondary;
        margin: 0 0 1 0;
    }
    """

    class UserSubmitted(TextualMessage):
        """Event emitted when user submits a message."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def compose(self) -> ComposeResult:
        yield Static('Chat', id='chat-title')
        yield VerticalScroll(id='chat-scroll')
        yield Static('', id='thinking-indicator')
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
        """Add a user message to the chat scroll."""
        scroll = self.query_one('#chat-scroll', VerticalScroll)
        scroll.mount(Static(f'[bold green]You:[/] {text}', classes='chat-user'))
        scroll.scroll_end(animate=False)

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to the chat scroll."""
        scroll = self.query_one('#chat-scroll', VerticalScroll)
        scroll.mount(Static(f'[bold cyan]Agent:[/] {text}', classes='chat-assistant'))
        scroll.scroll_end(animate=False)

    def set_thinking(self, thinking: bool) -> None:
        """Show/hide thinking indicator."""
        indicator = self.query_one('#thinking-indicator', Static)
        indicator.update('Agent is thinking...' if thinking else '')
