# Copyright (c) Twinkle Contributors. All rights reserved.
"""Log panel widget - scrolling log display."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static
from textual.widget import Widget


class LogPanel(Widget):
    """Scrolling log panel showing training logs in real-time."""

    DEFAULT_CSS = """
    LogPanel {
        layout: vertical;
        border: solid $accent;
        padding: 0;
    }

    LogPanel > #log-title {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        text-align: center;
    }

    LogPanel > #log-scroll {
        height: 1fr;
        padding: 0 1;
    }

    .log-entry {
        color: $text-muted;
    }
    """

    MAX_LINES = 500

    def compose(self) -> ComposeResult:
        yield Static('Logs', id='log-title')
        yield VerticalScroll(id='log-scroll')

    def append_log(self, message: str) -> None:
        """Append a log message to the panel."""
        scroll = self.query_one('#log-scroll', VerticalScroll)
        scroll.mount(Static(message, classes='log-entry'))
        # Trim old entries to prevent memory growth
        children = list(scroll.children)
        if len(children) > self.MAX_LINES:
            for child in children[: len(children) - self.MAX_LINES]:
                child.remove()
        scroll.scroll_end(animate=False)

    def clear(self) -> None:
        """Clear all log entries."""
        scroll = self.query_one('#log-scroll', VerticalScroll)
        scroll.remove_children()
