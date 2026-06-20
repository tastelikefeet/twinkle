# Copyright (c) Twinkle Contributors. All rights reserved.
"""Metrics panel widget - renders training metrics as ASCII charts."""

from __future__ import annotations

from io import StringIO
from typing import Any

import plotext as plt

from textual.app import ComposeResult
from textual.widgets import Static
from textual.widget import Widget


class MetricsPanel(Widget):
    """Renders training metrics (loss, reward, etc.) as terminal plots."""

    DEFAULT_CSS = """
    MetricsPanel {
        layout: vertical;
        border: solid $warning;
        padding: 0;
    }

    MetricsPanel > #metrics-title {
        dock: top;
        height: 1;
        background: $warning;
        color: $text;
        text-align: center;
    }

    MetricsPanel > #metrics-plot {
        height: 1fr;
        padding: 0 1;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics_history: list[dict[str, Any]] = []
        self._x_range: tuple[int | None, int | None] = (None, None)
        self._y_range: tuple[float | None, float | None] = (None, None)

    def compose(self) -> ComposeResult:
        yield Static('Metrics', id='metrics-title')
        yield Static('No data yet...', id='metrics-plot')

    def update_metrics(self, metrics: list[dict[str, Any]]) -> None:
        """Update metrics data and redraw the plot."""
        self._metrics_history = metrics
        self._redraw()

    def zoom(self, x_start: int | None = None, x_end: int | None = None,
             y_min: float | None = None, y_max: float | None = None) -> None:
        """Zoom into a specific range of the chart."""
        self._x_range = (x_start, x_end)
        self._y_range = (y_min, y_max)
        self._redraw()

    def reset_zoom(self) -> None:
        """Reset zoom to show all data."""
        self._x_range = (None, None)
        self._y_range = (None, None)
        self._redraw()

    def _redraw(self) -> None:
        """Redraw the metrics plot."""
        if not self._metrics_history:
            return

        plot_widget = self.query_one('#metrics-plot', Static)

        # Extract available metric keys (skip 'step' and 'ts')
        sample = self._metrics_history[0]
        metric_keys = [k for k in sample.keys() if k not in ('step', 'ts', 'epoch')]

        if not metric_keys:
            plot_widget.update('No plottable metrics.')
            return

        # Get plot size from widget dimensions
        width = max(self.size.width - 4, 40)
        height = max(self.size.height - 4, 10)

        plt.clf()
        plt.plotsize(width, height)
        plt.theme('dark')

        steps = [m.get('step', i) for i, m in enumerate(self._metrics_history)]

        # Apply x-range filter
        x_start, x_end = self._x_range
        if x_start is not None or x_end is not None:
            start_idx = x_start if x_start else 0
            end_idx = x_end if x_end else len(steps)
            steps = steps[start_idx:end_idx]
            data_slice = self._metrics_history[start_idx:end_idx]
        else:
            data_slice = self._metrics_history

        # Plot each metric
        for key in metric_keys[:3]:  # Limit to 3 metrics to avoid clutter
            values = [m.get(key, 0) for m in data_slice]
            if any(v is not None for v in values):
                plt.plot(steps, values, label=key)

        # Apply y-range
        y_min, y_max = self._y_range
        if y_min is not None or y_max is not None:
            plt.ylim(y_min or plt.ylim()[0], y_max or plt.ylim()[1])

        plt.title('Training Metrics')
        plt.xlabel('Step')

        # Render to string
        plot_str = plt.build()
        plot_widget.update(plot_str)
