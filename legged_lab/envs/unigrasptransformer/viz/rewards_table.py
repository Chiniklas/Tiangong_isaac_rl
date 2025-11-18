"""Matplotlib table for per-env reward terms."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch


class RewardsTableViewer:
    """Displays reward components for a single environment in a matplotlib table."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.plt.ion()
        self.figure: Any | None = None
        self.axis: Any | None = None
        self.table: Any | None = None

    def update(self, rows: List[Tuple[str, float]]) -> None:
        cell_text = [[name, f"{value:+.6f}"] for name, value in rows]

        if self.figure is None:
            self.figure = self.plt.figure("Reward Table")
            self.axis = self.figure.add_subplot(111)
            self.axis.axis("off")
            self.table = self.axis.table(cellText=cell_text, colLabels=["Term", "Value"], loc="center")
            self._style_table()
        else:
            self.table.remove()
            self.table = self.axis.table(cellText=cell_text, colLabels=["Term", "Value"], loc="center")
            self._style_table()

        self.figure.canvas.draw_idle()
        self.plt.pause(0.001)

    def _style_table(self) -> None:
        if self.table is None:
            return
        self.table.auto_set_font_size(False)
        self.table.set_fontsize(8)
        self.table.scale(1, 1.2)
