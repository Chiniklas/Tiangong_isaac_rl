"""Matplotlib table view for observation sections."""

from __future__ import annotations

from typing import Any, Dict


class ObservationsTableViewer:
    """Displays a single environment's observation block as a matplotlib table."""

    def __init__(self):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.plt.ion()
        self.figures: Dict[str, Any] = {}
        self.axes: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}

    def update(self, name: str, row: torch.Tensor, labels: list[str] | None = None) -> None:
        values = row.detach().cpu().numpy().reshape(-1)
        if labels is None or len(labels) != len(values):
            labels = [f"{idx:03d}" for idx in range(len(values))]
        cell_text = [[label, f"{val:+.6f}"] for label, val in zip(labels, values)]

        if name not in self.figures:
            fig = self.plt.figure(f"Observation Table - {name}")
            ax = fig.add_subplot(111)
            ax.axis("off")
            table = ax.table(cellText=cell_text, colLabels=["Index", "Value"], loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            self.figures[name] = fig
            self.axes[name] = ax
            self.tables[name] = table
        else:
            table = self.tables[name]
            # Remove old rows
            for key in list(table.get_celld().keys()):
                if key[0] > 0:
                    del table._cells[key]
            table.remove()
            ax = self.axes[name]
            table = ax.table(cellText=cell_text, colLabels=["Index", "Value"], loc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.2)
            self.tables[name] = table

        self.figures[name].canvas.draw_idle()
        self.plt.pause(0.001)
