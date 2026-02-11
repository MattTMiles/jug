"""Epoch averaging overlay for the residual plot.

Provides a helper class that manages the averaged-epoch scatter plot
overlay.  The overlay is purely visual — it does not modify the
underlying TOA data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pyqtgraph as pg

from jug.engine.selection import epoch_average, AveragedTOA
from jug.gui.theme import Colors

if TYPE_CHECKING:
    from jug.gui.main_window import MainWindow


class AveragingOverlay:
    """Manages the epoch-averaging scatter overlay on the residual plot.

    Usage::

        self.avg_overlay = AveragingOverlay(self)  # in MainWindow.__init__
        self.avg_overlay.update()  # after data changes
        self.avg_overlay.toggle(True)  # show / hide
    """

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.enabled = False
        self._scatter_item: Optional[pg.ScatterPlotItem] = None
        self._error_bar_item: Optional[pg.ErrorBarItem] = None
        self._averaged: List[AveragedTOA] = []

    @property
    def is_active(self) -> bool:
        return self.enabled and len(self._averaged) > 0

    def toggle(self, enabled: bool) -> None:
        """Enable or disable the averaging overlay."""
        self.enabled = enabled
        if enabled:
            self.update()
        else:
            self._remove_items()

    def update(self) -> None:
        """Recompute averaged epochs and update the overlay."""
        if not self.enabled:
            return

        w = self.window
        if w.mjd is None or w.residuals_us is None or w.errors_us is None:
            self._remove_items()
            return

        # Compute averaged epochs
        self._averaged = epoch_average(
            w.mjd,
            w.residuals_us,
            w.errors_us,
            dt_days=0.5,
        )

        if not self._averaged:
            self._remove_items()
            return

        # Extract arrays
        avg_mjd = np.array([a.mjd for a in self._averaged])
        avg_res = np.array([a.residual_us for a in self._averaged])
        avg_err = np.array([a.error_us for a in self._averaged])

        # Create or update scatter item (larger, distinct color)
        if self._scatter_item is None:
            self._scatter_item = pg.ScatterPlotItem(
                size=10,
                pen=pg.mkPen(Colors.ACCENT_PRIMARY, width=1.5),
                brush=pg.mkBrush(Colors.ACCENT_PRIMARY + "80"),  # semi-transparent
                symbol='d',  # diamond
            )
            w.plot_widget.addItem(self._scatter_item)

        self._scatter_item.setData(x=avg_mjd, y=avg_res)

        # Create or update error bars
        if self._error_bar_item is None:
            self._error_bar_item = pg.ErrorBarItem(
                pen=pg.mkPen(Colors.ACCENT_PRIMARY + "A0", width=1.5),
                beam=0.0,
            )
            w.plot_widget.addItem(self._error_bar_item)

        self._error_bar_item.setData(
            x=avg_mjd,
            y=avg_res,
            height=2 * avg_err,
        )

        # Update status
        n_epochs = len(self._averaged)
        n_toas = len(w.mjd)
        w.statusBar().showMessage(
            f"Epoch averaging: {n_toas} TOAs → {n_epochs} epochs", 3000
        )

    def _remove_items(self) -> None:
        """Remove overlay items from the plot."""
        w = self.window
        if self._scatter_item is not None:
            w.plot_widget.removeItem(self._scatter_item)
            self._scatter_item = None
        if self._error_bar_item is not None:
            w.plot_widget.removeItem(self._error_bar_item)
            self._error_bar_item = None
        self._averaged = []

    def clear(self) -> None:
        """Fully remove and reset (e.g. on Restart)."""
        self._remove_items()
        self.enabled = False
