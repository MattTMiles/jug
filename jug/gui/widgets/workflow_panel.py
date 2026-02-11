"""Session workflow GUI integration — undo/redo and session save/load.

Provides helper functions that wire ``SessionWorkflow`` into MainWindow
without adding significant complexity to ``main_window.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import QFileDialog, QMessageBox

from jug.engine.session_workflow import SessionWorkflow

if TYPE_CHECKING:
    from jug.gui.main_window import MainWindow


class WorkflowManager:
    """Manages SessionWorkflow integration with MainWindow.

    Usage::

        self.workflow_mgr = WorkflowManager(self)  # in MainWindow.__init__
        self.workflow_mgr.record_fit(params_before, params_after, metadata)
        self.workflow_mgr.undo()
    """

    def __init__(self, window: "MainWindow"):
        self.window = window
        self.workflow = SessionWorkflow()

    def init_for_session(self, n_toas: int) -> None:
        """Initialize/reset workflow when a new session loads."""
        self.workflow = SessionWorkflow()
        self.workflow.init_selection(n_toas)
        self._update_button_states()

    # --- Recording --------------------------------------------------------

    def record_fit(
        self,
        params_before: dict,
        params_after: dict,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a fit action in the undo history."""
        self.workflow.record_action("fit", params_before, params_after, metadata)
        self._update_button_states()

    def record_restart(self, params_before: dict, params_after: dict) -> None:
        """Record a restart action."""
        self.workflow.record_action("restart", params_before, params_after)
        self._update_button_states()

    def record_delete(self, params: dict, deleted_indices: list) -> None:
        """Record a TOA deletion action."""
        self.workflow.record_action(
            "delete_toas", params, params,
            metadata={"deleted_indices": deleted_indices},
        )
        self._update_button_states()

    # --- Undo / Redo ------------------------------------------------------

    def undo(self) -> Optional[dict]:
        """Undo the last action. Returns params_before or None."""
        params = self.workflow.undo()
        self._update_button_states()
        return params

    def redo(self) -> Optional[dict]:
        """Redo the next action. Returns params_after or None."""
        params = self.workflow.redo()
        self._update_button_states()
        return params

    # --- Save / Load ------------------------------------------------------

    def save_session(self) -> None:
        """Show Save dialog and write workflow JSON."""
        path, _ = QFileDialog.getSaveFileName(
            self.window,
            "Save Session",
            "",
            "JUG Session Files (*.jug.json);;All Files (*)",
        )
        if path:
            try:
                self.workflow.save_json(Path(path))
                self.window.statusBar().showMessage(
                    f"Session saved: {Path(path).name}", 5000
                )
            except Exception as e:
                QMessageBox.critical(
                    self.window, "Save Error", f"Failed to save session:\n{e}"
                )

    def load_session(self) -> Optional[dict]:
        """Show Open dialog and load workflow JSON.

        Returns the latest params_after if available, or None.
        """
        path, _ = QFileDialog.getOpenFileName(
            self.window,
            "Load Session",
            "",
            "JUG Session Files (*.jug.json);;JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return None

        try:
            self.workflow = SessionWorkflow.load_json(Path(path))
            self._update_button_states()
            self.window.statusBar().showMessage(
                f"Session loaded: {Path(path).name} ({self.workflow.summary()})",
                5000,
            )
            # Return latest params so caller can apply them
            if self.workflow.history:
                return self.workflow.history[self.workflow.undo_index].params_after
            return None
        except Exception as e:
            QMessageBox.critical(
                self.window, "Load Error", f"Failed to load session:\n{e}"
            )
            return None

    # --- Button state -----------------------------------------------------

    def _update_button_states(self) -> None:
        """Enable/disable undo/redo buttons based on workflow state."""
        w = self.window
        if hasattr(w, 'undo_button'):
            w.undo_button.setEnabled(self.workflow.can_undo)
        if hasattr(w, 'redo_button'):
            w.redo_button.setEnabled(self.workflow.can_redo)

    @property
    def summary(self) -> str:
        return self.workflow.summary()
