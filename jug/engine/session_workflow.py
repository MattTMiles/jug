"""Reproducible session workflow manager.

Extends ``TimingSession`` with:
  * A structured **undo/redo** history for parameter changes.
  * **Snapshot** save/restore to/from JSON for session state.
  * **Fit history** tracking (parameter values at each fit iteration).
  * Integration with ``SelectionState`` to track TOA deletion/selection.

This enables the GUI (and scripts) to:
  * Undo the last fit or manual parameter change.
  * Save a "session file" and resume later with identical state.
  * Replay a fitting workflow for reproducibility.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from jug.engine.selection import SelectionState


# ---------------------------------------------------------------------------
# History entry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HistoryEntry:
    """A single undo-able action in the session.

    Attributes
    ----------
    action : str
        Type of action: ``'fit'``, ``'manual_edit'``, ``'delete_toas'``,
        ``'undelete_toas'``, ``'load'``.
    timestamp : float
        Unix timestamp.
    params_before : dict
        Parameter snapshot before the action.
    params_after : dict
        Parameter snapshot after the action.
    metadata : dict
        Extra info (e.g. fit results, deleted indices).
    """
    action: str
    timestamp: float
    params_before: Dict[str, Any]
    params_after: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session workflow
# ---------------------------------------------------------------------------

@dataclass
class SessionWorkflow:
    """Undo/redo history and snapshot manager for timing sessions.

    This object sits alongside a ``TimingSession`` and records every
    meaningful state change.

    Attributes
    ----------
    history : list of HistoryEntry
        Chronological action history.
    undo_index : int
        Current position in history (for undo/redo).
    selection : SelectionState
        TOA selection/deletion state.
    fit_count : int
        Number of fits performed in this session.
    """
    history: List[HistoryEntry] = field(default_factory=list)
    undo_index: int = -1
    selection: Optional[SelectionState] = None
    fit_count: int = 0

    def init_selection(self, n_toas: int) -> None:
        """Initialize or reset the selection state."""
        self.selection = SelectionState(n_toas=n_toas)

    # --- Recording --------------------------------------------------------

    def record_action(
        self,
        action: str,
        params_before: Dict[str, Any],
        params_after: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an action in the history.

        When recording after an undo, any redo entries beyond the
        current position are discarded.
        """
        if metadata is None:
            metadata = {}

        entry = HistoryEntry(
            action=action,
            timestamp=time.time(),
            params_before=_serialize_params(params_before),
            params_after=_serialize_params(params_after),
            metadata=metadata,
        )

        # Truncate any redo branch
        if self.undo_index >= 0 and self.undo_index < len(self.history) - 1:
            self.history = self.history[: self.undo_index + 1]

        self.history.append(entry)
        self.undo_index = len(self.history) - 1

        if action == "fit":
            self.fit_count += 1

    # --- Undo / Redo ------------------------------------------------------

    @property
    def can_undo(self) -> bool:
        return self.undo_index >= 0

    @property
    def can_redo(self) -> bool:
        return self.undo_index < len(self.history) - 1

    def undo(self) -> Optional[Dict[str, Any]]:
        """Return the ``params_before`` of the current entry, then step back.

        Returns None if nothing to undo.
        """
        if not self.can_undo:
            return None
        entry = self.history[self.undo_index]
        self.undo_index -= 1
        return entry.params_before

    def redo(self) -> Optional[Dict[str, Any]]:
        """Return the ``params_after`` of the next entry, then step forward.

        Returns None if nothing to redo.
        """
        if not self.can_redo:
            return None
        self.undo_index += 1
        entry = self.history[self.undo_index]
        return entry.params_after

    # --- Serialisation ----------------------------------------------------

    def to_dict(self) -> Dict:
        """Serialise the workflow state to a JSON-compatible dict."""
        return {
            "fit_count": self.fit_count,
            "undo_index": self.undo_index,
            "selection": _make_json_safe(self.selection.snapshot()) if self.selection else None,
            "history": [
                {
                    "action": e.action,
                    "timestamp": e.timestamp,
                    "params_before": e.params_before,
                    "params_after": e.params_after,
                    "metadata": _make_json_safe(e.metadata),
                }
                for e in self.history
            ],
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SessionWorkflow":
        """Restore from a serialised dict."""
        wf = cls()
        wf.fit_count = d.get("fit_count", 0)
        wf.undo_index = d.get("undo_index", -1)

        sel_data = d.get("selection")
        if sel_data:
            sel = SelectionState(n_toas=sel_data["n_toas"])
            sel.deleted = np.array(sel_data["deleted"], dtype=bool)
            sel.selected = np.array(sel_data["selected"], dtype=bool)
            wf.selection = sel

        for h in d.get("history", []):
            wf.history.append(HistoryEntry(
                action=h["action"],
                timestamp=h.get("timestamp", 0.0),
                params_before=h.get("params_before", {}),
                params_after=h.get("params_after", {}),
                metadata=h.get("metadata", {}),
            ))

        return wf

    def save_json(self, path: Path) -> None:
        """Save workflow to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_json(cls, path: Path) -> "SessionWorkflow":
        """Load workflow from a JSON file."""
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # --- Summary ----------------------------------------------------------

    def summary(self) -> str:
        """One-line summary string."""
        parts = [f"{len(self.history)} actions", f"{self.fit_count} fits"]
        if self.selection:
            parts.append(f"{self.selection.n_active}/{self.selection.n_toas} active")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_params(params: Dict) -> Dict:
    """Deep-copy params, converting non-JSON-safe types."""
    out = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        elif isinstance(v, (dict, list)):
            continue  # skip nested structures
        else:
            try:
                json.dumps(v)
                out[k] = v
            except (TypeError, ValueError):
                out[k] = str(v)
    return out


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj
