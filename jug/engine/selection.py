"""Interactive TOA selection and averaging.

Provides programmatic APIs for:
  * **Selection / deselection** of TOAs by index, MJD range, or flag.
  * **Epoch averaging** — collapsing multi-frequency TOAs within a time
    window into a single "averaged" TOA, optionally per-backend.

These operations are non-destructive: the original TOA list is never
modified.  Instead, a ``SelectionState`` object tracks active/deleted
masks and stores averaged results in separate arrays.

The GUI plot layer can consume the ``SelectionState`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------

@dataclass
class SelectionState:
    """Mutable state tracking which TOAs are active and any selections.

    Attributes
    ----------
    n_toas : int
        Total number of original TOAs.
    deleted : np.ndarray of bool
        True for TOAs that are "deleted" (excluded from fitting).
    selected : np.ndarray of bool
        True for TOAs that are "selected" (highlighted in plots).
    """
    n_toas: int
    deleted: np.ndarray = field(default=None, repr=False)
    selected: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.deleted is None:
            self.deleted = np.zeros(self.n_toas, dtype=bool)
        if self.selected is None:
            self.selected = np.zeros(self.n_toas, dtype=bool)

    # --- Properties -------------------------------------------------------

    @property
    def active_mask(self) -> np.ndarray:
        """Boolean mask of active (not deleted) TOAs."""
        return ~self.deleted

    @property
    def active_indices(self) -> np.ndarray:
        """Indices of active TOAs."""
        return np.where(self.active_mask)[0]

    @property
    def n_active(self) -> int:
        return int(np.sum(self.active_mask))

    @property
    def n_deleted(self) -> int:
        return int(np.sum(self.deleted))

    @property
    def n_selected(self) -> int:
        return int(np.sum(self.selected))

    # --- Deletion ---------------------------------------------------------

    def delete_indices(self, indices: Sequence[int]) -> None:
        """Mark TOAs at given indices as deleted."""
        self.deleted[list(indices)] = True

    def undelete_indices(self, indices: Sequence[int]) -> None:
        """Restore deleted TOAs at given indices."""
        self.deleted[list(indices)] = False

    def undelete_all(self) -> None:
        """Restore all deleted TOAs."""
        self.deleted[:] = False

    def delete_by_mjd_range(
        self, mjd_values: np.ndarray, mjd_lo: float, mjd_hi: float
    ) -> int:
        """Delete TOAs within an MJD range. Returns count deleted."""
        mask = (mjd_values >= mjd_lo) & (mjd_values <= mjd_hi)
        count = int(np.sum(mask & ~self.deleted))
        self.deleted[mask] = True
        return count

    def delete_by_flag(
        self, flags: Sequence[Dict[str, str]], key: str, value: str
    ) -> int:
        """Delete TOAs matching a flag key-value pair. Returns count deleted."""
        count = 0
        for i, f in enumerate(flags):
            if f.get(key) == value and not self.deleted[i]:
                self.deleted[i] = True
                count += 1
        return count

    # --- Selection --------------------------------------------------------

    def select_indices(self, indices: Sequence[int]) -> None:
        """Select TOAs at given indices (additive)."""
        self.selected[list(indices)] = True

    def deselect_all(self) -> None:
        """Clear all selections."""
        self.selected[:] = False

    def select_by_mjd_range(
        self, mjd_values: np.ndarray, mjd_lo: float, mjd_hi: float
    ) -> int:
        """Select TOAs within an MJD range. Returns count selected."""
        mask = (mjd_values >= mjd_lo) & (mjd_values <= mjd_hi)
        count = int(np.sum(mask & ~self.selected))
        self.selected[mask] = True
        return count

    def toggle_selection(self, indices: Sequence[int]) -> None:
        """Toggle selection state of given indices."""
        for i in indices:
            self.selected[i] = not self.selected[i]

    # --- Snapshot ----------------------------------------------------------

    def snapshot(self) -> Dict:
        """Return a serialisable snapshot of the current state."""
        return {
            "n_toas": self.n_toas,
            "deleted": self.deleted.copy(),
            "selected": self.selected.copy(),
            "n_active": self.n_active,
            "n_deleted": self.n_deleted,
            "n_selected": self.n_selected,
        }

    @classmethod
    def from_snapshot(cls, snap: Dict) -> "SelectionState":
        """Restore from a snapshot."""
        s = cls(n_toas=snap["n_toas"])
        s.deleted = snap["deleted"].copy()
        s.selected = snap["selected"].copy()
        return s


# ---------------------------------------------------------------------------
# Epoch averaging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AveragedTOA:
    """Result of averaging TOAs within an epoch window.

    Attributes
    ----------
    mjd : float
        Weighted-mean MJD.
    residual_us : float
        Weighted-mean residual in microseconds.
    error_us : float
        Uncertainty of the averaged residual.
    source_indices : tuple of int
        Original TOA indices that were averaged.
    backend : str
        Backend label (or empty if mixed).
    """
    mjd: float
    residual_us: float
    error_us: float
    source_indices: Tuple[int, ...]
    backend: str = ""


def epoch_average(
    mjd_values: np.ndarray,
    residuals_us: np.ndarray,
    errors_us: np.ndarray,
    *,
    active_mask: Optional[np.ndarray] = None,
    dt_days: float = 0.5,
    backends: Optional[Sequence[str]] = None,
) -> List[AveragedTOA]:
    """Average TOAs into epochs.

    Within each epoch (grouped by backend if supplied, and by temporal
    proximity within ``dt_days``), residuals are inverse-variance-
    weighted averaged.

    Parameters
    ----------
    mjd_values : (n,)
    residuals_us : (n,)
    errors_us : (n,)
    active_mask : (n,) of bool, optional
        If given, only active TOAs are averaged.
    dt_days : float, default 0.5
    backends : (n,) of str, optional
        Per-TOA backend labels for per-backend grouping.

    Returns
    -------
    list of AveragedTOA
        Sorted by MJD.
    """
    n_toa = len(mjd_values)
    if active_mask is None:
        active_mask = np.ones(n_toa, dtype=bool)

    active_idx = np.where(active_mask)[0]
    if len(active_idx) == 0:
        return []

    # Group by backend, then by time
    if backends is not None:
        unique_be = sorted(set(backends[i] for i in active_idx))
    else:
        unique_be = [None]

    averaged: List[AveragedTOA] = []

    for be in unique_be:
        if be is not None:
            be_idx = np.array([
                i for i in active_idx if backends[i] == be
            ])
        else:
            be_idx = active_idx

        if len(be_idx) == 0:
            continue

        # Sort by MJD
        order = np.argsort(mjd_values[be_idx])
        sorted_idx = be_idx[order]
        sorted_mjd = mjd_values[sorted_idx]

        # Split into epochs by time gaps
        epochs: List[List[int]] = [[int(sorted_idx[0])]]
        for j in range(1, len(sorted_idx)):
            if sorted_mjd[j] - sorted_mjd[j - 1] > dt_days:
                epochs.append([])
            epochs[-1].append(int(sorted_idx[j]))

        # Average each epoch
        for ep in epochs:
            ep_arr = np.array(ep)
            r = residuals_us[ep_arr]
            e = errors_us[ep_arr]
            m = mjd_values[ep_arr]

            w = 1.0 / e ** 2
            sum_w = np.sum(w)

            avg_r = np.sum(r * w) / sum_w
            avg_m = np.sum(m * w) / sum_w
            avg_e = 1.0 / np.sqrt(sum_w)

            averaged.append(AveragedTOA(
                mjd=float(avg_m),
                residual_us=float(avg_r),
                error_us=float(avg_e),
                source_indices=tuple(ep),
                backend=be if be is not None else "",
            ))

    # Sort by MJD
    averaged.sort(key=lambda a: a.mjd)
    return averaged
