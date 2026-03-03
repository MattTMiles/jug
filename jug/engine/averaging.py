"""Residual averaging/scrunching utilities.

Averages computed residuals (not raw TOAs) by grouping on time window
and system identifier.  Because residuals are already corrected for DM
dispersion, clock offsets, and JUMP parameters, simple weighted averaging
is safe and avoids all precision pitfalls of raw-TOA averaging.
"""

from collections import defaultdict
from typing import Dict, List, Literal, Optional

import numpy as np


def average_residuals(
    mjd: np.ndarray,
    residuals_us: np.ndarray,
    errors_us: np.ndarray,
    toa_flags: Optional[List[Dict[str, str]]] = None,
    mode: Literal["time", "frequency", "backend"] = "time",
    window_days: float = 1.0,
    window_mhz: float = 100.0,
    freq_mhz: Optional[np.ndarray] = None,
    observatories: Optional[List[str]] = None,
) -> dict:
    """Average residuals by grouping and inverse-variance weighting.

    Parameters
    ----------
    mjd : array
        TDB MJD values for each TOA.
    residuals_us : array
        Timing residuals in microseconds.
    errors_us : array
        TOA uncertainties in microseconds.
    toa_flags : list of dict, optional
        Per-TOA flag dictionaries (``-sys``, ``-be``, etc.).
    mode : {'time', 'frequency', 'backend'}
        Grouping strategy:
        - ``'time'``: partition by ``-sys``, then cluster by time gap.
        - ``'frequency'``: partition by ``-sys``, then cluster by
          frequency gap (``window_mhz``).
        - ``'backend'``: partition by ``-be`` / ``-sys`` flag,
          then cluster by time gap.
    window_days : float
        Maximum time gap for time/backend modes.
    window_mhz : float
        Maximum frequency gap for frequency mode.
    freq_mhz : array, optional
        Observing frequency per TOA (returned as weighted mean per group).
    observatories : list of str, optional
        Observatory code per TOA.

    Returns
    -------
    dict
        ``mjd``, ``residuals_us``, ``errors_us``, ``freq_mhz`` (if input
        provided), ``n_averaged``, ``toa_flags`` (representative flags
        per averaged point for color coding).
    """
    n = len(mjd)
    if n == 0:
        return {"mjd": mjd, "residuals_us": residuals_us,
                "errors_us": errors_us, "n_averaged": np.array([], dtype=int),
                "toa_flags": []}

    # Build partition keys
    groups = _build_groups(n, mjd, mode, window_days, window_mhz,
                           toa_flags, observatories, freq_mhz)

    # Weighted-average each group
    out_mjd = []
    out_res = []
    out_err = []
    out_freq = []
    out_navg = []
    out_flags = []

    for indices in groups:
        idx = np.array(indices)
        e = errors_us[idx]
        safe_e = np.where(e > 0, e, np.median(e[e > 0]) if np.any(e > 0) else 1.0)
        w = 1.0 / safe_e**2
        w_sum = w.sum()

        out_mjd.append(np.sum(mjd[idx] * w) / w_sum)
        out_res.append(np.sum(residuals_us[idx] * w) / w_sum)
        out_err.append(1.0 / np.sqrt(w_sum))
        out_navg.append(len(idx))

        if freq_mhz is not None:
            out_freq.append(np.sum(freq_mhz[idx] * w) / w_sum)

        # Representative flags: from best-error TOA in group
        if toa_flags is not None:
            best_i = idx[int(np.argmin(e))]
            out_flags.append(dict(toa_flags[best_i]))
        else:
            out_flags.append({})

    # Sort by MJD
    order = np.argsort(out_mjd)
    result = {
        "mjd": np.array(out_mjd)[order],
        "residuals_us": np.array(out_res)[order],
        "errors_us": np.array(out_err)[order],
        "n_averaged": np.array(out_navg, dtype=int)[order],
        "toa_flags": [out_flags[i] for i in order],
    }
    if freq_mhz is not None:
        result["freq_mhz"] = np.array(out_freq)[order]
    return result


def _build_groups(
    n: int,
    mjd: np.ndarray,
    mode: str,
    window_days: float,
    window_mhz: float,
    toa_flags: Optional[List[Dict[str, str]]],
    observatories: Optional[List[str]],
    freq_mhz: Optional[np.ndarray],
) -> List[List[int]]:
    """Partition TOAs by mode key, then gap-cluster within each partition."""

    partitions: Dict[str, List[int]] = defaultdict(list)

    for i in range(n):
        flags = toa_flags[i] if toa_flags is not None and i < len(toa_flags) else {}

        if mode == "time":
            # Group by system (JUMP key) — collapses freq within same sys
            key = _sys_key(flags, observatories, i)

        elif mode == "frequency":
            # Partition by system first (avoid mixing JUMPs), cluster by freq below
            key = _sys_key(flags, observatories, i)

        elif mode == "backend":
            # Group by backend/system — preserves backend separation
            key = _sys_key(flags, observatories, i)

        else:
            raise ValueError(f"Unknown averaging mode: {mode!r}")

        partitions[key].append(i)

    # Gap-cluster within each partition
    groups: List[List[int]] = []
    for indices in partitions.values():
        if mode == "frequency" and freq_mhz is not None:
            # First cluster by time (1-day gap) to separate epochs,
            # then sub-cluster each epoch by frequency gap.
            indices.sort(key=lambda i: mjd[i])
            time_clusters: List[List[int]] = []
            current = [indices[0]]
            for j in range(1, len(indices)):
                if mjd[indices[j]] - mjd[indices[j - 1]] > 1.0:
                    time_clusters.append(current)
                    current = []
                current.append(indices[j])
            time_clusters.append(current)

            for tc in time_clusters:
                tc.sort(key=lambda i: freq_mhz[i])
                sub = [tc[0]]
                for j in range(1, len(tc)):
                    if freq_mhz[tc[j]] - freq_mhz[tc[j - 1]] > window_mhz:
                        groups.append(sub)
                        sub = []
                    sub.append(tc[j])
                groups.append(sub)
        else:
            # Sort by time, cluster by time gap
            indices.sort(key=lambda i: mjd[i])
            current = [indices[0]]
            for j in range(1, len(indices)):
                if mjd[indices[j]] - mjd[indices[j - 1]] > window_days:
                    groups.append(current)
                    current = []
                current.append(indices[j])
            groups.append(current)

    return groups


def _sys_key(flags: dict, observatories: Optional[List[str]], i: int) -> str:
    """Return system identifier for a single TOA."""
    sid = flags.get("sys", flags.get("be", ""))
    if not sid and observatories is not None and i < len(observatories):
        sid = observatories[i]
    return (sid or "unknown").lower()
