"""DMX (dispersion measure by epoch) support.

DMX is an alternative to a smooth polynomial DM model: the dataset is
split into time windows (``DMX_####`` groups) and a separate offset
``ΔDM_k`` is fitted for each window.  This is widely used in
NANOGrav / PINT analyses where DM variations are too rapid for a low-
order polynomial.

Design
------
* **Par-file driven**: DMX ranges are defined in the ``.par`` file via
  ``DMXR1_####`` (start MJD), ``DMXR2_####`` (end MJD), and optionally
  ``DMX_####`` (initial value), ``DMXF1_####`` / ``DMXF2_####`` (frequency
  bounds).
* **Design matrix columns**: one column per DMX window, with the standard
  DM chromatic derivative ``K_DM / ν²`` inside the window and zero
  outside.
* **Integration with fitter**: the DMX columns can be appended to the
  general design matrix exactly like other timing parameters.

This module provides:
  * ``DMXRange`` dataclass describing one DMX window
  * ``parse_dmx_ranges(params)`` — extract all DMX ranges from a par dict
  * ``assign_toas_to_dmx(toas_mjd, ranges)`` — build a membership array
  * ``build_dmx_design_matrix(…)`` — design matrix for DMX fitting
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import re

import numpy as np

from jug.utils.constants import K_DM_SEC


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DMXRange:
    """A single DMX time window.

    Attributes
    ----------
    index : int
        DMX range index (e.g. 1 for ``DMX_0001``).
    r1_mjd : float
        Start MJD (inclusive).
    r2_mjd : float
        End MJD (inclusive).
    value : float
        Initial DMX value (pc/cm³), or 0.0 if not specified.
    freq_lo_mhz : float
        Lower frequency bound (MHz), or 0.0 (no bound).
    freq_hi_mhz : float
        Upper frequency bound (MHz), or ``np.inf`` (no bound).
    label : str
        Canonical parameter name, e.g. ``'DMX_0001'``.
    """
    index: int
    r1_mjd: float
    r2_mjd: float
    value: float = 0.0
    freq_lo_mhz: float = 0.0
    freq_hi_mhz: float = np.inf
    label: str = ""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_dmx_ranges(params: Dict) -> List[DMXRange]:
    """Extract all DMX ranges from a par file dict.

    Looks for keys like ``DMXR1_####`` and ``DMXR2_####``.

    Parameters
    ----------
    params : dict
        Parsed par file dictionary.

    Returns
    -------
    list of DMXRange
        Sorted by index.  Only ranges with both R1 and R2 present are
        returned.
    """
    # Collect all DMX indices present
    pattern_r1 = re.compile(r'^DMXR1_(\d+)$', re.IGNORECASE)
    r1_keys: Dict[int, str] = {}
    for key in params:
        m = pattern_r1.match(key)
        if m:
            r1_keys[int(m.group(1))] = key

    ranges: List[DMXRange] = []
    for idx in sorted(r1_keys):
        r1_key = r1_keys[idx]
        r2_key = f"DMXR2_{idx:04d}"
        # Try both padded and unpadded
        if r2_key not in params:
            r2_key = f"DMXR2_{idx}"
        if r2_key not in params:
            continue  # skip incomplete range

        r1_mjd = float(params[r1_key])
        r2_mjd = float(params[r2_key])

        # DMX value
        dmx_key = f"DMX_{idx:04d}"
        if dmx_key not in params:
            dmx_key = f"DMX_{idx}"
        value = float(params.get(dmx_key, 0.0))

        # Frequency bounds (optional)
        f1_key = f"DMXF1_{idx:04d}"
        if f1_key not in params:
            f1_key = f"DMXF1_{idx}"
        f2_key = f"DMXF2_{idx:04d}"
        if f2_key not in params:
            f2_key = f"DMXF2_{idx}"
        freq_lo = float(params.get(f1_key, 0.0))
        freq_hi = float(params.get(f2_key, np.inf))

        label = f"DMX_{idx:04d}"

        ranges.append(DMXRange(
            index=idx,
            r1_mjd=r1_mjd,
            r2_mjd=r2_mjd,
            value=value,
            freq_lo_mhz=freq_lo,
            freq_hi_mhz=freq_hi,
            label=label,
        ))

    return ranges


# ---------------------------------------------------------------------------
# TOA assignment
# ---------------------------------------------------------------------------

def assign_toas_to_dmx(
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    ranges: Sequence[DMXRange],
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each TOA to a DMX window.

    Parameters
    ----------
    toas_mjd : (n_toa,)
    freq_mhz : (n_toa,)
    ranges : sequence of DMXRange

    Returns
    -------
    assignment : np.ndarray of int, shape (n_toa,)
        Index into ``ranges`` for each TOA, or -1 if not in any window.
    mask_matrix : np.ndarray of bool, shape (n_toa, n_ranges)
        Boolean membership matrix.
    """
    n_toa = len(toas_mjd)
    n_ranges = len(ranges)
    assignment = np.full(n_toa, -1, dtype=np.int32)
    mask_matrix = np.zeros((n_toa, n_ranges), dtype=bool)

    for k, rng in enumerate(ranges):
        in_time = (toas_mjd >= rng.r1_mjd) & (toas_mjd <= rng.r2_mjd)
        # DMXF1/DMXF2 are informational metadata, not filtering criteria
        # (consistent with PINT/Tempo2 behavior: assignment is by MJD only)
        mask = in_time
        mask_matrix[:, k] = mask
        # First match wins if ranges overlap (later ranges don't override)
        new = mask & (assignment == -1)
        assignment[new] = k

    return assignment, mask_matrix


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def build_dmx_design_matrix(
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    ranges: Sequence[DMXRange],
) -> Tuple[np.ndarray, List[str]]:
    """Build the DMX design matrix.

    Each column is the standard DM chromatic derivative
    ``K_DM / freq_mhz²`` inside the corresponding DMX window, and
    zero outside.

    Parameters
    ----------
    toas_mjd : (n_toa,)
    freq_mhz : (n_toa,)
    ranges : sequence of DMXRange

    Returns
    -------
    M_dmx : np.ndarray, shape (n_toa, n_ranges)
        Design matrix.
    labels : list of str
        Column labels (``DMXRange.label``).
    """
    n_toa = len(toas_mjd)
    n_ranges = len(ranges)
    _, mask_matrix = assign_toas_to_dmx(toas_mjd, freq_mhz, ranges)

    # DM derivative: d(delay)/d(DM) = K_DM / freq_mhz² [seconds per pc/cm³]
    dm_deriv = K_DM_SEC / (freq_mhz ** 2)  # (n_toa,)

    M_dmx = np.zeros((n_toa, n_ranges), dtype=np.float64)
    for k in range(n_ranges):
        M_dmx[mask_matrix[:, k], k] = dm_deriv[mask_matrix[:, k]]

    labels = [r.label for r in ranges]
    return M_dmx, labels


def get_dmx_delays(
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    ranges: Sequence[DMXRange],
) -> np.ndarray:
    """Compute the DMX delay contribution for each TOA.

    Parameters
    ----------
    toas_mjd : (n_toa,)
    freq_mhz : (n_toa,)
    ranges : sequence of DMXRange

    Returns
    -------
    delay_sec : np.ndarray, shape (n_toa,)
        DMX delay in seconds.
    """
    n_toa = len(toas_mjd)
    delay = np.zeros(n_toa, dtype=np.float64)
    assignment, _ = assign_toas_to_dmx(toas_mjd, freq_mhz, ranges)

    dm_deriv = K_DM_SEC / (freq_mhz ** 2)

    for i in range(n_toa):
        k = assignment[i]
        if k >= 0:
            delay[i] = ranges[k].value * dm_deriv[i]

    return delay
