"""Derivatives for FDJUMP / FDJUMPDM parameters.

FDJUMP parameters apply frequency-dependent timing offsets to subsets of TOAs,
combining JUMP-like flag-based selection with FD-like frequency dependence.

Model (log scale, default):
    delay = FDJUMP_val * log(freqSSB/1GHz)^idx   for matching TOAs
    delay = 0                                     for non-matching TOAs

Model (linear scale):
    delay = FDJUMP_val * (freqSSB/1GHz)^idx       for matching TOAs

FDJUMPDM (Tempo2 fdjumpIdx == -2) is a masked DM offset:
    delay = FDJUMPDM_val * K_DM / freq_MHz^2     for matching TOAs

The derivative d(delay)/d(param) is the frequency term multiplied by the mask.

Reference: Tempo2 t2fit_stdFitFuncs.C t2FitFunc_fdjump / formResiduals.C
"""

import numpy as np
from typing import Dict, List, Optional

from jug.utils.constants import K_DM_SEC


def _is_fdjumpdm(meta: Dict) -> bool:
    """True for FDJUMPDM entries (Tempo2 idx==-2 or explicit kind)."""
    return meta.get("kind") == "dm" or int(meta.get("fd_index", 0)) == -2


def _fdjump_freq_term(freq_mhz: np.ndarray, meta: Dict) -> np.ndarray:
    """Frequency kernel for one FDJUMP / FDJUMPDM parameter (seconds per unit)."""
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if _is_fdjumpdm(meta):
        # Guard infinite-frequency TOAs the same way as the DM delay path.
        freq_safe = np.where(freq_mhz > 1.0e-6, freq_mhz, np.inf)
        return K_DM_SEC / (freq_safe ** 2)

    fd_idx = int(meta["fd_index"])
    log_scale = meta.get("log_scale", True)
    freq_ghz = freq_mhz / 1000.0
    if log_scale:
        return np.log(freq_ghz) ** fd_idx
    return freq_ghz ** fd_idx


def compute_fdjump_derivatives(
    params: Dict,
    freq_mhz: np.ndarray,
    fdjump_params: List[str],
    toa_flags: Optional[List[Dict[str, str]]] = None,
    fdjump_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Compute design matrix columns for FDJUMP / FDJUMPDM parameters.

    Parameters
    ----------
    params : dict
        Timing model parameters including _fdjump_meta_* entries
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz, shape (n_toas,)
    fdjump_params : list of str
        FDJUMP / FDJUMPDM parameter names to compute derivatives for
    toa_flags : list of dict, optional
        TOA flag dictionaries (used to build masks if not precomputed)
    fdjump_masks : dict, optional
        Pre-computed boolean masks for each FDJUMP parameter

    Returns
    -------
    dict
        Mapping from parameter name to derivative array (n_toas,)
    """
    derivatives = {}

    for param_name in fdjump_params:
        meta_key = f'_fdjump_meta_{param_name}'
        meta = params.get(meta_key)
        if meta is None:
            continue

        freq_term = _fdjump_freq_term(freq_mhz, meta)

        # Mask
        if fdjump_masks is not None and param_name in fdjump_masks:
            mask = fdjump_masks[param_name]
        elif toa_flags is not None:
            flag_name = meta['flag_name']
            flag_value = meta['flag_value']
            mask = np.zeros(len(freq_mhz), dtype=bool)
            for i, flags in enumerate(toa_flags):
                val = flags.get(flag_name)
                if isinstance(val, list):
                    if flag_value in val:
                        mask[i] = True
                elif val == flag_value:
                    mask[i] = True
        else:
            mask = np.ones(len(freq_mhz), dtype=bool)

        deriv = np.where(mask, freq_term, 0.0)
        derivatives[param_name] = deriv

    return derivatives


def compute_fdjump_delay(
    params: Dict,
    freq_mhz: np.ndarray,
    fdjump_params: List[str],
    fdjump_masks: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute total FDJUMP / FDJUMPDM delay contribution.

    Parameters
    ----------
    params : dict
        Timing model parameters
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz
    fdjump_params : list of str
        FDJUMP / FDJUMPDM parameter names
    fdjump_masks : dict
        Boolean masks for each parameter

    Returns
    -------
    np.ndarray
        Delay in seconds, shape (n_toas,)
    """
    delay = np.zeros(len(freq_mhz), dtype=np.float64)

    for param_name in fdjump_params:
        meta_key = f'_fdjump_meta_{param_name}'
        meta = params.get(meta_key)
        if meta is None:
            continue

        value = float(params.get(param_name, 0.0))
        freq_term = _fdjump_freq_term(freq_mhz, meta)
        mask = fdjump_masks.get(param_name, np.ones(len(freq_mhz), dtype=bool))
        delay += np.where(mask, value * freq_term, 0.0)

    return delay
