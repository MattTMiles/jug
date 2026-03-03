"""Derivatives for FDJUMP (frequency-dependent JUMP) parameters.

FDJUMP parameters apply frequency-dependent timing offsets to subsets of TOAs,
combining JUMP-like flag-based selection with FD-like frequency dependence.

Model (log scale, default):
    delay = FDJUMP_val * log(freqSSB/1GHz)^idx   for matching TOAs
    delay = 0                                     for non-matching TOAs

Model (linear scale):
    delay = FDJUMP_val * (freqSSB/1GHz)^idx       for matching TOAs

The derivative d(delay)/d(FDJUMP) is the frequency term multiplied by the mask.

Reference: Tempo2 t2fit_stdFitFuncs.C t2FitFunc_fdjump
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional


def compute_fdjump_derivatives(
    params: Dict,
    freq_mhz: np.ndarray,
    fdjump_params: List[str],
    toa_flags: Optional[List[Dict[str, str]]] = None,
    fdjump_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Compute design matrix columns for FDJUMP parameters.

    Parameters
    ----------
    params : dict
        Timing model parameters including _fdjump_meta_* entries
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz, shape (n_toas,)
    fdjump_params : list of str
        FDJUMP parameter names to compute derivatives for
    toa_flags : list of dict, optional
        TOA flag dictionaries (used to build masks if not precomputed)
    fdjump_masks : dict, optional
        Pre-computed boolean masks for each FDJUMP parameter

    Returns
    -------
    dict
        Mapping from FDJUMP parameter name to derivative array (n_toas,)
    """
    derivatives = {}
    freq_ghz = np.asarray(freq_mhz, dtype=np.float64) / 1000.0

    for param_name in fdjump_params:
        meta_key = f'_fdjump_meta_{param_name}'
        meta = params.get(meta_key)
        if meta is None:
            continue

        fd_idx = meta['fd_index']
        log_scale = meta.get('log_scale', True)

        # Frequency term
        if log_scale:
            freq_term = np.log(freq_ghz) ** fd_idx
        else:
            freq_term = freq_ghz ** fd_idx

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
    """Compute total FDJUMP delay contribution.

    Parameters
    ----------
    params : dict
        Timing model parameters
    freq_mhz : np.ndarray
        Barycentric frequencies in MHz
    fdjump_params : list of str
        FDJUMP parameter names
    fdjump_masks : dict
        Boolean masks for each FDJUMP

    Returns
    -------
    np.ndarray
        FDJUMP delay in seconds, shape (n_toas,)
    """
    delay = np.zeros(len(freq_mhz), dtype=np.float64)
    freq_ghz = np.asarray(freq_mhz, dtype=np.float64) / 1000.0

    for param_name in fdjump_params:
        meta_key = f'_fdjump_meta_{param_name}'
        meta = params.get(meta_key)
        if meta is None:
            continue

        fd_idx = meta['fd_index']
        log_scale = meta.get('log_scale', True)
        value = float(params.get(param_name, 0.0))

        if log_scale:
            freq_term = np.log(freq_ghz) ** fd_idx
        else:
            freq_term = freq_ghz ** fd_idx

        mask = fdjump_masks.get(param_name, np.ones(len(freq_mhz), dtype=bool))
        delay += np.where(mask, value * freq_term, 0.0)

    return delay
