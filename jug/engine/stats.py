"""Canonical residual statistics functions.

This module provides the single source of truth for computing timing
residual statistics. All components (GUI, CLI, Python API) must use
these functions to ensure consistent results.

The primary function is compute_residual_stats() which computes
weighted RMS using TOA uncertainties.
"""

import numpy as np
from typing import Dict, Optional


def compute_residual_stats(
    residuals_us: np.ndarray,
    errors_us: Optional[np.ndarray] = None,
    subtract_mean: bool = False
) -> Dict[str, float]:
    """Compute canonical residual statistics.
    
    This is THE source of truth for RMS computation. All JUG components
    must use this function to ensure GUI/CLI/Python API consistency.
    
    Parameters
    ----------
    residuals_us : np.ndarray
        Residual values in microseconds
    errors_us : np.ndarray, optional
        TOA uncertainties in microseconds. If None, uses equal weights.
    subtract_mean : bool, default False
        If True, subtract weighted mean before computing RMS.
        Default False matches current engine behavior.
    
    Returns
    -------
    dict
        Statistics dictionary with keys:
        - 'weighted_rms_us': Weighted RMS in microseconds (primary stat)
        - 'weighted_mean_us': Weighted mean in microseconds
        - 'unweighted_rms_us': Unweighted RMS (std) in microseconds
        - 'n_toas': Number of TOAs
        - 'wsum': Sum of weights (for chi2 calculations)
    
    Notes
    -----
    The weighted RMS formula is:
        wrms = sqrt(sum(w * r^2) / sum(w))
    
    where w = 1/sigma^2 and r = residuals (optionally mean-subtracted).
    
    This matches the engine computation in simple_calculator.py and
    is consistent with standard pulsar timing practice.
    
    Examples
    --------
    >>> residuals = np.array([1.0, 2.0, 3.0])
    >>> errors = np.array([0.1, 0.2, 0.3])
    >>> stats = compute_residual_stats(residuals, errors)
    >>> print(f"Weighted RMS: {stats['weighted_rms_us']:.6f} μs")
    """
    # Ensure float64 arrays
    residuals_us = np.asarray(residuals_us, dtype=np.float64)
    
    n_toas = len(residuals_us)
    
    # Handle empty array
    if n_toas == 0:
        return {
            'weighted_rms_us': 0.0,
            'weighted_mean_us': 0.0,
            'unweighted_rms_us': 0.0,
            'n_toas': 0,
            'wsum': 0.0,
        }
    
    # Compute weights from errors
    if errors_us is not None:
        errors_us = np.asarray(errors_us, dtype=np.float64)
        # Handle zero/negative errors: treat as very large error (very small weight)
        valid_errors = np.where(errors_us > 0, errors_us, np.inf)
        weights = 1.0 / (valid_errors ** 2)
    else:
        # Equal weights if no errors provided
        weights = np.ones(n_toas, dtype=np.float64)
    
    # Sum of weights
    wsum = np.sum(weights)
    
    # Handle edge case where all weights are zero
    if wsum == 0:
        return {
            'weighted_rms_us': 0.0,
            'weighted_mean_us': 0.0,
            'unweighted_rms_us': float(np.std(residuals_us)) if n_toas > 0 else 0.0,
            'n_toas': n_toas,
            'wsum': 0.0,
        }
    
    # Weighted mean
    weighted_mean = np.sum(weights * residuals_us) / wsum
    
    # Residuals for RMS calculation (optionally mean-subtracted)
    if subtract_mean:
        r = residuals_us - weighted_mean
    else:
        r = residuals_us
    
    # Weighted RMS: sqrt(sum(w * r^2) / sum(w))
    weighted_rms = np.sqrt(np.sum(weights * r * r) / wsum)
    
    # Unweighted RMS (standard deviation)
    unweighted_rms = np.std(residuals_us)
    
    return {
        'weighted_rms_us': float(weighted_rms),
        'weighted_mean_us': float(weighted_mean),
        'unweighted_rms_us': float(unweighted_rms),
        'n_toas': n_toas,
        'wsum': float(wsum),
    }


def compute_chi2_reduced(
    residuals_us: np.ndarray,
    errors_us: np.ndarray,
    n_params: int = 0
) -> Dict[str, float]:
    """Compute reduced chi-squared statistic.
    
    Parameters
    ----------
    residuals_us : np.ndarray
        Residual values in microseconds
    errors_us : np.ndarray
        TOA uncertainties in microseconds
    n_params : int
        Number of fitted parameters (for degrees of freedom)
    
    Returns
    -------
    dict
        Statistics including chi2, dof, and chi2_reduced
    """
    residuals_us = np.asarray(residuals_us, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)
    
    n_toas = len(residuals_us)
    dof = max(1, n_toas - n_params)
    
    if n_toas == 0 or np.any(errors_us <= 0):
        return {
            'chi2': 0.0,
            'dof': dof,
            'chi2_reduced': 0.0,
        }
    
    chi2 = np.sum((residuals_us / errors_us) ** 2)
    chi2_reduced = chi2 / dof
    
    return {
        'chi2': float(chi2),
        'dof': dof,
        'chi2_reduced': float(chi2_reduced),
    }
