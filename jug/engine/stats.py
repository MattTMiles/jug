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


# ---------------------------------------------------------------------------
# Phase 3.2 — Residual representations
# ---------------------------------------------------------------------------

def compute_normalized_residuals(
    residuals_us: np.ndarray,
    errors_us: np.ndarray,
) -> np.ndarray:
    """Compute normalized residuals r/σ.

    Parameters
    ----------
    residuals_us : np.ndarray
        Residuals in microseconds.
    errors_us : np.ndarray
        TOA uncertainties in microseconds.

    Returns
    -------
    np.ndarray
        Normalized (dimensionless) residuals.
    """
    residuals_us = np.asarray(residuals_us, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)
    safe_errors = np.where(errors_us > 0, errors_us, np.inf)
    return residuals_us / safe_errors


def compute_whitened_residuals(
    residuals_us: np.ndarray,
    errors_us: np.ndarray,
    noise_cov_diag: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute whitened residuals (noise-decorrelated).

    If a diagonal noise covariance is provided, whitening divides by
    ``sqrt(σ² + C_diag)`` (the total effective uncertainty).  Without
    ``noise_cov_diag`` this falls back to simple normalization.

    Parameters
    ----------
    residuals_us : np.ndarray
        Residuals in microseconds.
    errors_us : np.ndarray
        TOA uncertainties in microseconds.
    noise_cov_diag : np.ndarray, optional
        Diagonal of the noise covariance matrix (μs²).  If the noise
        model is not diagonal this should be the marginal variance
        per TOA after conditioning.

    Returns
    -------
    np.ndarray
        Whitened residuals (dimensionless).
    """
    residuals_us = np.asarray(residuals_us, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)

    total_var = errors_us ** 2
    if noise_cov_diag is not None:
        total_var = total_var + np.asarray(noise_cov_diag, dtype=np.float64)

    safe_var = np.where(total_var > 0, total_var, np.inf)
    return residuals_us / np.sqrt(safe_var)


def build_residual_representations(
    prefit_us: np.ndarray,
    postfit_us: np.ndarray,
    errors_us: np.ndarray,
    noise_cov_diag: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Build all residual views from raw data.

    Parameters
    ----------
    prefit_us, postfit_us : np.ndarray
        Pre- and post-fit residuals in microseconds.
    errors_us : np.ndarray
        TOA uncertainties in microseconds.
    noise_cov_diag : np.ndarray, optional
        Diagonal noise covariance (μs²) for whitening.

    Returns
    -------
    dict
        Keys: ``prefit_us``, ``postfit_us``, ``normalized``,
        ``whitened`` (if cov provided, else same as normalized).
    """
    prefit_us = np.asarray(prefit_us, dtype=np.float64)
    postfit_us = np.asarray(postfit_us, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)

    normalized = compute_normalized_residuals(postfit_us, errors_us)
    whitened = compute_whitened_residuals(postfit_us, errors_us, noise_cov_diag)

    return {
        "prefit_us": prefit_us,
        "postfit_us": postfit_us,
        "normalized": normalized,
        "whitened": whitened,
    }


# ---------------------------------------------------------------------------
# Phase 3.3 — Fit summary metrics
# ---------------------------------------------------------------------------

def fit_summary(
    prefit_us: np.ndarray,
    postfit_us: np.ndarray,
    errors_us: np.ndarray,
    fitted_params: Dict[str, float],
    initial_params: Dict[str, float],
    iterations: int = 0,
    converged: bool = False,
    noise_cov_diag: Optional[np.ndarray] = None,
) -> Dict:
    """Compute a comprehensive fit summary.

    Combines residual statistics, chi-squared, parameter deltas,
    convergence info, and numeric conditioning warnings.

    Parameters
    ----------
    prefit_us, postfit_us : np.ndarray
        Pre- and post-fit residuals in microseconds.
    errors_us : np.ndarray
        TOA uncertainties in microseconds.
    fitted_params : dict[str, float]
        Final fitted parameter values.
    initial_params : dict[str, float]
        Initial parameter values (for computing deltas).
    iterations : int
        Number of iterations the fitter ran.
    converged : bool
        Whether the fitter converged.
    noise_cov_diag : np.ndarray, optional
        Diagonal noise covariance for whitened stats.

    Returns
    -------
    dict
        Comprehensive summary with keys:

        - ``prefit_stats``, ``postfit_stats``: dicts from
          ``compute_residual_stats``
        - ``chi2``: dict from ``compute_chi2_reduced``
        - ``param_deltas``: dict of {name: delta_value}
        - ``iterations``: int
        - ``converged``: bool
        - ``warnings``: list of string warnings
        - ``residuals``: dict from ``build_residual_representations``
    """
    postfit_us = np.asarray(postfit_us, dtype=np.float64)
    prefit_us = np.asarray(prefit_us, dtype=np.float64)
    errors_us = np.asarray(errors_us, dtype=np.float64)

    # Stats
    prefit_stats = compute_residual_stats(prefit_us, errors_us)
    postfit_stats = compute_residual_stats(postfit_us, errors_us)

    n_params = len(fitted_params)
    chi2 = compute_chi2_reduced(postfit_us, errors_us, n_params)

    # Parameter deltas
    param_deltas = {}
    for name, val in fitted_params.items():
        init_val = initial_params.get(name, 0.0)
        try:
            param_deltas[name] = float(val) - float(init_val)
        except (TypeError, ValueError):
            param_deltas[name] = None

    # Residual representations
    residuals = build_residual_representations(
        prefit_us, postfit_us, errors_us, noise_cov_diag
    )

    # Conditioning / sanity warnings
    warnings = []
    if not converged:
        warnings.append("Fitter did not converge.")
    if chi2["chi2_reduced"] > 10.0:
        warnings.append(
            f"Reduced χ² = {chi2['chi2_reduced']:.2f} is very large "
            "(expect ~1 for a good fit)."
        )
    if chi2["chi2_reduced"] > 0 and chi2["chi2_reduced"] < 0.1:
        warnings.append(
            f"Reduced χ² = {chi2['chi2_reduced']:.4f} is suspiciously low "
            "(may indicate over-fitting or inflated errors)."
        )
    if postfit_stats["weighted_rms_us"] > prefit_stats["weighted_rms_us"]:
        warnings.append(
            "Post-fit wRMS is larger than pre-fit wRMS — "
            "fit may have diverged."
        )

    return {
        "prefit_stats": prefit_stats,
        "postfit_stats": postfit_stats,
        "chi2": chi2,
        "param_deltas": param_deltas,
        "iterations": iterations,
        "converged": converged,
        "warnings": warnings,
        "residuals": residuals,
    }
