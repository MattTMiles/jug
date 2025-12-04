"""
Pure longdouble precision implementation for spin parameters (F0, F1, F2).

This module provides fitting functions that keep F0/F1/F2 calculations entirely
in numpy longdouble precision (~80 bits) without any JAX conversions that would
downgrade to float64.
"""

import numpy as np
import scipy.linalg
from typing import Tuple, Optional


def compute_spin_phase_longdouble(
    dt_sec_ld: np.ndarray, 
    f0_ld: np.longdouble, 
    f1_ld: np.longdouble,
    f2_ld: Optional[np.longdouble] = None
) -> np.ndarray:
    """
    Compute spin phase in pure longdouble precision.
    
    Phase = dt * (F0 + dt * (F1/2 + dt * F2/6))
    
    Args:
        dt_sec_ld: Time offsets from PEPOCH in longdouble (seconds)
        f0_ld: Spin frequency (Hz)
        f1_ld: First frequency derivative (Hz/s)
        f2_ld: Second frequency derivative (Hz/s^2), optional
        
    Returns:
        phase: Spin phase (cycles) in longdouble
    """
    dt_sec_ld = np.asarray(dt_sec_ld, dtype=np.longdouble)
    
    if f2_ld is not None and f2_ld != 0:
        # Taylor series: F0 + F1*t/1! + F2*t^2/2! + F3*t^3/3!
        # Phase integral: t*(F0 + t*(F1/2 + t*F2/6))
        phase = dt_sec_ld * (
            f0_ld + dt_sec_ld * (
                f1_ld / np.longdouble(2.0) + dt_sec_ld * f2_ld / np.longdouble(6.0)
            )
        )
    else:
        # Just F0 + F1
        phase = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / np.longdouble(2.0)))
    
    return phase


def compute_spin_derivatives_longdouble(
    dt_sec_ld: np.ndarray,
    f0_ld: np.longdouble,
    fit_f2: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Compute design matrix columns for spin parameters in longdouble.
    
    Uses PINT sign convention: M = -d(phase)/d(param) / F0
    
    Args:
        dt_sec_ld: Time offsets from PEPOCH (seconds) in longdouble
        f0_ld: Current F0 value (Hz) in longdouble
        fit_f2: Whether to include F2 derivative
        
    Returns:
        d_f0: Column for F0 (N,) in longdouble
        d_f1: Column for F1 (N,) in longdouble  
        d_f2: Column for F2 (N,) in longdouble or None
    """
    dt_sec_ld = np.asarray(dt_sec_ld, dtype=np.longdouble)
    
    # d(phase)/d(F0) = dt
    # M_F0 = -dt / F0
    d_f0 = -(dt_sec_ld / f0_ld)
    
    # d(phase)/d(F1) = dt^2 / 2
    # M_F1 = -(dt^2 / 2) / F0
    d_f1 = -(dt_sec_ld**2 / np.longdouble(2.0)) / f0_ld
    
    d_f2 = None
    if fit_f2:
        # d(phase)/d(F2) = dt^3 / 6
        # M_F2 = -(dt^3 / 6) / F0
        d_f2 = -(dt_sec_ld**3 / np.longdouble(6.0)) / f0_ld
    
    return d_f0, d_f1, d_f2


def wls_solve_longdouble(
    residuals_ld: np.ndarray,
    errors_ld: np.ndarray,
    M_ld: np.ndarray
) -> np.ndarray:
    """
    WLS solver in pure longdouble precision using scipy.
    
    Solves: (M^T W M) delta = M^T W r
    where W = diag(1/sigma^2)
    
    Args:
        residuals_ld: Timing residuals (seconds) in longdouble
        errors_ld: TOA uncertainties (seconds) in longdouble
        M_ld: Design matrix (N x n_params) in longdouble
        
    Returns:
        delta_params: Parameter updates in longdouble
    """
    residuals_ld = np.asarray(residuals_ld, dtype=np.longdouble)
    errors_ld = np.asarray(errors_ld, dtype=np.longdouble)
    M_ld = np.asarray(M_ld, dtype=np.longdouble)
    
    # Weight by 1/sigma
    weights_solve = np.longdouble(1.0) / errors_ld
    M_weighted = M_ld * weights_solve[:, None]
    r_weighted = residuals_ld * weights_solve
    
    # Solve normal equations
    ATA = M_weighted.T @ M_weighted
    ATb = M_weighted.T @ r_weighted
    delta_params = scipy.linalg.solve(ATA, ATb, assume_a='pos')
    
    return delta_params


def fit_spin_longdouble(
    dt_sec_ld: np.ndarray,
    f0_start_ld: np.longdouble,
    f1_start_ld: np.longdouble,
    errors_sec: np.ndarray,
    weights: np.ndarray,
    f2_start_ld: Optional[np.longdouble] = None,
    max_iter: int = 25,
    threshold: float = 1e-14,
    verbose: bool = True
) -> dict:
    """
    Fit spin parameters (F0, F1, optionally F2) in pure longdouble precision.
    
    This replicates the WLS fitting logic but keeps all spin parameter
    calculations in numpy longdouble (~80 bits) without JAX conversions.
    
    Args:
        dt_sec_ld: Time offsets from PEPOCH (seconds) in longdouble
        f0_start_ld: Initial F0 (Hz) in longdouble
        f1_start_ld: Initial F1 (Hz/s) in longdouble
        errors_sec: TOA uncertainties (seconds) in float64
        weights: TOA weights (1/sigma^2) in float64
        f2_start_ld: Initial F2 (Hz/s^2) in longdouble, optional
        max_iter: Maximum iterations
        threshold: Convergence threshold (not used currently)
        verbose: Print iteration info
        
    Returns:
        dict with keys:
            'f0': Final F0 value
            'f1': Final F1 value
            'f2': Final F2 value (or None)
            'residuals_sec': Final residuals in seconds
            'prefit_rms_us': Prefit RMS in microseconds
            'postfit_rms_us': Postfit RMS in microseconds
            'n_iter': Number of iterations performed
    """
    # Convert errors and weights to longdouble
    errors_ld = np.array(errors_sec, dtype=np.longdouble)
    weights_ld = np.array(weights, dtype=np.longdouble)
    
    # Initialize parameters
    f0_curr = f0_start_ld
    f1_curr = f1_start_ld
    f2_curr = f2_start_ld if f2_start_ld is not None else None
    fit_f2 = f2_curr is not None
    
    prefit_rms = None
    prefit_residuals = None
    
    for iteration in range(max_iter):
        # Compute phase in longdouble
        phase = compute_spin_phase_longdouble(dt_sec_ld, f0_curr, f1_curr, f2_curr)
        
        # Wrap phase
        phase_wrapped = phase - np.round(phase)
        
        # Convert to residuals (seconds)
        residuals = phase_wrapped / f0_curr
        
        # Subtract weighted mean
        weighted_mean = np.sum(residuals * weights_ld) / np.sum(weights_ld)
        residuals = residuals - weighted_mean
        
        # Compute RMS
        rms_sec = np.sqrt(np.sum(residuals**2 * weights_ld) / np.sum(weights_ld))
        rms_us = float(rms_sec * np.longdouble(1e6))
        
        if iteration == 0:
            prefit_rms = rms_us
            prefit_residuals = residuals.copy()
            if verbose:
                print(f"Iteration {iteration}: RMS = {rms_us:.6f} μs (prefit)")
        else:
            if verbose:
                print(f"Iteration {iteration}: RMS = {rms_us:.6f} μs")
        
        # Compute derivatives in longdouble
        d_f0, d_f1, d_f2 = compute_spin_derivatives_longdouble(dt_sec_ld, f0_curr, fit_f2)
        
        # Subtract weighted mean from derivatives
        d_f0 = d_f0 - np.sum(d_f0 * weights_ld) / np.sum(weights_ld)
        d_f1 = d_f1 - np.sum(d_f1 * weights_ld) / np.sum(weights_ld)
        
        # Build design matrix
        if fit_f2:
            d_f2 = d_f2 - np.sum(d_f2 * weights_ld) / np.sum(weights_ld)
            M = np.column_stack([d_f0, d_f1, d_f2])
        else:
            M = np.column_stack([d_f0, d_f1])
        
        # Solve WLS in longdouble
        delta_params = wls_solve_longdouble(residuals, errors_ld, M)
        
        # Update parameters
        f0_curr += delta_params[0]
        f1_curr += delta_params[1]
        if fit_f2:
            f2_curr += delta_params[2]
        
        # Check convergence (could add explicit check here)
        if iteration > 0 and np.max(np.abs(delta_params)) < 1e-20:
            if verbose:
                print(f"Converged after {iteration+1} iterations")
            break
    
    # Final iteration to get postfit residuals
    phase = compute_spin_phase_longdouble(dt_sec_ld, f0_curr, f1_curr, f2_curr)
    phase_wrapped = phase - np.round(phase)
    residuals_final = phase_wrapped / f0_curr
    weighted_mean = np.sum(residuals_final * weights_ld) / np.sum(weights_ld)
    residuals_final = residuals_final - weighted_mean
    
    rms_final = np.sqrt(np.sum(residuals_final**2 * weights_ld) / np.sum(weights_ld))
    postfit_rms = float(rms_final * np.longdouble(1e6))
    
    return {
        'f0': f0_curr,
        'f1': f1_curr,
        'f2': f2_curr,
        'residuals_sec': residuals_final,
        'prefit_rms_us': prefit_rms,
        'postfit_rms_us': postfit_rms,
        'n_iter': iteration + 1
    }
