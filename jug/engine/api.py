"""
High-Level Engine API
=====================

This module provides the public API for the JUG timing engine.
It offers both session-based (cached) and legacy (one-shot) interfaces.

Session-Based API (Recommended):
---------------------------------
session = open_session('pulsar.par', 'pulsar.tim')
result1 = session.compute_residuals()
result2 = session.fit_parameters(['F0', 'F1'])

Legacy API (Backward Compatibility):
-------------------------------------
result = compute_residuals('pulsar.par', 'pulsar.tim')
result = fit_parameters('pulsar.par', 'pulsar.tim', ['F0', 'F1'])
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from jug.engine.session import TimingSession
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized


def open_session(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Optional[str] = None,
    verbose: bool = False
) -> TimingSession:
    """
    Open a timing session for repeated operations.
    
    This is the recommended way to use JUG for multiple operations
    (e.g., compute residuals, fit parameters, compute again).
    The session caches file parsing and expensive setup operations.
    
    Parameters
    ----------
    par_file : Path or str
        Path to .par file
    tim_file : Path or str
        Path to .tim file
    clock_dir : str, optional
        Directory containing clock files
    verbose : bool, default False
        Print status messages
    
    Returns
    -------
    session : TimingSession
        A timing session object
    
    Examples
    --------
    >>> from jug.engine import open_session
    >>> session = open_session('J1909.par', 'J1909.tim')
    >>> result = session.compute_residuals()
    >>> print(f"RMS: {result['rms_us']:.3f} mus")
    >>> 
    >>> # Fit parameters (reuses cached file parsing)
    >>> fit_result = session.fit_parameters(['F0', 'F1'])
    >>> print(f"Fitted F0: {fit_result['final_params']['F0']:.15f} Hz")
    """
    return TimingSession(
        par_file=par_file,
        tim_file=tim_file,
        clock_dir=clock_dir,
        verbose=verbose
    )


def compute_residuals(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Optional[str] = None,
    subtract_tzr: bool = True,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute timing residuals (legacy one-shot API).
    
    This function provides backward compatibility with existing code.
    For multiple operations on the same files, use open_session() instead.
    
    Parameters
    ----------
    par_file : Path or str
        Path to .par file
    tim_file : Path or str
        Path to .tim file
    clock_dir : str, optional
        Directory containing clock files
    subtract_tzr : bool, default True
        Whether to subtract TZR offset
    verbose : bool, default False
        Print status messages
    
    Returns
    -------
    result : dict
        Residuals result with keys:
        - 'residuals_us': Residuals in microseconds
        - 'rms_us': RMS in microseconds
        - 'tdb_mjd': TDB times
        - etc.
    
    Examples
    --------
    >>> from jug.engine import compute_residuals
    >>> result = compute_residuals('J1909.par', 'J1909.tim')
    >>> print(f"RMS: {result['rms_us']:.3f} mus")
    
    Notes
    -----
    This calls compute_residuals_simple() directly without caching.
    For repeated operations, use open_session() instead.
    """
    return compute_residuals_simple(
        par_file=par_file,
        tim_file=tim_file,
        clock_dir=clock_dir,
        subtract_tzr=subtract_tzr,
        verbose=verbose
    )


def fit_parameters(
    par_file: Path | str,
    tim_file: Path | str,
    fit_params: List[str],
    max_iter: int = 25,
    convergence_threshold: float = 1e-14,
    clock_dir: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Fit timing model parameters (legacy one-shot API).
    
    This function provides backward compatibility with existing code.
    For multiple operations on the same files, use open_session() instead.
    
    Parameters
    ----------
    par_file : Path or str
        Path to .par file
    tim_file : Path or str
        Path to .tim file
    fit_params : list of str
        Parameters to fit (e.g., ['F0', 'F1', 'DM'])
    max_iter : int, default 25
        Maximum iterations
    convergence_threshold : float, default 1e-14
        Convergence threshold
    clock_dir : str, optional
        Directory containing clock files
    device : str, optional
        'cpu', 'gpu', or None (auto-detect)
    verbose : bool, default False
        Print fitting progress
    
    Returns
    -------
    result : dict
        Fit results with keys:
        - 'final_params': Fitted parameter values
        - 'uncertainties': Parameter uncertainties
        - 'final_rms': Final RMS in mus
        - 'iterations': Number of iterations
        - 'converged': Whether fit converged
        - etc.
    
    Examples
    --------
    >>> from jug.engine import fit_parameters
    >>> result = fit_parameters('J1909.par', 'J1909.tim', ['F0', 'F1'])
    >>> print(f"F0 = {result['final_params']['F0']:.15f} Hz")
    >>> print(f"RMS = {result['final_rms']:.3f} mus")
    
    Notes
    -----
    This calls fit_parameters_optimized() directly without caching.
    For repeated operations, use open_session() instead.
    """
    return fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        clock_dir=clock_dir,
        device=device,
        verbose=verbose
    )
