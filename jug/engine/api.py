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
from typing import Dict, List, Optional, Any

from jug.engine.session import TimingSession
from jug.residuals.engine_conventions import EngineConventionProfile
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized


def open_session(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Optional[str] = None,
    verbose: bool = False,
    compatibility: str = "pint",
    engine_conventions: EngineConventionProfile | None = None,
) -> TimingSession:
    """
    Open a timing session for repeated operations.

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
    compatibility : str, default "pint"
        Timing compatibility mode (only ``"pint"`` is supported).
    engine_conventions : EngineConventionProfile, optional
        Explicit engine convention profile (must match *compatibility*).
    """
    return TimingSession(
        par_file=par_file,
        tim_file=tim_file,
        clock_dir=clock_dir,
        verbose=verbose,
        compatibility=compatibility,
        engine_conventions=engine_conventions,
    )


def compute_residuals(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Optional[str] = None,
    subtract_tzr: bool = True,
    verbose: bool = False,
    compatibility: str = "pint",
    engine_conventions: EngineConventionProfile | None = None,
) -> Dict[str, Any]:
    """Compute timing residuals (legacy one-shot API)."""
    return compute_residuals_simple(
        par_file=par_file,
        tim_file=tim_file,
        clock_dir=clock_dir,
        subtract_tzr=subtract_tzr,
        verbose=verbose,
        compatibility=compatibility,
        engine_conventions=engine_conventions,
    )


def fit_parameters(
    par_file: Path | str,
    tim_file: Path | str,
    fit_params: List[str],
    max_iter: int = 25,
    convergence_threshold: float = 1e-14,
    clock_dir: Optional[str] = None,
    device: Optional[str] = None,
    verbose: bool = False,
    compatibility: str = "pint",
) -> Dict[str, Any]:
    """Fit timing model parameters (legacy one-shot API)."""
    return fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=max_iter,
        convergence_threshold=convergence_threshold,
        clock_dir=clock_dir,
        device=device,
        verbose=verbose,
        compatibility=compatibility,
    )