"""JUG fitting module for parameter optimization.

This module provides linearized least squares fitting (standard pulsar timing approach).

Key Components
--------------
optimized_fitter : Module
    Production-ready optimized fitter with JAX JIT compilation.
    Main API: fit_parameters_optimized()

derivatives_spin : Module
    Analytical derivatives for spin parameters (F0, F1, F2, ...)
    Uses PINT-compatible formulas for exact agreement.

wls_fitter : Module
    Weighted least squares solver using SVD decomposition.
    Returns parameter updates and covariance matrix.

Usage Example
-------------
High-level API (recommended):

>>> from jug.fitting import fit_parameters_optimized
>>> from pathlib import Path
>>>
>>> result = fit_parameters_optimized(
>>>     par_file=Path("pulsar.par"),
>>>     tim_file=Path("pulsar.tim"),
>>>     fit_params=['F0', 'F1']
>>> )
>>> print(f"F0 = {result['final_params']['F0']:.15f} Hz")
>>> print(f"RMS = {result['final_rms']:.3f} mus")

Low-level API (for custom fitting):

>>> from jug.fitting.derivatives_spin import compute_spin_derivatives
>>> from jug.fitting.wls_fitter import wls_solve_svd
>>> import numpy as np
>>>
>>> # Compute derivatives for F0 and F1
>>> params = {'F0': 339.315, 'F1': -1.6e-15, 'PEPOCH': 55000.0}
>>> toas_mjd = np.array([55000.0, 55001.0, 55002.0])
>>> derivs = compute_spin_derivatives(params, toas_mjd, ['F0', 'F1'])
>>>
>>> # Build design matrix
>>> M = np.column_stack([derivs['F0'], derivs['F1']])
>>>
>>> # Solve WLS (residuals and errors in seconds!)
>>> residuals_sec = np.array([1e-6, 2e-6, -1e-6])
>>> errors_sec = np.array([1e-6, 1e-6, 1e-6])
>>> delta_params, cov, _ = wls_solve_svd(residuals_sec, errors_sec, M)
>>>
>>> # Update parameters
>>> f0_new = params['F0'] + delta_params[0]
>>> f1_new = params['F1'] + delta_params[1]

Status
------
[x] Spin parameters (F0, F1, F2): VALIDATED against Tempo2/PINT
* DM parameters (DM, DM1, DM2, ...): Coming in Milestone 3
* Astrometry parameters (RA, DEC, PM, PX): Coming in Milestone 3
* Binary parameters: Coming in Milestone 3
"""

from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.derivatives_jump import compute_jump_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.fitting.optimized_fitter import fit_parameters_optimized

__all__ = [
    'compute_spin_derivatives',
    'compute_jump_derivatives',
    'wls_solve_svd',
    'fit_parameters_optimized'
]
