"""Analytical derivatives for spin parameters (F0, F1, F2, ...).

Pint mode uses PINT-compatible phase derivatives (``spindown.py``).
Tempo2 mode uses tempo2 ``t2FitFunc_stdFreq``-equivalent time columns
(``-dt^(k+1)/((k+1)! F0)``), matching ``compute_designmatrix`` and libstempo.

Reference: PINT src/pint/models/spindown.py; tempo2 t2fit_stdFitFuncs.C
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
import math
from typing import Dict

from jug.utils.constants import SECS_PER_DAY


def _normalize_spin_compatibility(compatibility: str | None) -> str:
    mode = str(compatibility or "pint").lower()
    if mode in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
        return "tempo2"
    return "pint"


def _spin_param_names(fit_params: list) -> list[str]:
    return [
        p
        for p in fit_params
        if p.startswith("F") and len(p) > 1 and p[1:].isdigit()
    ]


@jax.jit
def taylor_horner(dt: jnp.ndarray, coeffs: list) -> jnp.ndarray:
    """Evaluate Taylor series using Horner's method.
    
    Computes: coeffs[0] + coeffs[1]*dt + coeffs[2]*dt^2/2! + coeffs[3]*dt^3/3! + ...
    
    This is PINT's taylor_horner function adapted for JAX.
    
    Parameters
    ----------
    dt : jnp.ndarray
        Time differences from PEPOCH in seconds
    coeffs : list of float
        Taylor series coefficients [c0, c1, c2, ...]
        Coefficient i is divided by factorial(i) internally
        
    Returns
    -------
    result : jnp.ndarray
        Evaluated Taylor series
        
    Notes
    -----
    Uses Horner's method with factorial division for numerical stability:
    result = c[n]/n! * dt + c[n-1]/(n-1)!
    result = result * dt + c[n-2]/(n-2)!
    ...
    
    Example
    -------
    taylor_horner(2.0, [10, 3, 4, 12])
    # Computes: 10 + 3*2/1! + 4*2^2/2! + 12*2^3/3!
    # = 10 + 6 + 8 + 16 = 40.0
    """
    if len(coeffs) == 0:
        return jnp.zeros_like(dt)
    
    result = 0.0
    fact = len(coeffs)
    
    # Horner's method with factorial division
    for coeff in coeffs[::-1]:  # Reverse order
        result = result * dt / fact + coeff
        fact -= 1.0
    
    return result


def d_phase_d_F(
    dt_sec: jnp.ndarray,
    param_name: str,
    f_terms: list
) -> jnp.ndarray:
    """Compute derivative of phase with respect to spin parameter.
    
    This implements PINT's d_phase_d_F method from spindown.py.
    
    The phase is: phase = F0*dt + F1*dt^2/2! + F2*dt^3/3! + ...
    
    Derivatives:
    d(phase)/d(F0) = dt^1/1! (coeffs at position 1)
    d(phase)/d(F1) = dt^2/2! (coeffs at position 2)
    d(phase)/d(F2) = dt^3/3! (coeffs at position 3)
    d(phase)/d(F3) = dt^4/4! (coeffs at position 4)
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time difference from PEPOCH in seconds, shape (n_toas,)
    param_name : str
        Parameter name, e.g., 'F0', 'F1', 'F2', etc.
    f_terms : list
        Current spin frequency terms [F0, F1, F2, F3, ...]
        Not used in derivative, but kept for API compatibility
        
    Returns
    -------
    derivative : jnp.ndarray
        d(phase)/d(param) in units of cycles/param_unit
        For F0: cycles/Hz
        For F1: cycles/(Hz/s)
        etc.
        
    Notes
    -----
    The derivative is computed by setting coefficient at position (order+1) to 1
    and all others to 0, then evaluating the Taylor series.
    
    taylor_horner evaluates: c[0] + c[1]*dt/1! + c[2]*dt^2/2! + ...
    So for d/dF0 (dt^1/1!), we put 1 at position 1: [0, 1]
    For d/dF1 (dt^2/2!), we put 1 at position 2: [0, 0, 1]
    etc.
    """
    # Extract order from parameter name (F0 -> 0, F1 -> 1, etc.)
    if not param_name.startswith('F'):
        raise ValueError(f"Expected F parameter, got {param_name}")
    
    try:
        order = int(param_name[1:])  # 'F0' -> 0, 'F1' -> 1, etc.
    except ValueError:
        raise ValueError(f"Cannot parse order from {param_name}")
    
    # Create coefficient array with 1 at position (order+1), 0 elsewhere
    # d/dF0 = dt^1/1! -> position 1
    # d/dF1 = dt^2/2! -> position 2
    # d/dF2 = dt^3/3! -> position 3
    max_order = max(order, len(f_terms) - 1) if f_terms else order
    coeffs = [0.0] * (max_order + 2)  # +2 because position is order+1
    coeffs[order + 1] = 1.0
    
    # Evaluate Taylor series to get derivative
    derivative = taylor_horner(dt_sec, coeffs)
    
    # Return POSITIVE derivative (matches PINT's d_phase_d_F)
    # The negative sign is applied later in compute_spin_derivatives()
    # to match PINT's designmatrix() convention (line 2365: q = -self.d_phase_d_param)
    return derivative


def compute_spin_derivatives(
    params: Dict,
    toas_mjd: jnp.ndarray,
    fit_params: list,
    *,
    compatibility: str = "pint",
    dt_sec: np.ndarray | None = None,
) -> Dict[str, jnp.ndarray]:
    """Compute spin-parameter design-matrix columns in seconds per fit unit.

    Parameters
    ----------
    params
        Timing model parameters including PEPOCH, F0, F1, ...
    toas_mjd
        TOA times in MJD (used for pint mode and tempo2 fallback).
    fit_params
        Spin parameters to differentiate (e.g. ``['F0', 'F1']``).
    compatibility
        ``pint`` (default) or ``tempo2`` / ``tempo2-compatible``.
    dt_sec
        Optional emission-time offsets in seconds.  When provided in tempo2
        mode, used instead of ``(toas_mjd - PEPOCH) * SECS_PER_DAY`` so the
        fitter matches ``compute_designmatrix`` (``dt_sec_ld`` from residuals).

    Returns
    -------
    dict
        Mapping parameter name to derivative column, shape ``(n_toas,)``.
    """
    spin_params = _spin_param_names(fit_params)
    if not spin_params:
        return {}

    mode = _normalize_spin_compatibility(compatibility)
    f0 = float(params.get("F0", 1.0))

    if mode == "tempo2":
        if dt_sec is not None:
            dt = np.asarray(dt_sec, dtype=np.float64)
        else:
            pepoch_mjd = float(params.get("PEPOCH", toas_mjd[0]))
            dt = (np.asarray(toas_mjd, dtype=np.float64) - pepoch_mjd) * float(SECS_PER_DAY)
        derivatives: Dict[str, jnp.ndarray] = {}
        for param in spin_params:
            order = int(param[1:])
            col = -(dt ** (order + 1)) / (math.factorial(order + 1) * f0)
            derivatives[param] = jnp.asarray(col, dtype=jnp.float64)
        return derivatives

    pepoch_mjd = params.get("PEPOCH", toas_mjd[0])
    dt_jax = (toas_mjd - pepoch_mjd) * SECS_PER_DAY

    f_terms = []
    for i in range(10):
        f_key = f"F{i}"
        if f_key in params:
            f_terms.append(params[f_key])
        else:
            break

    derivatives = {}
    for param in spin_params:
        deriv_phase = d_phase_d_F(dt_jax, param, f_terms)
        derivatives[param] = -deriv_phase / f0

    return derivatives


