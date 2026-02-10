"""Analytical derivatives for spin parameters (F0, F1, F2, ...).

This module implements PINT-compatible analytical derivatives for
spin frequency parameters. The formulas are copied from PINT's
spindown.py to ensure exact compatibility.

Reference: PINT src/pint/models/spindown.py
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict


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
    # d/dF0 = dt^1/1! → position 1
    # d/dF1 = dt^2/2! → position 2
    # d/dF2 = dt^3/3! → position 3
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
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """Compute all spin parameter derivatives for design matrix.
    
    NOTE: Not JIT'd because of Python-level dict/string operations.
    The inner taylor_horner calls ARE JIT'd.
    
    Parameters
    ----------
    params : dict
        Timing model parameters including PEPOCH, F0, F1, F2, etc.
    toas_mjd : jnp.ndarray
        TOA times in MJD
    fit_params : list
        List of parameters to fit (e.g., ['F0', 'F1'])
    **kwargs
        Additional arguments (for API compatibility)
        
    Returns
    -------
    derivatives : dict
        Dictionary mapping parameter name to derivative column
        Each value is jnp.ndarray of shape (n_toas,)
        
    Examples
    --------
    >>> params = {'PEPOCH': 58000.0, 'F0': 339.3, 'F1': -1.6e-15}
    >>> toas = np.array([58000.0, 58001.0, 58002.0])
    >>> derivs = compute_spin_derivatives(params, toas, ['F0', 'F1'])
    >>> derivs['F0'].shape
    (3,)
    """
    # Get PEPOCH
    pepoch_mjd = params.get('PEPOCH', toas_mjd[0])
    
    # Compute dt in seconds
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0  # MJD to seconds
    
    # Get all F terms for API compatibility (not used in derivative)
    f_terms = []
    for i in range(10):  # Support up to F9
        f_key = f'F{i}'
        if f_key in params:
            f_terms.append(params[f_key])
        else:
            break
    
    # Compute derivatives for each requested F parameter
    derivatives = {}
    for param in fit_params:
        if param.startswith('F'):
            deriv_phase = d_phase_d_F(dt_sec, param, f_terms)  # cycles/Hz (POSITIVE)
            # Apply PINT's convention (timing_model.py line 2365):
            # q = -self.d_phase_d_param(toas, delay, param)
            # Then divide by F0 to convert phase → time units (line 2368)
            f0 = params.get('F0', 1.0)
            derivatives[param] = -deriv_phase / f0  # seconds/Hz (NEGATIVE)
    
    return derivatives


# For backward compatibility / testing
@jax.jit
def d_phase_d_F0(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F0.
    
    d(phase)/d(F0) = dt
    """
    return taylor_horner(dt_sec, [0.0, 1.0])


@jax.jit
def d_phase_d_F1(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F1.
    
    d(phase)/d(F1) = dt^2 / 2!
    """
    return taylor_horner(dt_sec, [0.0, 0.0, 1.0])


@jax.jit
def d_phase_d_F2(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F2.
    
    d(phase)/d(F2) = dt^3 / 3!
    """
    return taylor_horner(dt_sec, [0.0, 0.0, 0.0, 1.0])


@jax.jit
def d_phase_d_F3(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F3.
    
    d(phase)/d(F3) = dt^4 / 4!
    """
    return taylor_horner(dt_sec, [0.0, 0.0, 0.0, 0.0, 1.0])


if __name__ == '__main__':
    # Quick test
    print("Testing spin derivatives...")
    
    # Test case from PINT documentation
    dt = np.array([2.0])
    coeffs = [10, 3, 4, 12]
    result = taylor_horner(dt, coeffs)
    print(f"taylor_horner(2.0, [10, 3, 4, 12]) = {result[0]:.1f}")
    print(f"Expected: 40.0")
    print(f"Match: {abs(result[0] - 40.0) < 1e-10}")
    
    # Test derivatives
    dt = np.array([1.0, 2.0, 3.0])
    
    # d/dF0 = dt
    d_f0 = d_phase_d_F0(dt)
    print(f"\nd/dF0 at dt=[1,2,3]: {d_f0}")
    print(f"Expected: [1, 2, 3]")
    
    # d/dF1 = dt^2/2
    d_f1 = d_phase_d_F1(dt)
    print(f"\nd/dF1 at dt=[1,2,3]: {d_f1}")
    print(f"Expected: [0.5, 2.0, 4.5]")
    
    # d/dF2 = dt^3/6
    d_f2 = d_phase_d_F2(dt)
    print(f"\nd/dF2 at dt=[1,2,3]: {d_f2}")
    print(f"Expected: [0.167, 1.333, 4.5]")
    
    print("\n✓ Spin derivatives module ready!")
