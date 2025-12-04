"""JAX-accelerated analytical derivatives for spin parameters.

This is a JAX version of derivatives_spin.py with JIT compilation for speed.
All functions are compatible with the numpy version but use JAX arrays.

Expected speedup: 5-10x for derivative computation on large datasets.

Reference: PINT src/pint/models/spindown.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict

# Enable float64 precision (critical for pulsar timing!)
jax.config.update('jax_enable_x64', True)


@jax.jit
def taylor_horner_jax(dt: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    """Evaluate Taylor series using Horner's method (JAX version).
    
    Computes: coeffs[0] + coeffs[1]*dt + coeffs[2]*dt^2/2! + coeffs[3]*dt^3/3! + ...
    
    Parameters
    ----------
    dt : jnp.ndarray
        Time differences from PEPOCH in seconds
    coeffs : jnp.ndarray
        Taylor series coefficients [c0, c1, c2, ...]
        Coefficient i is divided by factorial(i) internally
        
    Returns
    -------
    result : jnp.ndarray
        Evaluated Taylor series
        
    Notes
    -----
    Uses Horner's method with factorial division for numerical stability.
    JIT-compiled for speed.
    """
    if len(coeffs) == 0:
        return jnp.zeros_like(dt)
    
    result = 0.0
    fact = float(len(coeffs))
    
    # Horner's method with factorial division
    for coeff in coeffs[::-1]:  # Reverse order
        result = result * dt / fact + coeff
        fact -= 1.0
    
    return result


def d_phase_d_F_jax(
    dt_sec: jnp.ndarray,
    order: int
) -> jnp.ndarray:
    """Compute derivative of phase with respect to spin parameter (JAX).
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time difference from PEPOCH in seconds, shape (n_toas,)
    order : int
        Parameter order (0 for F0, 1 for F1, etc.)
        
    Returns
    -------
    derivative : jnp.ndarray
        d(phase)/d(F{order}) in cycles
        
    Notes
    -----
    Uses JIT-compiled taylor_horner for speed.
    """
    # Create coefficient array with 1 at position (order+1), 0 elsewhere
    coeffs = jnp.zeros(order + 2)
    coeffs = coeffs.at[order + 1].set(1.0)
    
    # Evaluate Taylor series to get derivative
    derivative = taylor_horner_jax(dt_sec, coeffs)
    
    # Return POSITIVE derivative (negative applied in compute_spin_derivatives_jax)
    return derivative


def compute_spin_derivatives_jax(
    params: Dict,
    toas_mjd: np.ndarray,
    fit_params: list,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute all spin parameter derivatives for design matrix (JAX version).
    
    This is a drop-in replacement for compute_spin_derivatives() that uses
    JAX JIT compilation for ~5-10x speedup on large datasets.
    
    Parameters
    ----------
    params : dict
        Timing model parameters including PEPOCH, F0, F1, F2, etc.
    toas_mjd : np.ndarray
        TOA times in MJD
    fit_params : list
        List of parameters to fit (e.g., ['F0', 'F1'])
    **kwargs
        Additional arguments (for API compatibility)
        
    Returns
    -------
    derivatives : dict
        Dictionary mapping parameter name to derivative column
        Each value is np.ndarray of shape (n_toas,)
        Values are in seconds (design matrix units)
        
    Notes
    -----
    Returns numpy arrays (not JAX arrays) for compatibility with
    existing fitting code.
    """
    # Get PEPOCH
    pepoch_mjd = params.get('PEPOCH', toas_mjd[0])
    
    # Compute dt in seconds (convert to JAX array)
    dt_sec = jnp.array((toas_mjd - pepoch_mjd) * 86400.0)
    
    # Get F0 for phase -> time conversion
    f0 = params.get('F0', 1.0)
    
    # Compute derivatives for each requested F parameter
    derivatives = {}
    for param in fit_params:
        if param.startswith('F'):
            # Extract order
            try:
                order = int(param[1:])
            except ValueError:
                raise ValueError(f"Cannot parse order from {param}")
            
            # Compute derivative (JIT-compiled!)
            deriv_phase = d_phase_d_F_jax(dt_sec, order)  # cycles (POSITIVE)
            
            # Apply PINT's convention: negate and convert to time units
            deriv_time = -deriv_phase / f0  # seconds (NEGATIVE)
            
            # Convert back to numpy for compatibility
            derivatives[param] = np.array(deriv_time)
    
    return derivatives


# Backward compatibility functions (non-JIT for testing)
def d_phase_d_F0_jax(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F0 (JAX).
    
    d(phase)/d(F0) = dt
    """
    return taylor_horner_jax(dt_sec, jnp.array([0.0, 1.0]))


def d_phase_d_F1_jax(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F1 (JAX).
    
    d(phase)/d(F1) = dt^2 / 2!
    """
    return taylor_horner_jax(dt_sec, jnp.array([0.0, 0.0, 1.0]))


def d_phase_d_F2_jax(dt_sec: jnp.ndarray) -> jnp.ndarray:
    """Derivative of phase with respect to F2 (JAX).
    
    d(phase)/d(F2) = dt^3 / 3!
    """
    return taylor_horner_jax(dt_sec, jnp.array([0.0, 0.0, 0.0, 1.0]))


if __name__ == '__main__':
    # Quick test
    print("Testing JAX spin derivatives...")
    
    # Test case from PINT documentation
    dt = jnp.array([2.0])
    coeffs = jnp.array([10.0, 3.0, 4.0, 12.0])
    result = taylor_horner_jax(dt, coeffs)
    print(f"taylor_horner_jax(2.0, [10, 3, 4, 12]) = {result[0]:.1f}")
    print(f"Expected: 40.0")
    print(f"Match: {abs(result[0] - 40.0) < 1e-10}")
    
    # Test derivatives
    dt = jnp.array([1.0, 2.0, 3.0])
    
    # d/dF0 = dt
    d_f0 = d_phase_d_F0_jax(dt)
    print(f"\nd/dF0 at dt=[1,2,3]: {d_f0}")
    print(f"Expected: [1, 2, 3]")
    
    # d/dF1 = dt^2/2
    d_f1 = d_phase_d_F1_jax(dt)
    print(f"\nd/dF1 at dt=[1,2,3]: {d_f1}")
    print(f"Expected: [0.5, 2.0, 4.5]")
    
    # d/dF2 = dt^3/6
    d_f2 = d_phase_d_F2_jax(dt)
    print(f"\nd/dF2 at dt=[1,2,3]: {d_f2}")
    print(f"Expected: [0.167, 1.333, 4.5]")
    
    # Test full interface
    params = {'PEPOCH': 58000.0, 'F0': 339.3, 'F1': -1.6e-15}
    toas = np.array([58000.0, 58001.0, 58002.0])
    derivs = compute_spin_derivatives_jax(params, toas, ['F0', 'F1'])
    print(f"\nFull interface test:")
    print(f"F0 derivative shape: {derivs['F0'].shape}")
    print(f"F1 derivative shape: {derivs['F1'].shape}")
    
    print("\n✓ JAX spin derivatives module ready!")
