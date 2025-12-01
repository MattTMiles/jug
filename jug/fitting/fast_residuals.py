"""
Ultra-fast residual computation using cached delays.

This module provides JAX-compiled residual computation that only
recomputes the spin phase, with all other delays pre-cached.

Key optimization: Separate static (clock, bary, binary) from dynamic (phase).
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, Tuple

# Enable JAX 64-bit precision
jax.config.update("jax_enable_x64", True)

from jug.utils.constants import SECS_PER_DAY


@jax.jit
def compute_phase_residuals_jax(
    toas_mjd: jnp.ndarray,
    t_emission_mjd: jnp.ndarray,
    f0: float,
    f1: float,
    pepoch: float,
    weights: jnp.ndarray
) -> Tuple[jnp.ndarray, float]:
    """
    Fast phase residual computation (JAX JIT-compiled).
    
    This only recomputes the spin phase using pre-cached emission times.
    Everything else (clock, barycentric, binary delays) is pre-computed
    and baked into t_emission_mjd.
    
    Parameters
    ----------
    toas_mjd : jnp.ndarray
        Original TOA times in MJD
    t_emission_mjd : jnp.ndarray
        Emission times in MJD (includes all delays except spin phase)
    f0 : float
        Spin frequency (Hz)
    f1 : float  
        Spin frequency derivative (Hz/s)
    pepoch : float
        Reference epoch (MJD)
    weights : jnp.ndarray
        TOA weights (1/error^2)
        
    Returns
    -------
    residuals_sec : jnp.ndarray
        Timing residuals in seconds
    rms_us : float
        Weighted RMS in microseconds
    """
    # Time since PEPOCH in seconds
    dt_sec = (t_emission_mjd - pepoch) * SECS_PER_DAY
    
    # Compute spin phase (cycles)
    phase_cycles = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    
    # Convert phase to time residuals (seconds)
    # residual = TOA - phase/f0
    model_toa_mjd = pepoch + phase_cycles / (f0 * SECS_PER_DAY)
    residuals_mjd = toas_mjd - model_toa_mjd
    residuals_sec = residuals_mjd * SECS_PER_DAY
    
    # Subtract weighted mean
    mean_residual = jnp.sum(residuals_sec * weights) / jnp.sum(weights)
    residuals_sec = residuals_sec - mean_residual
    
    # Compute weighted RMS
    rms_sec = jnp.sqrt(jnp.sum(residuals_sec**2 * weights) / jnp.sum(weights))
    rms_us = rms_sec * 1e6
    
    return residuals_sec, rms_us


def compute_emission_times_once(
    par_file,
    tim_file,
    clock_dir="data/clock"
):
    """
    Compute emission times ONCE - these are static for spin fitting.
    
    This function computes all the expensive delays (clock, barycentric, binary)
    that don't change when fitting F0/F1. The result is cached and reused.
    
    Returns
    -------
    dict with keys:
        - toas_mjd: Original TOA times
        - t_emission_mjd: Emission times (with all delays)
        - errors_sec: TOA errors in seconds
        - weights: TOA weights (1/error^2)
        - pepoch: Reference epoch
        - f0_ref: Reference F0
        - f1_ref: Reference F1
    """
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    
    print("Computing static delays (this happens ONCE)...")
    
    # Parse files
    params = parse_par_file(par_file)
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract data
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    
    # Compute full residuals to get intermediate products
    # We need to extract the emission times from the calculation
    # For now, use a hack: compute with reference F0/F1, then back out emission times
    
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = compute_residuals_simple(
            par_file, tim_file,
            clock_dir=clock_dir,
            subtract_tzr=False  # Don't subtract mean yet
        )
    
    # The residuals are: TOA - phase/f0
    # So: phase/f0 = TOA - residual
    # And: emission_time = pepoch + phase/f0 
    residuals_sec = result['residuals_us'] * 1e-6
    f0_ref = params['F0']
    f1_ref = params.get('F1', 0.0)
    pepoch = params['PEPOCH']
    
    # Back-calculate emission times
    # residual = toa - (pepoch + phase/f0)
    # => pepoch + phase/f0 = toa - residual
    model_toa_mjd = toas_mjd - residuals_sec / SECS_PER_DAY
    
    # Now we need to back out the emission time (before DM delay was added)
    # For spin-only fitting, the emission time includes all delays
    # This is actually model_toa_mjd rearranged
    
    # Actually, let's use TDB times from result if available
    if 'tdb_mjd' in result:
        # Use barycentric times as emission times
        # (this includes clock + bary delays but not phase)
        t_emission_mjd = result['tdb_mjd']
    else:
        # Fall back to approximation
        t_emission_mjd = model_toa_mjd
    
    print(f"  Cached {len(toas_mjd)} emission times")
    print(f"  Reference F0 = {f0_ref:.10f} Hz")
    print(f"  Reference F1 = {f1_ref:.10e} Hz/s")
    
    return {
        'toas_mjd': toas_mjd,
        't_emission_mjd': t_emission_mjd,
        'errors_sec': errors_sec,
        'weights': weights,
        'pepoch': pepoch,
        'f0_ref': f0_ref,
        'f1_ref': f1_ref
    }
