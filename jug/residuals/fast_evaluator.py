"""
Fast residual re-evaluation using cached delays.

This module provides ultra-fast residual computation by reusing
expensive delay calculations (TOA parsing, clocks, TDB, Shapiro, etc.)
and only re-evaluating the timing model with new parameters.
"""
import numpy as np
import jax.numpy as jnp
from jax import jit


@jit
def evaluate_timing_model_jax(
    dt: jnp.ndarray,
    f0: float,
    f1: float = 0.0,
    f2: float = 0.0
) -> jnp.ndarray:
    """
    Evaluate timing model with given parameters (JAX JIT-compiled).
    
    Parameters
    ----------
    dt : jnp.ndarray
        Time since PEPOCH in seconds
    f0 : float
        Spin frequency (Hz)
    f1 : float, optional
        First frequency derivative (Hz/s)
    f2 : float, optional
        Second frequency derivative (Hz/s^2)
    
    Returns
    -------
    phases : jnp.ndarray
        Predicted pulse phases at each TOA
    """
    # Taylor series for phase (always compute all terms, JAX will optimize)
    phase = f0 * dt + 0.5 * f1 * dt * dt + (1.0 / 6.0) * f2 * dt * dt * dt
    
    return phase


def compute_residuals_fast_v2(
    toa_data: dict,
    params: dict,
    subtract_mean: bool = True
) -> tuple[np.ndarray, float]:
    """
    Ultra-fast residual computation using cached TOA data.
    
    This skips all parsing and delay computation, only evaluates timing model.
    
    Parameters
    ----------
    toa_data : dict
        Cached TOA data with keys 'dt_sec' (time at pulsar minus PEPOCH), 'errors_us'
    params : dict
        Full parameter set (merged original + overrides)
    subtract_mean : bool, optional
        Whether to subtract weighted mean from residuals (default: True)
    
    Returns
    -------
    residuals_us : np.ndarray
        Residuals in microseconds
    rms_us : float
        RMS in microseconds
        
    Notes
    -----
    This is ~1000x faster than full recomputation because it:
    - Skips .par and .tim file parsing
    - Skips clock file loading
    - Skips UTC->TDB conversion
    - Skips most barycentric delay calculations
    - Only re-evaluates timing model with new parameters
    
    Accuracy: Exact match with slow path (no approximations)
    
    Algorithm:
    1. Check if DM params changed - if so, recompute DM delay
    2. dt_sec = tdb_sec - PEPOCH_sec - delay_sec (cached or adjusted!)
    3. phase = F0*dt + 0.5*F1*dt^2 + (1/6)*F2*dt^3
    4. frac_phase = phase - round(phase)  # Wrap to nearest pulse
    5. residuals = frac_phase / F0 * 1e6
    6. residuals -= weighted_mean (if subtract_mean=True)
    """
    # Extract timing parameters
    f0 = params.get('F0', 0.0)
    f1 = params.get('F1', 0.0)
    f2 = params.get('F2', 0.0)
    dm = params.get('DM', 0.0)
    dm1 = params.get('DM1', 0.0)
    dm2 = params.get('DM2', 0.0)
    
    # Get cached dt_sec (time at pulsar minus PEPOCH, delay-corrected)
    # NOTE: dt_sec includes dispersion delay with ORIGINAL DM parameters
    # We need to adjust if DM parameters changed
    dt_sec = toa_data['dt_sec'].copy()
    
    # Check if DM parameters changed from original
    if 'original_dm_params' in toa_data:
        orig_dm = toa_data['original_dm_params'].get('DM', 0.0)
        orig_dm1 = toa_data['original_dm_params'].get('DM1', 0.0)
        orig_dm2 = toa_data['original_dm_params'].get('DM2', 0.0)
        
        # If DM params changed, adjust dt_sec by removing old dispersion delay and adding new
        dm_changed = (dm != orig_dm or dm1 != orig_dm1 or dm2 != orig_dm2)
        
        if dm_changed:
            # Get frequency data
            freq_mhz = toa_data.get('freq_mhz')
            tdb_mjd = toa_data.get('tdb_mjd')
            pepoch = params.get('PEPOCH')
            dmepoch = params.get('DMEPOCH', pepoch)  # Default to PEPOCH if no DMEPOCH
            
            if freq_mhz is not None and tdb_mjd is not None and dmepoch is not None:
                # Use fast DM delay computation
                from jug.fitting.optimized_fitter import compute_dm_delay_fast
                
                # Original DM delay
                orig_dm_params = {'DM': orig_dm, 'DM1': orig_dm1, 'DM2': orig_dm2}
                delay_orig = compute_dm_delay_fast(tdb_mjd, freq_mhz, orig_dm_params, dmepoch)
                
                # New DM delay
                new_dm_params = {'DM': dm, 'DM1': dm1, 'DM2': dm2}
                delay_new = compute_dm_delay_fast(tdb_mjd, freq_mhz, new_dm_params, dmepoch)
                
                # Adjust dt_sec: remove old delay, add new delay
                # dt_sec = tdb - pepoch - delays
                # We want: dt_sec_new = tdb - pepoch - (other_delays + delay_new)
                #                     = (tdb - pepoch - other_delays - delay_orig) + delay_orig - delay_new
                #                     = dt_sec_old + delay_orig - delay_new
                dt_sec = dt_sec + delay_orig - delay_new
    
    # Evaluate timing model (same as simple_calculator.py line 507)
    f1_half = f1 / 2.0
    f2_sixth = f2 / 6.0
    phase = dt_sec * (f0 + dt_sec * (f1_half + dt_sec * f2_sixth))
    
    # Wrap phase to nearest integer pulse (PINT-style)
    frac_phase = phase - np.round(phase)
    
    # Convert to microseconds
    residuals_us = (frac_phase / f0) * 1e6
    
    # Subtract WEIGHTED mean if requested (same as simple_calculator.py line 646-651)
    if subtract_mean and 'errors_us' in toa_data:
        errors_us = toa_data['errors_us']
        if errors_us is not None:
            weights = 1.0 / (errors_us ** 2)
            weighted_mean = np.sum(residuals_us * weights) / np.sum(weights)
            residuals_us = residuals_us - weighted_mean
    
    # Compute weighted RMS (same as simple_calculator.py line 659)
    if 'errors_us' in toa_data and toa_data['errors_us'] is not None:
        errors_us = toa_data['errors_us']
        weights = 1.0 / (errors_us ** 2)
        rms_us = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    else:
        # Fallback to unweighted if no errors
        rms_us = np.sqrt(np.mean(residuals_us ** 2))
    
    return residuals_us, rms_us
