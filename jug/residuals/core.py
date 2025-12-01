"""Core JAX-compatible residual computation for fitting.

This module provides pure JAX functions for computing residuals that can be
used with automatic differentiation and JIT compilation. Unlike simple_calculator.py,
these functions have no file I/O and are designed for use in optimization loops.
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
from typing import Dict

# Enable float64 precision for JAX (critical for pulsar timing!)
jax.config.update('jax_enable_x64', True)

from jug.utils.constants import SECS_PER_DAY, K_DM_SEC


@jax.jit
def spin_phase_jax(dt_sec, f0, f1, f2, f3):
    """Compute spin phase using polynomial expansion.
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time difference from PEPOCH in seconds
    f0, f1, f2, f3 : float
        Spin frequency and derivatives
    
    Returns
    -------
    jnp.ndarray
        Phase in cycles
    """
    # Horner's method: phase = dt * (F0 + dt * (F1/2 + dt * (F2/6 + dt * F3/24)))
    f1_half = f1 / 2.0
    f2_sixth = f2 / 6.0
    f3_24th = f3 / 24.0
    
    phase = dt_sec * (f0 + dt_sec * (f1_half + dt_sec * (f2_sixth + dt_sec * f3_24th)))
    return phase


@jax.jit
def dm_delay_jax(freq_mhz, dm_coeffs, dm_factorials, dm_epoch_mjd, tdb_mjd):
    """Compute DM delay with polynomial expansion.
    
    Parameters
    ----------
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz
    dm_coeffs : jnp.ndarray
        DM polynomial coefficients [DM, DM1, DM2, ...]
    dm_factorials : jnp.ndarray
        Factorials [1, 1, 2, 6, 24, ...]
    dm_epoch_mjd : float
        DM reference epoch in MJD
    tdb_mjd : jnp.ndarray
        Barycentric times in MJD
    
    Returns
    -------
    jnp.ndarray
        DM delay in seconds
    """
    # DM polynomial: DM(t) = sum(DM_i * (t-DMEPOCH)^i / i!)
    dt_years = (tdb_mjd - dm_epoch_mjd) / 365.25
    powers = jnp.arange(len(dm_coeffs))
    dt_powers = dt_years[:, jnp.newaxis] ** powers[jnp.newaxis, :]
    dm_eff = jnp.sum(dm_coeffs * dt_powers / dm_factorials, axis=1)
    
    # DM delay: K_DM * DM / freq^2
    dm_delay_sec = K_DM_SEC * dm_eff / (freq_mhz ** 2)
    return dm_delay_sec


@jax.jit
def _compute_residuals_from_dt(
    dt_sec,
    tzr_phase,
    f0, f1, f2, f3,
    uncertainties_us
):
    """Compute residuals from emission times (dt_sec).
    
    This version uses pre-computed emission times from simple_calculator
    (computed with longdouble precision) to avoid precision loss from
    reconstructing dt_sec from float64 delays.
    
    NOTE: This does NOT subtract weighted mean - that should be done outside
    to avoid coupling in the Jacobian!
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Emission times (TDB - PEPOCH - delays) in seconds, from simple_calculator
    tzr_phase : float
        TZR phase offset in cycles
    f0, f1, f2, f3 : float
        Spin frequency and derivatives
    uncertainties_us : jnp.ndarray
        TOA uncertainties in microseconds
        
    Returns
    -------
    jnp.ndarray
        Residuals in seconds (NOT weighted-mean subtracted!)
    """
    # Model phase using Horner's method (same as simple_calculator)
    f1_half = f1 / 2.0
    f2_sixth = f2 / 6.0
    f3_24th = f3 / 24.0
    model_phase = dt_sec * (f0 + dt_sec * (f1_half + dt_sec * (f2_sixth + dt_sec * f3_24th)))
    
    # Phase wrapping
    frac_phase = jnp.mod(model_phase - tzr_phase + 0.5, 1.0) - 0.5
    
    # Convert to seconds (NOT microseconds, and NOT mean-subtracted!)
    time_residuals_sec = frac_phase / f0
    
    return time_residuals_sec


@jax.jit
def _compute_residuals_core(
    tdb_mjd,
    freq_mhz,
    geometric_delay_sec,
    other_delays_minus_dm_sec,
    pepoch_mjd,
    dm_epoch_mjd,
    tzr_phase,
    f0, f1, f2, f3,
    dm_coeffs,
    dm_factorials,
    uncertainties_us
):
    """Core residual computation (pure JAX, JIT-compiled).
    
    This is the inner loop that gets JIT-compiled. All parameter extraction
    and dict manipulation happens outside.
    
    Parameters
    ----------
    geometric_delay_sec : jnp.ndarray
        Roemer + Shapiro delays (pre-computed)
    other_delays_minus_dm_sec : jnp.ndarray
        SW + FD + binary delays (pre-computed, minus reference DM)
    uncertainties_us : jnp.ndarray
        TOA uncertainties in microseconds (for weighted mean)
    """
    # Compute DM delay with current parameters
    dm_delay_sec = dm_delay_jax(freq_mhz, dm_coeffs, dm_factorials, dm_epoch_mjd, tdb_mjd)
    
    # Total delay = geometric + (other - DM_ref) + DM_new
    # where other_delays_minus_dm already has DM_ref subtracted
    total_delay_sec = geometric_delay_sec + other_delays_minus_dm_sec + dm_delay_sec
    
    # Time at emission
    pepoch_sec = pepoch_mjd * SECS_PER_DAY
    tdb_sec = tdb_mjd * SECS_PER_DAY
    dt_sec = tdb_sec - pepoch_sec - total_delay_sec
    
    # Model phase
    model_phase = spin_phase_jax(dt_sec, f0, f1, f2, f3)
    
    # Residuals as phase difference converted to time
    # Use same formula as simple_calculator:
    # frac_phase = mod(phase - tzr_phase + 0.5, 1.0) - 0.5
    frac_phase = jnp.mod(model_phase - tzr_phase + 0.5, 1.0) - 0.5
    
    # Convert to microseconds BEFORE subtracting weighted mean (matches simple_calculator)
    time_residuals_us = (frac_phase / f0) * 1e6
    
    # Subtract weighted mean (matching PINT/simple_calculator behavior)
    # Note: weights use uncertainties_us directly (not converted to seconds)
    weights = 1.0 / (uncertainties_us ** 2)
    weighted_mean = jnp.sum(time_residuals_us * weights) / jnp.sum(weights)
    time_residuals_us = time_residuals_us - weighted_mean
    
    # Convert back to seconds for return
    time_residuals_sec = time_residuals_us * 1e-6
    
    return time_residuals_sec


def compute_residuals_jax_from_dt(
    params_array,
    param_names,
    dt_sec,
    tzr_phase,
    uncertainties_us,
    fixed_params
):
    """Compute residuals from emission times (using longdouble precision from simple_calculator).
    
    This is the preferred method as it preserves the precision of simple_calculator's
    longdouble computation.
    
    Parameters
    ----------
    params_array : array
        Values of parameters being fitted (F0, F1, etc.)
    param_names : tuple
        Names of fitted parameters
    dt_sec : jnp.ndarray
        Emission times from simple_calculator (TDB - PEPOCH - delays)
    tzr_phase : float
        TZR phase offset
    uncertainties_us : jnp.ndarray
        TOA uncertainties
    fixed_params : dict
        Parameters not being fitted
        
    Returns
    -------
    jnp.ndarray
        Residuals in seconds
    """
    # Build parameter dict (avoid float() which breaks autodiff)
    params = dict(fixed_params)
    for i, name in enumerate(param_names):
        params[name] = params_array[i]  # Keep as JAX array
    
    # Extract spin parameters (keep as JAX arrays for autodiff)
    f0 = params.get('F0', 0.0)
    f1 = params.get('F1', 0.0)
    f2 = params.get('F2', 0.0)
    f3 = params.get('F3', 0.0)
    
    # Call JIT-compiled function
    return _compute_residuals_from_dt(
        dt_sec, tzr_phase, f0, f1, f2, f3, uncertainties_us
    )


def compute_residuals_jax(
    params_array,
    param_names,
    tdb_mjd,
    freq_mhz,
    geometric_delay_sec,
    other_delays_minus_dm_sec,
    pepoch_mjd,
    dm_epoch_mjd,
    tzr_phase,
    uncertainties_us,
    fixed_params
):
    """Compute timing residuals using JAX (wrapper for JIT function).
    
    This function extracts parameters from the array and calls the JIT-compiled
    core function. All dict manipulation happens here (outside JIT).
    
    Parameters
    ----------
    params_array : jnp.ndarray or np.ndarray
        Array of parameter values being fitted (e.g., [F0, F1, DM])
    param_names : tuple of str
        Names of parameters in params_array
    tdb_mjd : jnp.ndarray
        Barycentric times (MJD)
    freq_mhz : jnp.ndarray
        Observing frequencies (MHz)
    geometric_delay_sec : jnp.ndarray
        Pre-computed Roemer + Shapiro delays (seconds)
    other_delays_minus_dm_sec : jnp.ndarray
        Pre-computed SW + FD + binary delays minus reference DM (seconds)
    pepoch_mjd : float
        Reference epoch (MJD)
    dm_epoch_mjd : float
        DM reference epoch (MJD)
    tzr_phase : float
        TZR phase offset (cycles)
    uncertainties_us : jnp.ndarray
        TOA uncertainties in microseconds
    fixed_params : dict
        Fixed parameters not being fitted
    
    Returns
    -------
    jnp.ndarray
        Residuals in seconds
    """
    # Build parameter dict from array
    # CRITICAL: Do NOT use float() here - it breaks JAX autodiff!
    # Keep values as JAX arrays to preserve gradient information
    params = dict(fixed_params)
    for i, name in enumerate(param_names):
        params[name] = params_array[i]  # Keep as JAX array for autodiff
    
    # Extract spin parameters
    f0 = params.get('F0', 0.0)
    f1 = params.get('F1', 0.0)
    f2 = params.get('F2', 0.0)
    f3 = params.get('F3', 0.0)
    
    # Extract DM parameters
    dm_coeffs = []
    k = 0
    while True:
        key = 'DM' if k == 0 else f'DM{k}'
        if key in params:
            dm_coeffs.append(params[key])
            k += 1
        else:
            break
    dm_coeffs = jnp.array(dm_coeffs if dm_coeffs else [0.0])
    dm_factorials = jnp.array([float(math.factorial(i)) for i in range(len(dm_coeffs))])
    
    # Call JIT-compiled core
    return _compute_residuals_core(
        tdb_mjd, freq_mhz, geometric_delay_sec, other_delays_minus_dm_sec,
        pepoch_mjd, dm_epoch_mjd, tzr_phase,
        f0, f1, f2, f3,
        dm_coeffs, dm_factorials,
        uncertainties_us
    )


def prepare_fixed_data(par_file, tim_file, clock_dir="data/clock", observatory="meerkat"):
    """Pre-compute all data that doesn't change during fitting.
    
    This function performs all the expensive operations once:
    - File parsing
    - TDB conversion
    - Geometric delay computation (Roemer + Shapiro)
    - Binary delay computation
    - DM delay computation (at reference parameters)
    
    For fitting, only spin parameters (F0, F1, F2, F3) and DM parameters will be varied.
    Binary parameters are assumed fixed for now.
    
    Parameters
    ----------
    par_file : Path or str
        Path to .par file
    tim_file : Path or str
        Path to .tim file
    clock_dir : Path or str
        Directory containing clock files
    observatory : str
        Observatory name
    
    Returns
    -------
    dict
        Dictionary containing:
        - All JAX arrays needed for residual computation
        - All fixed parameters from .par file
        - Pre-computed delays
    
    Notes
    -----
    Currently assumes binary parameters are NOT being fitted.
    To fit binary parameters, would need to modify this function.
    """
    from jug.io.par_reader import parse_par_file, get_longdouble, parse_ra, parse_dec
    from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
    from jug.io.clock import parse_clock_file
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.delays.barycentric import (
        compute_ssb_obs_pos_vel,
        compute_pulsar_direction,
        compute_roemer_delay,
        compute_shapiro_delay,
        compute_barycentric_freq
    )
    from jug.utils.constants import T_SUN_SEC, T_PLANET, OBSERVATORIES
    from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris
    from astropy.time import Time
    from astropy import units as u
    
    print("\n" + "="*60)
    print("Preparing fixed data for JAX fitting...")
    print("="*60)
    
    # Use simple_calculator to get all delays computed correctly
    print(f"\n1. Using simple_calculator to compute all delays...")
    result = compute_residuals_simple(par_file, tim_file, clock_dir, observatory)
    
    # Extract what we need from simple_calculator
    par_params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    
    # Get TDB times from result
    tdb_mjd = result['tdb_mjd']
    
    # Get barycentric frequencies from result (don't recompute!)
    freq_bary_mhz = result['freq_bary_mhz']
    
    # Get total delay from result (don't recompute!)
    total_delay_sec = result['total_delay_sec']
    
    # Get uncertainties
    uncertainties_us = result['errors_us']
    
    # Get emission time (dt_sec) computed with longdouble precision
    dt_sec = result['dt_sec']
    
    print(f"\n2. Using emission times from simple_calculator (computed with longdouble)...")
    print(f"   Mean emission time: {np.mean(dt_sec):.6f} s")
    
    # Prepare arrays for JAX
    tdb_jax = jnp.array(tdb_mjd, dtype=jnp.float64)
    freq_bary_jax = jnp.array(freq_bary_mhz, dtype=jnp.float64)
    
    # Extract DM parameters
    dm_coeffs = []
    k = 0
    while True:
        key = 'DM' if k == 0 else f'DM{k}'
        if key in par_params:
            dm_coeffs.append(float(par_params[key]))
            k += 1
        else:
            break
    dm_coeffs_jax = jnp.array(dm_coeffs if dm_coeffs else [0.0])
    dm_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(dm_coeffs))])
    dm_epoch_jax = float(par_params.get('DMEPOCH', par_params['PEPOCH']))
    
    # Compute DM delay at reference parameters
    dm_delay_ref_sec = np.array(dm_delay_jax(
        freq_bary_jax, dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax, tdb_jax
    ), dtype=np.float64)
    
    print(f"   Mean DM delay (ref): {np.mean(dm_delay_ref_sec):.6f} s")
    
    # Store total_delay_minus_dm so JAX can add DM_new during fitting
    # total_delay = (total_delay_minus_dm) + DM_new
    total_delay_minus_dm = total_delay_sec - dm_delay_ref_sec
    
    print(f"   Mean delay (minus DM): {np.mean(total_delay_minus_dm):.6f} s")
    
    # Compute TZR phase - use the EXACT value from simple_calculator
    tzr_phase = result.get('tzr_phase', 0.0)
    if 'TZRMJD' in par_params:
        print(f"\n3. TZR phase from simple_calculator: {tzr_phase:.6f} cycles")
    
    print(f"\n✅ Fixed data prepared for {len(tdb_mjd)} TOAs")
    print("="*60 + "\n")
    
    # Build fixed data dict
    fixed_data = {
        # JAX arrays
        'tdb_mjd': jnp.array(tdb_mjd, dtype=jnp.float64),
        'freq_mhz': jnp.array(freq_bary_mhz, dtype=jnp.float64),
        'dt_sec': jnp.array(dt_sec, dtype=jnp.float64),  # Emission times from simple_calculator (longdouble->float64)
        'geometric_delay_sec': jnp.array(total_delay_minus_dm, dtype=jnp.float64),  # For old interface
        'other_delays_minus_dm_sec': jnp.array(np.zeros(len(tdb_mjd)), dtype=jnp.float64),  # For old interface
        'uncertainties_us': jnp.array(uncertainties_us, dtype=jnp.float64),
        
        # Scalars
        'pepoch': float(par_params['PEPOCH']),
        'dm_epoch': float(par_params.get('DMEPOCH', par_params['PEPOCH'])),
        'tzr_phase': float(tzr_phase),
        'n_toas': len(tdb_mjd),
        
        # All parameters from .par file (as regular Python types for fixed_params dict)
        'par_params': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in par_params.items()}
    }
    
    return fixed_data
