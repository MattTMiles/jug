"""Core JAX-compatible residual computation.

This module provides pure JAX functions for computing residuals,
suitable for fitting and differentiation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

from jug.delays.combined import compute_total_delay_jax
from jug.utils.constants import SECS_PER_DAY


@dataclass
class PrecomputedData:
    """Precomputed data for residual calculation.
    
    All the expensive operations (file I/O, ephemeris, clock corrections)
    are done once and stored here. The residual function can then be called
    many times with different parameters without repeating this work.
    """
    # TOA data
    tdb_mjd: np.ndarray           # TDB times (MJD)
    freq_mhz: np.ndarray          # Observing frequencies (MHz)
    errors_us: np.ndarray         # TOA uncertainties (μs)
    n_toas: int
    
    # Barycentric delays (precomputed, don't depend on fitted params)
    roemer_shapiro_sec: np.ndarray  # Roemer + Shapiro delay (seconds)
    
    # Observatory/ephemeris data
    obs_sun_km: np.ndarray        # Sun position vectors
    
    # Binary orbital data (if needed)
    has_binary: bool
    
    # Fixed parameters for delay computation
    # These are parameters we're NOT fitting
    dm_coeffs: jnp.ndarray
    dm_factorials: jnp.ndarray
    dm_epoch: float
    ne_sw: float
    fd_coeffs: jnp.ndarray
    has_fd: bool
    
    # Reference epoch
    pepoch_sec: float
    
    # TZR information
    tzrmjd_tdb: float
    tzr_phase: float


def compute_residuals_jax(
    params_dict: Dict[str, float],
    data: PrecomputedData
) -> jnp.ndarray:
    """Compute timing residuals with given parameters (pure JAX).
    
    This function is JAX-compatible and can be JIT-compiled and
    differentiated. All expensive precomputations are in `data`.
    
    Parameters
    ----------
    params_dict : dict
        Timing model parameters (F0, F1, F2, binary params, etc.)
    data : PrecomputedData
        Precomputed data from prepare_fitting_data()
        
    Returns
    -------
    residuals_us : jax array
        Timing residuals in microseconds
    """
    # Extract spin parameters
    F0 = params_dict.get('F0', 0.0)
    F1 = params_dict.get('F1', 0.0)
    F2 = params_dict.get('F2', 0.0)
    
    # Precompute coefficients
    F1_half = F1 / 2.0
    F2_sixth = F2 / 6.0
    
    # Convert to JAX arrays
    tdb_mjd_jax = jnp.array(data.tdb_mjd)
    freq_mhz_jax = jnp.array(data.freq_mhz)
    roemer_shapiro_jax = jnp.array(data.roemer_shapiro_sec)
    
    # Extract binary parameters if needed
    if data.has_binary:
        PB = params_dict.get('PB', 0.0)
        A1 = params_dict.get('A1', 0.0)
        TASC = params_dict.get('TASC', 0.0)
        EPS1 = params_dict.get('EPS1', 0.0)
        EPS2 = params_dict.get('EPS2', 0.0)
        PBDOT = params_dict.get('PBDOT', 0.0)
        XDOT = params_dict.get('XDOT', 0.0)
        
        # Shapiro parameters
        M2 = params_dict.get('M2', 0.0)
        SINI = params_dict.get('SINI', 0.0)
        # Convert M2/SINI to r/s if needed
        from jug.utils.constants import T_SUN_SEC
        r_shap = T_SUN_SEC * M2 if M2 > 0 else params_dict.get('H3', 0.0)
        s_shap = SINI if SINI > 0 else params_dict.get('STIG', 0.0)
        
        gamma = params_dict.get('GAMMA', 0.0)
    else:
        PB = A1 = TASC = EPS1 = EPS2 = 0.0
        PBDOT = XDOT = gamma = r_shap = s_shap = 0.0
    
    # Convert to JAX
    pb_jax = jnp.array(PB)
    a1_jax = jnp.array(A1)
    tasc_jax = jnp.array(TASC)
    eps1_jax = jnp.array(EPS1)
    eps2_jax = jnp.array(EPS2)
    pbdot_jax = jnp.array(PBDOT)
    xdot_jax = jnp.array(XDOT)
    gamma_jax = jnp.array(gamma)
    r_shap_jax = jnp.array(r_shap)
    s_shap_jax = jnp.array(s_shap)
    
    # Compute other delays using JAX kernel
    other_delays_sec = compute_total_delay_jax(
        tdb_mjd_jax, freq_mhz_jax,
        data.obs_sun_km, jnp.zeros(3),  # L_hat not used in current implementation
        data.dm_coeffs, data.dm_factorials, jnp.array(data.dm_epoch),
        jnp.array(data.ne_sw),
        data.fd_coeffs, jnp.array(data.has_fd),
        roemer_shapiro_jax, jnp.array(data.has_binary),
        pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax,
        pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax
    )
    
    # Total delay
    total_delay_sec = other_delays_sec
    
    # Compute phase
    dt_sec = tdb_mjd_jax * SECS_PER_DAY - data.pepoch_sec - total_delay_sec
    phase = F0 * dt_sec + F1_half * dt_sec**2 + F2_sixth * dt_sec**3
    
    # Wrap phase relative to TZR
    frac_phase = jnp.mod(phase - data.tzr_phase + 0.5, 1.0) - 0.5
    
    # Convert to microseconds
    residuals_us = frac_phase / F0 * 1e6
    
    # Subtract weighted mean
    weights = 1.0 / (data.errors_us ** 2)
    weighted_mean = jnp.sum(residuals_us * weights) / jnp.sum(weights)
    residuals_us = residuals_us - weighted_mean
    
    return residuals_us


# JIT-compile for speed
compute_residuals_jax = jax.jit(compute_residuals_jax, static_argnames=['data'])
