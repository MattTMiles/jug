"""Analytical derivatives for DM parameters (DM, DM1, DM2, ...).

This module implements PINT-compatible analytical derivatives for
dispersion measure parameters. The formulas follow PINT's dispersion.py.

DM affects timing through cold-plasma dispersion delay:
    tau_DM = K_DM * DM / freq^2

where K_DM = 1/2.41e-4 ~= 4149.378 MHz^2 pc^-^1 cm^3 s

DM can vary with time as a polynomial:
    DM(t) = DM + DM1*t + DM2*t^2/2 + ...

Reference: PINT src/pint/models/dispersion_model.py
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import math
import numpy as np
from typing import Dict

from jug.utils.constants import K_DM_SEC, SECS_PER_DAY, SECS_PER_YEAR


@jax.jit
def d_delay_d_DM(freq_mhz: jnp.ndarray) -> jnp.ndarray:
    """Compute derivative of dispersion delay with respect to DM.
    
    The dispersion delay is:
        tau_DM = K_DM * DM / freq^2
    
    Therefore:
        dtau/dDM = K_DM / freq^2
    
    Parameters
    ----------
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz, shape (n_toas,)
        
    Returns
    -------
    derivative : jnp.ndarray
        d(delay)/d(DM) in units of seconds/(pc cm^-^3)
        Shape (n_toas,)
        
    Notes
    -----
    - Derivative is POSITIVE: increasing DM increases delay
    - Already in time units (seconds), no F0 conversion needed
    - Frequency dependent: lower freq -> larger derivative
    
    Example
    -------
    >>> freq = np.array([1400.0, 700.0])  # MHz
    >>> deriv = d_delay_d_DM(freq)
    >>> print(deriv)
    [2.12e-03  8.47e-03]  # Lower freq has 4* larger derivative
    """
    return K_DM_SEC / (freq_mhz ** 2)


@jax.jit
def d_delay_d_DM1(dt_sec: jnp.ndarray, freq_mhz: jnp.ndarray) -> jnp.ndarray:
    """Compute derivative of dispersion delay with respect to DM1.

    DM1 represents linear DM evolution in pc cm^-^3 yr^-^1:
        DM(t) = DM + DM1 * t_yr

    The delay contribution from DM1 is:
        tau_DM1 = K_DM * DM1 * t_yr / freq^2

    Therefore:
        dtau/dDM1 = K_DM * t_yr / freq^2

    where t_yr is time since DMEPOCH in years.
    Must match the forward model which uses dt_years = (MJD - DMEPOCH) / 365.25.

    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time difference from DMEPOCH in seconds, shape (n_toas,)
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz, shape (n_toas,)

    Returns
    -------
    derivative : jnp.ndarray
        d(delay)/d(DM1) in units of seconds/(pc cm^-^3 yr^-^1)
        Shape (n_toas,)
    """
    dt_years = dt_sec / SECS_PER_YEAR
    return K_DM_SEC * dt_years / (freq_mhz ** 2)


@jax.jit
def d_delay_d_DM2(dt_sec: jnp.ndarray, freq_mhz: jnp.ndarray) -> jnp.ndarray:
    """Compute derivative of dispersion delay with respect to DM2.

    DM2 represents quadratic DM evolution in pc cm^-^3 yr^-^2:
        DM(t) = DM + DM1*t_yr + 0.5*DM2*t_yr^2

    The delay contribution from DM2 is:
        tau_DM2 = K_DM * 0.5 * DM2 * t_yr^2 / freq^2

    Therefore:
        dtau/dDM2 = 0.5 * K_DM * t_yr^2 / freq^2

    where t_yr is time since DMEPOCH in years.
    Must match the forward model which uses dt_years = (MJD - DMEPOCH) / 365.25.

    Parameters
    ----------
    dt_sec : jnp.ndarray
        Time difference from DMEPOCH in seconds, shape (n_toas,)
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz, shape (n_toas,)

    Returns
    -------
    derivative : jnp.ndarray
        d(delay)/d(DM2) in units of seconds/(pc cm^-^3 yr^-^2)
        Shape (n_toas,)
    """
    dt_years = dt_sec / SECS_PER_YEAR
    return 0.5 * K_DM_SEC * (dt_years ** 2) / (freq_mhz ** 2)


def compute_dm_derivatives(
    params: Dict,
    toas_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    fit_params: list,
    **kwargs
) -> Dict[str, jnp.ndarray]:
    """Compute all DM parameter derivatives for design matrix.
    
    This is the main interface function, analogous to compute_spin_derivatives()
    in derivatives_spin.py.
    
    NOTE: Not JIT'd because of Python-level dict/string operations and variable
    loop over fit_params. The inner derivative functions ARE JIT'd.
    
    Parameters
    ----------
    params : dict
        Timing model parameters including DMEPOCH, DM, DM1, DM2, etc.
    toas_mjd : jnp.ndarray
        TOA times in MJD, shape (n_toas,)
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz, shape (n_toas,)
    fit_params : list
        List of DM parameters to fit (e.g., ['DM', 'DM1'])
    **kwargs
        Additional arguments (for API compatibility)
        
    Returns
    -------
    derivatives : dict
        Dictionary mapping parameter name to derivative column
        Each value is jnp.ndarray of shape (n_toas,) in seconds/param_unit
        
    Notes
    -----
    Sign Convention:
    - DM derivatives are POSITIVE (unlike spin which are negative)
    - Increasing DM increases delay -> arrival time increases
    - This matches PINT's convention for delay-based parameters
    
    Units:
    - Derivatives are in time units (seconds per parameter unit)
    - No F0 conversion needed (unlike phase-based spin parameters)
    - Already in correct units for WLS design matrix
    
    Examples
    --------
    >>> params = {'DMEPOCH': 58000.0, 'DM': 10.39}
    >>> toas = np.array([58000.0, 58001.0, 58002.0])
    >>> freq = np.array([1400.0, 1400.0, 1400.0])
    >>> derivs = compute_dm_derivatives(params, toas, freq, ['DM', 'DM1'])
    >>> derivs['DM'].shape
    (3,)
    >>> np.all(derivs['DM'] > 0)  # Positive derivatives
    True
    """
    # Get DMEPOCH (reference epoch for DM evolution)
    # If not specified, use first TOA as reference
    dmepoch_mjd = params.get('DMEPOCH', toas_mjd[0])
    
    # Compute time difference from DMEPOCH in seconds
    dt_sec = (toas_mjd - dmepoch_mjd) * SECS_PER_DAY
    
    # Compute derivatives for each requested DM parameter
    derivatives = {}
    
    for param in fit_params:
        if param == 'DM':
            # Base DM: dtau/dDM = K_DM / freq^2
            derivatives[param] = d_delay_d_DM(freq_mhz)
            
        elif param == 'DM1':
            # Linear DM evolution: dtau/dDM1 = K_DM * t / freq^2
            derivatives[param] = d_delay_d_DM1(dt_sec, freq_mhz)
            
        elif param == 'DM2':
            # Quadratic DM evolution: dtau/dDM2 = 0.5 * K_DM * t^2 / freq^2
            derivatives[param] = d_delay_d_DM2(dt_sec, freq_mhz)
            
        elif param.startswith('DM') and len(param) > 2:
            # Higher-order DM terms (DM3, DM4, ...)
            try:
                order = int(param[2:])  # 'DM3' -> 3, 'DM4' -> 4
                # General formula: dtau/dDM_n = (K_DM * t_yr^n / n!) / freq^2
                dt_years = dt_sec / SECS_PER_YEAR
                factorial = math.factorial(order)
                derivatives[param] = K_DM_SEC * (dt_years ** order) / factorial / (freq_mhz ** 2)
            except (ValueError, OverflowError):
                raise ValueError(f"Cannot parse DM parameter: {param}")
        else:
            # Not a DM parameter - skip
            continue
    
    return derivatives



