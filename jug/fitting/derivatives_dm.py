"""Analytical derivatives for DM parameters (DM, DM1, DM2, ...).

This module implements PINT-compatible analytical derivatives for
dispersion measure parameters. The formulas follow PINT's dispersion.py.

DM affects timing through cold-plasma dispersion delay:
    τ_DM = K_DM × DM / freq²

where K_DM = 1/2.41e-4 ≈ 4148.808 MHz² pc⁻¹ cm³ s

DM can vary with time as a polynomial:
    DM(t) = DM + DM1×t + DM2×t²/2 + ...

Reference: PINT src/pint/models/dispersion_model.py
"""

import numpy as np
from typing import Dict

# Physical constant
K_DM_SEC = 1.0 / 2.41e-4  # ≈ 4148.808 MHz² pc⁻¹ cm³ s
SECS_PER_DAY = 86400.0


def d_delay_d_DM(freq_mhz: np.ndarray) -> np.ndarray:
    """Compute derivative of dispersion delay with respect to DM.
    
    The dispersion delay is:
        τ_DM = K_DM × DM / freq²
    
    Therefore:
        ∂τ/∂DM = K_DM / freq²
    
    Parameters
    ----------
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
        
    Returns
    -------
    derivative : np.ndarray
        ∂(delay)/∂(DM) in units of seconds/(pc cm⁻³)
        Shape (n_toas,)
        
    Notes
    -----
    - Derivative is POSITIVE: increasing DM increases delay
    - Already in time units (seconds), no F0 conversion needed
    - Frequency dependent: lower freq → larger derivative
    
    Example
    -------
    >>> freq = np.array([1400.0, 700.0])  # MHz
    >>> deriv = d_delay_d_DM(freq)
    >>> print(deriv)
    [2.12e-03  8.47e-03]  # Lower freq has 4× larger derivative
    """
    return K_DM_SEC / (freq_mhz ** 2)


def d_delay_d_DM1(dt_sec: np.ndarray, freq_mhz: np.ndarray) -> np.ndarray:
    """Compute derivative of dispersion delay with respect to DM1.
    
    DM1 represents linear DM evolution in pc cm⁻³ day⁻¹:
        DM(t) = DM + DM1 × t
    
    The delay contribution from DM1 is:
        τ_DM1 = K_DM × DM1 × t / freq²
    
    Therefore:
        ∂τ/∂DM1 = K_DM × t / freq²
    
    where t is in days (DM1 has units pc cm⁻³ day⁻¹).
    
    Parameters
    ----------
    dt_sec : np.ndarray
        Time difference from DMEPOCH in seconds, shape (n_toas,)
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
        
    Returns
    -------
    derivative : np.ndarray
        ∂(delay)/∂(DM1) in units of seconds/(pc cm⁻³ day⁻¹)
        Shape (n_toas,)
        
    Notes
    -----
    - Time must be converted from seconds to days
    - Linear in time: derivative grows with |t|
    - POSITIVE: increasing DM1 increases delay at later times
    """
    dt_days = dt_sec / SECS_PER_DAY
    return K_DM_SEC * dt_days / (freq_mhz ** 2)


def d_delay_d_DM2(dt_sec: np.ndarray, freq_mhz: np.ndarray) -> np.ndarray:
    """Compute derivative of dispersion delay with respect to DM2.
    
    DM2 represents quadratic DM evolution in pc cm⁻³ day⁻²:
        DM(t) = DM + DM1×t + 0.5×DM2×t²
    
    The delay contribution from DM2 is:
        τ_DM2 = K_DM × 0.5 × DM2 × t² / freq²
    
    Therefore:
        ∂τ/∂DM2 = 0.5 × K_DM × t² / freq²
    
    where t is in days (DM2 has units pc cm⁻³ day⁻²).
    
    Parameters
    ----------
    dt_sec : np.ndarray
        Time difference from DMEPOCH in seconds, shape (n_toas,)
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
        
    Returns
    -------
    derivative : np.ndarray
        ∂(delay)/∂(DM2) in units of seconds/(pc cm⁻³ day⁻²)
        Shape (n_toas,)
        
    Notes
    -----
    - Time must be converted from seconds to days
    - Quadratic in time: derivative grows as t²
    - Factor of 0.5 from polynomial definition
    - POSITIVE: increasing DM2 increases delay at later times
    """
    dt_days = dt_sec / SECS_PER_DAY
    return 0.5 * K_DM_SEC * (dt_days ** 2) / (freq_mhz ** 2)


def compute_dm_derivatives(
    params: Dict,
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    fit_params: list,
    **kwargs
) -> Dict[str, np.ndarray]:
    """Compute all DM parameter derivatives for design matrix.
    
    This is the main interface function, analogous to compute_spin_derivatives()
    in derivatives_spin.py.
    
    Parameters
    ----------
    params : dict
        Timing model parameters including DMEPOCH, DM, DM1, DM2, etc.
    toas_mjd : np.ndarray
        TOA times in MJD, shape (n_toas,)
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
    fit_params : list
        List of DM parameters to fit (e.g., ['DM', 'DM1'])
    **kwargs
        Additional arguments (for API compatibility)
        
    Returns
    -------
    derivatives : dict
        Dictionary mapping parameter name to derivative column
        Each value is np.ndarray of shape (n_toas,) in seconds/param_unit
        
    Notes
    -----
    Sign Convention:
    - DM derivatives are POSITIVE (unlike spin which are negative)
    - Increasing DM increases delay → arrival time increases
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
            # Base DM: ∂τ/∂DM = K_DM / freq²
            derivatives[param] = d_delay_d_DM(freq_mhz)
            
        elif param == 'DM1':
            # Linear DM evolution: ∂τ/∂DM1 = K_DM × t / freq²
            derivatives[param] = d_delay_d_DM1(dt_sec, freq_mhz)
            
        elif param == 'DM2':
            # Quadratic DM evolution: ∂τ/∂DM2 = 0.5 × K_DM × t² / freq²
            derivatives[param] = d_delay_d_DM2(dt_sec, freq_mhz)
            
        elif param.startswith('DM') and len(param) > 2:
            # Higher-order DM terms (DM3, DM4, ...)
            try:
                order = int(param[2:])  # 'DM3' -> 3, 'DM4' -> 4
                # General formula: ∂τ/∂DM_n = (K_DM × t^n / n!) / freq²
                dt_days = dt_sec / SECS_PER_DAY
                factorial = np.math.factorial(order)
                derivatives[param] = K_DM_SEC * (dt_days ** order) / factorial / (freq_mhz ** 2)
            except (ValueError, OverflowError):
                raise ValueError(f"Cannot parse DM parameter: {param}")
        else:
            # Not a DM parameter - skip
            continue
    
    return derivatives


if __name__ == '__main__':
    # Quick test
    print("Testing DM derivatives...")
    
    # Test case: single TOA at two frequencies
    freq = np.array([1400.0, 700.0])  # MHz
    
    # Test d/dDM (constant in time, varies with freq)
    deriv_dm = d_delay_d_DM(freq)
    print(f"\n∂τ/∂DM at freq=[1400, 700] MHz:")
    print(f"  {deriv_dm * 1e6} μs/(pc cm⁻³)")
    print(f"  Expected ratio: 4.0 (freq² scaling)")
    print(f"  Actual ratio: {deriv_dm[1] / deriv_dm[0]:.3f}")
    
    # Test d/dDM1 (linear in time)
    dt_days = np.array([0.0, 100.0, 200.0])
    dt_sec = dt_days * SECS_PER_DAY
    freq_const = np.array([1400.0, 1400.0, 1400.0])
    
    deriv_dm1 = d_delay_d_DM1(dt_sec, freq_const)
    print(f"\n∂τ/∂DM1 at t=[0, 100, 200] days:")
    print(f"  {deriv_dm1 * 1e6} μs/(pc cm⁻³ day⁻¹)")
    print(f"  Expected: linear growth with time")
    print(f"  Ratio [1]/[0]: {deriv_dm1[1] / deriv_dm1[0] if deriv_dm1[0] != 0 else 'inf'}")
    
    # Test d/dDM2 (quadratic in time)
    deriv_dm2 = d_delay_d_DM2(dt_sec, freq_const)
    print(f"\n∂τ/∂DM2 at t=[0, 100, 200] days:")
    print(f"  {deriv_dm2 * 1e6} μs/(pc cm⁻³ day⁻²)")
    print(f"  Expected: quadratic growth (t²)")
    
    # Test compute_dm_derivatives interface
    params = {
        'DMEPOCH': 58000.0,
        'DM': 10.39,
        'DM1': 0.001,
    }
    toas = np.array([58000.0, 58100.0, 58200.0])
    freq_test = np.array([1400.0, 1400.0, 1400.0])
    
    derivs = compute_dm_derivatives(params, toas, freq_test, ['DM', 'DM1'])
    print(f"\ncompute_dm_derivatives() test:")
    print(f"  DM derivatives shape: {derivs['DM'].shape}")
    print(f"  DM1 derivatives shape: {derivs['DM1'].shape}")
    print(f"  All DM derivatives positive: {np.all(derivs['DM'] > 0)}")
    print(f"  DM1 grows with time: {np.all(np.diff(derivs['DM1']) > 0)}")
    
    print("\n✓ DM derivatives module ready!")
