"""Analytical derivatives for FD (frequency-dependent) parameters.

FD parameters model profile evolution with observing frequency using
a polynomial in log(freq):

    delay = FD1 * log(f/1GHz) + FD2 * log(f/1GHz)^2 + FD3 * log(f/1GHz)^3 + ...

where f is the observing frequency in GHz.

The derivatives are trivial:
    d(delay)/d(FD_n) = log(f/1GHz)^n

This module follows the same conventions as PINT for FD delay computation.

Usage
-----
>>> from jug.fitting.derivatives_fd import compute_fd_derivatives
>>> derivs = compute_fd_derivatives(params, freq_mhz, fit_params=['FD1', 'FD2'])
>>> print(derivs['FD1'].shape)  # (n_toas,)

References
----------
- PINT: pint/models/frequency_dependent.py
- Arzoumanian et al. (2015), ApJ, 813, 65 (NANOGrav 9-yr)
"""

import numpy as np
from typing import Dict, List
import re


def compute_fd_derivatives(
    params: Dict,
    freq_mhz: np.ndarray,
    fit_params: List[str],
) -> Dict[str, np.ndarray]:
    """Compute FD parameter derivatives.

    The derivative d(delay)/d(FD_n) = log(freq/1GHz)^n

    Parameters
    ----------
    params : dict
        Timing model parameters (FD values not actually used for derivatives)
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
    fit_params : list of str
        Parameters to fit (e.g., ['FD1', 'FD2', 'FD3'])

    Returns
    -------
    dict
        Dictionary mapping FD parameter name to derivative column (seconds)
        Each value has shape (n_toas,)

    Notes
    -----
    FD parameters are in seconds in the par file and internally.
    The derivative is dimensionless (log ratio), so derivative * FD gives seconds.

    The formula matches PINT and Tempo2:
        delay = sum_n FD_n * log(freq/1GHz)^n

    Examples
    --------
    >>> import numpy as np
    >>> freq_mhz = np.array([1000., 1500., 2000.])  # MHz
    >>> params = {'FD1': 1e-5, 'FD2': -2e-6}
    >>> derivs = compute_fd_derivatives(params, freq_mhz, ['FD1', 'FD2'])
    >>> derivs['FD1']  # log(1), log(1.5), log(2)
    array([0.        , 0.40546511, 0.69314718])
    """
    # Filter to only FD parameters
    fd_params = [p for p in fit_params if p.startswith('FD')]
    
    if not fd_params:
        return {}
    
    # Convert frequency to GHz and compute log
    freq_ghz = np.asarray(freq_mhz) / 1000.0
    log_freq = np.log(freq_ghz)
    
    derivatives = {}
    
    for param_name in fd_params:
        # Extract the order (FD1 -> 1, FD2 -> 2, etc.)
        match = re.match(r'FD(\d+)', param_name)
        if match:
            order = int(match.group(1))
            # d(delay)/d(FD_n) = log(freq/1GHz)^n
            derivatives[param_name] = log_freq ** order
        else:
            raise ValueError(f"Invalid FD parameter name: {param_name}")
    
    return derivatives


def compute_fd_delay(
    freq_mhz: np.ndarray,
    fd_params: Dict[str, float],
) -> np.ndarray:
    """Compute total FD (frequency-dependent) delay.

    The FD delay is:
        delay = FD1 * log(f/1GHz) + FD2 * log(f/1GHz)^2 + ...

    Parameters
    ----------
    freq_mhz : np.ndarray
        Observing frequencies in MHz, shape (n_toas,)
    fd_params : dict
        Dictionary of FD parameter values, e.g., {'FD1': 1e-5, 'FD2': -2e-6}

    Returns
    -------
    np.ndarray
        FD delay in seconds, shape (n_toas,)
    """
    freq_ghz = np.asarray(freq_mhz) / 1000.0
    log_freq = np.log(freq_ghz)
    
    delay = np.zeros_like(log_freq)
    
    for param_name, value in fd_params.items():
        if param_name.startswith('FD'):
            match = re.match(r'FD(\d+)', param_name)
            if match:
                order = int(match.group(1))
                delay += value * (log_freq ** order)
    
    return delay


def get_fd_derivative_column(
    freq_mhz: np.ndarray,
    order: int,
) -> np.ndarray:
    """Get derivative column for a specific FD parameter.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Observing frequencies in MHz
    order : int
        FD order (1 for FD1, 2 for FD2, etc.)

    Returns
    -------
    np.ndarray
        Derivative column, shape (n_toas,)
    """
    freq_ghz = np.asarray(freq_mhz) / 1000.0
    log_freq = np.log(freq_ghz)
    return log_freq ** order


if __name__ == '__main__':
    # Quick test
    print("Testing FD derivatives...")
    
    freq_mhz = np.array([800., 1000., 1200., 1500., 2000.])
    params = {'FD1': 1e-5, 'FD2': -2e-6, 'FD3': 5e-7}
    
    derivs = compute_fd_derivatives(params, freq_mhz, ['FD1', 'FD2', 'FD3'])
    
    print(f"Frequencies (MHz): {freq_mhz}")
    print(f"log(freq/1GHz): {np.log(freq_mhz/1000)}")
    print()
    
    for name, deriv in derivs.items():
        print(f"{name} derivatives: {deriv}")
    
    # Verify formula: FD delay = sum(FD_n * log(f)^n)
    fd_delay = (params['FD1'] * derivs['FD1'] + 
                params['FD2'] * derivs['FD2'] + 
                params['FD3'] * derivs['FD3'])
    print(f"\nFD delay (s): {fd_delay}")
    print(f"FD delay (μs): {fd_delay * 1e6}")
    
    print("\n✓ FD derivatives module ready!")

