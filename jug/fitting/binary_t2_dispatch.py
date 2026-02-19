"""T2 (Tempo2 universal) binary model dispatch.

T2 is Tempo2's general-purpose binary model that automatically selects the
appropriate computation based on which parameters are present in the par file:

- If TASC is present → ELL1-style (Laplace-Lagrange) computation
- If T0 is present   → DD-style (Keplerian) computation

This allows users with ``BINARY T2`` in their par files to use JUG without
changing the model name. The T2 model is mathematically equivalent to DD or
ELL1 depending on parameterization; it just auto-detects which one to use.

FB (orbital frequency) parameters are supported in both modes—ELL1 handles
them natively, and DD converts FB0 → PB when needed.

References
----------
- Edwards et al. (2006), MNRAS 372, 1549 (Tempo2 paper)
- Tempo2 source: T2model.C
"""

import warnings
from typing import Dict, List

import numpy as np
import jax.numpy as jnp


def _is_ell1_parameterization(params: Dict) -> bool:
    """Determine whether T2 params use ELL1 (TASC) or DD (T0) parameterization.
    
    Tempo2's T2 model checks for TASC: if present, it uses ELL1-style
    computation with Laplace-Lagrange parameters (EPS1, EPS2). Otherwise
    it falls back to the standard Keplerian (DD) parameterization.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary from par file.
    
    Returns
    -------
    bool
        True if ELL1-style, False if DD-style.
    """
    has_tasc = 'TASC' in params and float(params.get('TASC', 0.0)) != 0.0
    has_t0 = 'T0' in params and float(params.get('T0', 0.0)) != 0.0
    
    if has_tasc and has_t0:
        warnings.warn(
            "T2 model has both TASC and T0; using ELL1 (TASC) parameterization",
            UserWarning, stacklevel=3
        )
        return True
    
    if has_tasc:
        return True
    
    if has_t0:
        return False
    
    # Neither TASC nor T0 — check for EPS1/EPS2 (ELL1 indicators)
    if 'EPS1' in params or 'EPS2' in params:
        return True
    
    # Default to DD
    return False


def compute_t2_binary_delay(
    toas_bary_mjd: jnp.ndarray,
    params: Dict,
) -> jnp.ndarray:
    """Compute T2 binary delay by dispatching to ELL1 or DD.
    
    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD.
    params : dict
        Parameter dictionary (must contain BINARY=T2 and either
        TASC/EPS1/EPS2 for ELL1 mode or T0/ECC/OM for DD mode).
    
    Returns
    -------
    delay : jnp.ndarray
        Binary delay in seconds.
    """
    if _is_ell1_parameterization(params):
        from jug.fitting.derivatives_binary import compute_ell1_binary_delay
        return compute_ell1_binary_delay(toas_bary_mjd, params)
    else:
        from jug.fitting.derivatives_dd import compute_dd_binary_delay
        return compute_dd_binary_delay(toas_bary_mjd, params)


def compute_binary_derivatives_t2(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str],
) -> Dict[str, jnp.ndarray]:
    """Compute T2 binary derivatives by dispatching to ELL1 or DD.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary.
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD.
    fit_params : list of str
        Parameters to compute derivatives for.
    
    Returns
    -------
    derivatives : dict
        Maps parameter names to derivative arrays.
    """
    if _is_ell1_parameterization(params):
        from jug.fitting.derivatives_binary import compute_binary_derivatives_ell1
        return compute_binary_derivatives_ell1(params, toas_bary_mjd, fit_params)
    else:
        # If KOM or KIN present, use DDK derivatives (supports Kopeikin corrections)
        has_kin_kom = 'KIN' in params or 'KOM' in params
        if has_kin_kom:
            from jug.fitting.derivatives_dd import compute_binary_derivatives_ddk
            return compute_binary_derivatives_ddk(params, toas_bary_mjd, fit_params)
        from jug.fitting.derivatives_dd import compute_binary_derivatives_dd
        return compute_binary_derivatives_dd(params, toas_bary_mjd, fit_params)
