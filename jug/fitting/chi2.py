"""Chi-squared computation for fitting.

Computes weighted chi-squared for residuals using JAX.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List
import numpy as np


def compute_chi2_jax(
    param_array: jnp.ndarray,
    param_names: List[str],
    fixed_params: Dict[str, float],
    toa_data: Dict
) -> float:
    """Compute weighted chi-squared (JAX-compiled).
    
    This is the core objective function for fitting.
    
    Parameters
    ----------
    param_array : jax array
        Scaled parameter values being optimized
    param_names : list of str
        Names of parameters in param_array
    fixed_params : dict
        Parameters not being fitted
    toa_data : dict
        TOA data including residuals_us and errors_us
        
    Returns
    -------
    chi2 : float
        Weighted chi-squared value
    """
    # This is a placeholder - will be replaced with actual residual computation
    # For now, just return a dummy value to test the infrastructure
    
    # In the real implementation, this will:
    # 1. Unpack param_array into parameter dict
    # 2. Compute residuals with updated parameters
    # 3. Calculate weighted chi-squared
    
    # Placeholder: simple quadratic to test optimizer
    chi2 = jnp.sum(param_array ** 2)
    return chi2


# TODO: Implement full chi2 computation that calls residual calculator
# This requires refactoring compute_residuals_simple() to be JAX-compatible
