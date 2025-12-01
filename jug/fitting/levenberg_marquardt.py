"""
Levenberg-Marquardt fitter for pulsar timing parameters.

Uses damped least squares with adaptive damping parameter.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Dict
import numpy as np


def levenberg_marquardt_step(
    residual_func: Callable,
    params: jnp.ndarray,
    param_names: list,
    toas_mjd: jnp.ndarray,
    freqs_mhz: jnp.ndarray,
    errors_us: jnp.ndarray,
    damping: float = 1e-3,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """
    Single Levenberg-Marquardt step.
    
    Parameters
    ----------
    residual_func : callable
        Function that takes (toas, freqs, params_dict) and returns residuals in seconds
    params : array
        Current parameter values [n_params]
    param_names : list
        Names of parameters
    toas_mjd : array
        TOAs in MJD [n_toas]
    freqs_mhz : array
        Frequencies in MHz [n_toas]
    errors_us : array
        TOA errors in microseconds [n_toas]
    damping : float
        Damping parameter (lambda). Larger = more gradient descent, smaller = more Gauss-Newton
    
    Returns
    -------
    new_params : array
        Updated parameters
    new_chi2 : float
        Chi-squared after update
    delta : array
        Parameter update step
    """
    n_params = len(params)
    n_toas = len(toas_mjd)
    
    # Convert errors to seconds for weighting
    errors_sec = errors_us * 1e-6
    weights = 1.0 / (errors_sec ** 2)
    
    # Compute current residuals and chi2
    params_dict = {name: params[i] for i, name in enumerate(param_names)}
    residuals = residual_func(toas_mjd, freqs_mhz, params_dict)
    chi2 = jnp.sum(weights * residuals**2)
    
    # Compute Jacobian using JAX automatic differentiation
    # This is MUCH more accurate than finite differences for pulsar timing!
    def residuals_for_grad(param_array):
        """Wrapper for residual function that takes array input."""
        params_dict_temp = {name: param_array[j] for j, name in enumerate(param_names)}
        return residual_func(toas_mjd, freqs_mhz, params_dict_temp)

    # Compute Jacobian using forward-mode autodiff (more efficient for many TOAs, few params)
    jacobian_func = jax.jacfwd(residuals_for_grad)
    jacobian = jacobian_func(params)  # Shape: (n_toas, n_params)
    
    # Build weighted normal equations: (J^T W J + λ diag(J^T W J)) δ = J^T W r
    JtW = jacobian.T * weights
    JtWJ = JtW @ jacobian
    
    # Add damping to diagonal (Levenberg-Marquardt)
    damped_matrix = JtWJ + damping * jnp.diag(jnp.diag(JtWJ))
    
    # Solve for parameter update
    rhs = -JtW @ residuals
    delta = jnp.linalg.solve(damped_matrix, rhs)
    
    # Update parameters
    new_params = params + delta
    
    # Compute new chi2
    new_params_dict = {name: new_params[i] for i, name in enumerate(param_names)}
    new_residuals = residual_func(toas_mjd, freqs_mhz, new_params_dict)
    new_chi2 = jnp.sum(weights * new_residuals**2)
    
    return new_params, new_chi2, delta


def fit_levenberg_marquardt(
    residual_func: Callable,
    initial_params: Dict[str, float],
    toas_mjd: np.ndarray,
    freqs_mhz: np.ndarray,
    errors_us: np.ndarray,
    max_iterations: int = 20,
    tolerance: float = 1e-9,
    initial_damping: float = 1e-3,
    damping_factor: float = 10.0,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Fit pulsar timing parameters using Levenberg-Marquardt algorithm.
    
    Parameters
    ----------
    residual_func : callable
        Function that computes residuals
    initial_params : dict
        Initial parameter values
    toas_mjd, freqs_mhz, errors_us : arrays
        TOA data
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence tolerance on chi2 change
    initial_damping : float
        Initial damping parameter
    damping_factor : float
        Factor to increase/decrease damping
    verbose : bool
        Print iteration info
    
    Returns
    -------
    fitted_params : dict
        Fitted parameter values
    """
    # Convert to JAX arrays
    toas_jax = jnp.array(toas_mjd, dtype=jnp.float64)
    freqs_jax = jnp.array(freqs_mhz, dtype=jnp.float64)
    errors_jax = jnp.array(errors_us, dtype=jnp.float64)
    
    # Setup parameters
    param_names = list(initial_params.keys())
    params = jnp.array([initial_params[name] for name in param_names], dtype=jnp.float64)
    
    # Compute initial chi2
    params_dict = {name: params[i] for i, name in enumerate(param_names)}
    residuals = residual_func(toas_jax, freqs_jax, params_dict)
    errors_sec = errors_jax * 1e-6
    weights = 1.0 / (errors_sec ** 2)
    chi2 = float(jnp.sum(weights * residuals**2))
    
    if verbose:
        print(f"Initial chi2: {chi2:.3f}")
    
    damping = initial_damping
    prev_chi2 = chi2
    
    for iteration in range(max_iterations):
        # Try a step with current damping
        new_params, new_chi2, delta = levenberg_marquardt_step(
            residual_func, params, param_names, 
            toas_jax, freqs_jax, errors_jax, damping
        )
        
        new_chi2 = float(new_chi2)
        
        # Check if step improved chi2
        if new_chi2 < chi2:
            # Accept step and decrease damping (move towards Gauss-Newton)
            params = new_params
            chi2_improvement = chi2 - new_chi2
            chi2 = new_chi2
            damping = max(damping / damping_factor, 1e-10)
            
            if verbose:
                print(f"Iter {iteration+1}: chi2={chi2:.3f}, Δchi2={chi2_improvement:.6f}, λ={damping:.2e}")
                for i, name in enumerate(param_names):
                    print(f"  {name}: {float(params[i]):.12e} (Δ={float(delta[i]):.2e})")
            
            # Check convergence
            if chi2_improvement < tolerance:
                if verbose:
                    print(f"Converged! (Δchi2 < {tolerance})")
                break
        else:
            # Reject step and increase damping (move towards gradient descent)
            damping = min(damping * damping_factor, 1e10)
            if verbose:
                print(f"Iter {iteration+1}: Rejected step (chi2 increased to {new_chi2:.3f}), increasing λ to {damping:.2e}")
    
    # Convert back to dict
    fitted_params = {name: float(params[i]) for i, name in enumerate(param_names)}
    
    return fitted_params
