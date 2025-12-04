"""Linear least squares optimizer for pulsar timing.

This implements the standard approach used in TEMPO/PINT:
linearize the timing model and solve analytically.
"""

import numpy as np
from typing import Dict, List, Tuple, Callable


def fit_linearized(
    compute_residuals_func: Callable,
    param_names: List[str],
    initial_params: Dict[str, float],
    fixed_data: Dict,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-15,
    verbose: bool = True
) -> Dict:
    """Fit timing model parameters using linearized least squares.
    
    This is the standard approach in pulsar timing: linearize the model
    around current parameters and solve analytically for parameter updates.
    
    Parameters
    ----------
    compute_residuals_func : callable
        Function that takes (params_dict, fixed_data) and returns residuals_us
    param_names : list of str
        Names of parameters to fit
    initial_params : dict
        Initial parameter values (all parameters)
    fixed_data : dict
        Data that doesn't change (TOAs, errors, etc.)
    max_iterations : int
        Maximum number of iterations
    convergence_threshold : float
        Stop when parameter changes < this threshold
    verbose : bool
        Print progress
        
    Returns
    -------
    result : dict
        - 'params': Fitted parameter dict
        - 'uncertainties': Parameter uncertainties
        - 'chi2': Final chi-squared
        - 'residuals_us': Final residuals
        - 'iterations': Number of iterations
        - 'converged': Whether fit converged
    """
    if verbose:
        print(f"\nLinearized Least Squares Fitting")
        print(f"  Fitting {len(param_names)} parameters:")
        for name in param_names:
            print(f"    {name} = {initial_params[name]:.6e}")
        print()
    
    params = initial_params.copy()
    errors_us = fixed_data['errors_us']
    n_toas = len(errors_us)
    converged = False  # Initialize
    
    for iteration in range(max_iterations):
        # Compute residuals with current parameters
        residuals_us = compute_residuals_func(params, fixed_data)
        
        # Compute chi-squared
        chi2 = np.sum((residuals_us / errors_us) ** 2)
        
        if verbose and (iteration == 0 or iteration % 1 == 0):
            rms = np.sqrt(np.mean(residuals_us**2))
            print(f"  Iteration {iteration}: chi2 = {chi2:.2f}, RMS = {rms:.3f} μs")
        
        # Compute design matrix (numerical derivatives)
        design_matrix = np.zeros((n_toas, len(param_names)))
        
        for i, param_name in enumerate(param_names):
            # Compute derivative numerically
            delta = abs(params[param_name]) * 1e-8 or 1e-15
            
            params_plus = params.copy()
            params_plus[param_name] += delta
            residuals_plus = compute_residuals_func(params_plus, fixed_data)
            
            params_minus = params.copy()
            params_minus[param_name] -= delta
            residuals_minus = compute_residuals_func(params_minus, fixed_data)
            
            # Central difference
            derivative = (residuals_plus - residuals_minus) / (2 * delta)
            design_matrix[:, i] = derivative
        
        # Weighted least squares
        weights = 1.0 / errors_us
        W = np.diag(weights)
        
        # Normal equations: (A^T W^2 A) * Δparams = -A^T W^2 * residuals
        # (Negative sign because we want to minimize residuals)
        ATA = design_matrix.T @ (W @ W) @ design_matrix
        ATb = -design_matrix.T @ (W @ W) @ residuals_us
        
        # Solve for parameter updates
        try:
            delta_params = np.linalg.solve(ATA, ATb)
        except np.linalg.LinAlgError:
            print("  WARNING: Singular matrix, cannot invert!")
            break
        
        # Update parameters
        param_updates = {}
        max_change = 0.0
        for i, param_name in enumerate(param_names):
            params[param_name] += delta_params[i]
            param_updates[param_name] = delta_params[i]
            max_change = max(max_change, abs(delta_params[i] / params[param_name]))
        
        # Check convergence
        if max_change < convergence_threshold:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations!")
            converged = True
            break
    else:
        if verbose:
            print(f"  Reached maximum iterations ({max_iterations})")
        converged = False
    
    # Compute final residuals and uncertainties
    residuals_us = compute_residuals_func(params, fixed_data)
    chi2 = np.sum((residuals_us / errors_us) ** 2)
    
    # Covariance matrix from final design matrix
    try:
        covariance = np.linalg.inv(ATA)
        uncertainties = {
            param_names[i]: np.sqrt(covariance[i, i])
            for i in range(len(param_names))
        }
    except np.linalg.LinAlgError:
        uncertainties = {name: np.nan for name in param_names}
    
    if verbose:
        print(f"\n  Final Results:")
        print(f"    Chi2: {chi2:.2f}")
        print(f"    Reduced chi2: {chi2 / n_toas:.3f}")
        print(f"    RMS residual: {np.sqrt(np.mean(residuals_us**2)):.3f} μs")
        print(f"\n  Fitted Parameters:")
        for name in param_names:
            print(f"    {name} = {params[name]:.12e} ± {uncertainties[name]:.2e}")
    
    return {
        'params': params,
        'uncertainties': uncertainties,
        'chi2': chi2,
        'reduced_chi2': chi2 / n_toas,
        'residuals_us': residuals_us,
        'iterations': iteration + 1 if converged else max_iterations,
        'converged': converged
    }
