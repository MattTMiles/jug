"""Gauss-Newton least squares solver with Levenberg-Marquardt damping.

This implements the standard pulsar timing fitting approach:
linearize the model and solve analytically.
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Optional
import copy


def fit_gauss_newton(
    compute_residuals_func: Callable,
    compute_design_matrix_func: Callable,
    params_init: Dict[str, float],
    fit_params: List[str],
    data: Dict,
    max_iter: int = 20,
    lambda_init: float = 1e-3,
    lambda_min: float = 1e-10,
    lambda_max: float = 1e10,
    convergence_threshold: float = 1e-10,
    verbose: bool = True
) -> Dict:
    """Fit timing model parameters using Gauss-Newton with LM damping.
    
    This is the standard approach in pulsar timing: linearize around
    current parameters and solve analytically. Levenberg-Marquardt
    damping adds robustness for ill-conditioned problems.
    
    Parameters
    ----------
    compute_residuals_func : callable
        Function(params, data) -> residuals_us
    compute_design_matrix_func : callable
        Function(params, data, fit_params) -> design_matrix
    params_init : dict
        Initial parameter values (all parameters)
    fit_params : list of str
        Names of parameters to fit
    data : dict
        Fixed data (TOAs, errors, frequencies, etc.)
    max_iter : int
        Maximum number of iterations
    lambda_init : float
        Initial LM damping parameter
    lambda_min : float
        Minimum damping parameter
    lambda_max : float
        Maximum damping parameter
    convergence_threshold : float
        Stop when relative parameter changes < this
    verbose : bool
        Print progress
    
    Returns
    -------
    result : dict
        - 'params': Fitted parameters (dict)
        - 'uncertainties': Parameter uncertainties (dict)
        - 'covariance': Covariance matrix (n_params, n_params)
        - 'chi2': Final chi-squared
        - 'reduced_chi2': Chi2 / DOF
        - 'rms_us': RMS residual in microseconds
        - 'weighted_rms_us': Weighted RMS residual
        - 'residuals_us': Final residuals
        - 'iterations': Number of iterations
        - 'converged': Whether fit converged
        - 'backend': 'numpy'
    """
    if verbose:
        print("\n" + "="*70)
        print("Gauss-Newton Least Squares Fitting (NumPy)")
        print("="*70)
        print(f"Fitting {len(fit_params)} parameters:")
        for name in fit_params:
            print(f"  {name:10s} = {params_init[name]:.12e}")
        print()
    
    params = copy.deepcopy(params_init)
    errors_us = data['errors_us']
    n_toas = len(errors_us)
    n_params = len(fit_params)
    dof = n_toas - n_params
    
    lambda_param = lambda_init
    chi2_old = np.inf
    
    converged = False
    iteration = 0
    
    for iteration in range(max_iter):
        # Compute residuals and chi-squared
        residuals_us = compute_residuals_func(params, data)
        chi2 = np.sum((residuals_us / errors_us) ** 2)
        rms_us = np.sqrt(np.mean(residuals_us**2))
        wrms_us = np.sqrt(np.sum((residuals_us / errors_us)**2) / n_toas)
        
        if verbose:
            print(f"Iteration {iteration}:")
            print(f"  Chi2 = {chi2:.2f}, Reduced chi2 = {chi2/dof:.3f}")
            print(f"  RMS = {rms_us:.3f} μs, Weighted RMS = {wrms_us:.3f} μs")
        
        # Compute design matrix (analytical derivatives)
        M = compute_design_matrix_func(params, data, fit_params)
        
        # Normal equations: M^T M * delta = -M^T * r
        # (Note: M is already weighted by 1/errors)
        MTM = M.T @ M
        MTr = M.T @ (residuals_us / errors_us)
        
        # Add Levenberg-Marquardt damping
        # Helps with ill-conditioned problems and acts as trust region
        damped_MTM = MTM + lambda_param * np.diag(np.diag(MTM))
        
        # Solve for parameter updates
        try:
            delta = np.linalg.solve(damped_MTM, -MTr)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"  WARNING: Singular matrix at iteration {iteration}")
            # Increase damping and try again
            lambda_param = min(lambda_param * 10, lambda_max)
            continue
        
        # Try the step
        params_new = update_params(params, fit_params, delta)
        residuals_new = compute_residuals_func(params_new, data)
        chi2_new = np.sum((residuals_new / errors_us) ** 2)
        
        # Check if step improved fit
        if chi2_new < chi2:
            # Accept step
            params = params_new
            chi2_improvement = chi2 - chi2_new
            
            if verbose:
                print(f"  Step accepted, chi2 improved by {chi2_improvement:.2f}")
                print(f"  Lambda: {lambda_param:.2e}")
            
            # Decrease damping (trust region expansion)
            lambda_param = max(lambda_param * 0.1, lambda_min)
            
            # Check convergence
            max_rel_change = np.max(np.abs(delta / extract_param_values(params, fit_params)))
            
            if verbose:
                print(f"  Max relative change: {max_rel_change:.2e}")
            
            if max_rel_change < convergence_threshold:
                if verbose:
                    print(f"\n✅ Converged after {iteration + 1} iterations!")
                converged = True
                break
            
            chi2_old = chi2
        else:
            # Reject step, increase damping (trust region contraction)
            if verbose:
                print(f"  Step rejected, chi2 increased: {chi2:.2f} -> {chi2_new:.2f}")
                print(f"  Increasing damping: {lambda_param:.2e} -> {lambda_param*10:.2e}")
            
            lambda_param = min(lambda_param * 10, lambda_max)
            
            if lambda_param >= lambda_max:
                if verbose:
                    print("  WARNING: Damping reached maximum, stopping")
                break
    
    if not converged and verbose:
        print(f"\n⚠️  Reached maximum iterations ({max_iter}) without convergence")
    
    # Compute final residuals and statistics
    residuals_us = compute_residuals_func(params, data)
    chi2 = np.sum((residuals_us / errors_us) ** 2)
    rms_us = np.sqrt(np.mean(residuals_us**2))
    wrms_us = np.sqrt(np.sum((residuals_us / errors_us)**2) / n_toas)
    
    # Compute covariance matrix and uncertainties
    M = compute_design_matrix_func(params, data, fit_params)
    MTM = M.T @ M
    
    try:
        covariance = np.linalg.inv(MTM)
        uncertainties = {
            fit_params[i]: np.sqrt(covariance[i, i])
            for i in range(n_params)
        }
    except np.linalg.LinAlgError:
        covariance = np.full((n_params, n_params), np.nan)
        uncertainties = {name: np.nan for name in fit_params}
        if verbose:
            print("  WARNING: Could not compute covariance matrix")
    
    if verbose:
        print(f"\n{'='*70}")
        print("Final Results:")
        print(f"{'='*70}")
        print(f"Chi2: {chi2:.2f}")
        print(f"Reduced chi2: {chi2/dof:.3f}")
        print(f"RMS residual: {rms_us:.3f} μs")
        print(f"Weighted RMS residual: {wrms_us:.3f} μs")
        print(f"Iterations: {iteration + 1}")
        print(f"\nFitted Parameters:")
        for name in fit_params:
            print(f"  {name:10s} = {params[name]:20.12e} ± {uncertainties[name]:.2e}")
        print("="*70 + "\n")
    
    return {
        'params': params,
        'uncertainties': uncertainties,
        'covariance': covariance,
        'chi2': chi2,
        'reduced_chi2': chi2 / dof,
        'rms_us': rms_us,
        'weighted_rms_us': wrms_us,
        'residuals_us': residuals_us,
        'iterations': iteration + 1,
        'converged': converged,
        'backend': 'numpy'
    }


def update_params(
    params: Dict[str, float],
    fit_params: List[str],
    delta: np.ndarray
) -> Dict[str, float]:
    """Update parameters with computed deltas.
    
    Parameters
    ----------
    params : dict
        Current parameters
    fit_params : list of str
        Names of parameters being fitted
    delta : ndarray
        Parameter updates
    
    Returns
    -------
    params_new : dict
        Updated parameters
    """
    params_new = copy.deepcopy(params)
    for i, name in enumerate(fit_params):
        params_new[name] += delta[i]
    return params_new


def extract_param_values(
    params: Dict[str, float],
    param_names: List[str]
) -> np.ndarray:
    """Extract parameter values as array.
    
    Parameters
    ----------
    params : dict
        Parameters dictionary
    param_names : list of str
        Names of parameters to extract
    
    Returns
    -------
    values : ndarray
        Parameter values
    """
    return np.array([params[name] for name in param_names])
