"""JAX-accelerated Gauss-Newton least squares solver.

This module provides JIT-compiled matrix operations for fast fitting.
Expected speedup: 10-60x for datasets > 500 TOAs.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Callable, Tuple, List
from functools import partial


@jax.jit
def compute_weighted_chi2_jax(residuals: jnp.ndarray, errors: jnp.ndarray) -> float:
    """Compute weighted chi-squared.
    
    Parameters
    ----------
    residuals : jnp.ndarray
        Residuals in seconds
    errors : jnp.ndarray
        Uncertainties in seconds
        
    Returns
    -------
    chi2 : float
        Weighted chi-squared statistic
    """
    weights = 1.0 / errors**2
    return jnp.sum(weights * residuals**2)


@jax.jit
def scale_design_matrix(M: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scale design matrix columns by their RMS for numerical stability.
    
    This prevents numerical overflow when forming M^T W M for parameters
    with very different scales (e.g., F0 ~ 10^5, F1 ~ 10^12).
    
    Parameters
    ----------
    M : jnp.ndarray
        Design matrix, shape (n_toas, n_params)
        
    Returns
    -------
    M_scaled : jnp.ndarray
        Scaled design matrix, each column has RMS = 1
    scales : jnp.ndarray
        Scaling factors (RMS of each column), shape (n_params,)
    """
    # Compute RMS of each column
    scales = jnp.sqrt(jnp.mean(M**2, axis=0))
    
    # Avoid division by zero for constant columns
    scales = jnp.where(scales > 0, scales, 1.0)
    
    # Scale each column
    M_scaled = M / scales[jnp.newaxis, :]
    
    return M_scaled, scales


@jax.jit
def gauss_newton_step_jax(
    residuals: jnp.ndarray,
    design_matrix: jnp.ndarray,
    errors: jnp.ndarray,
    lambda_lm: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform one Gauss-Newton step with Levenberg-Marquardt damping.
    
    Solves: (M^T W M + λI) Δp = M^T W r
    
    Uses column scaling for numerical stability.
    
    Parameters
    ----------
    residuals : jnp.ndarray
        Current residuals (seconds)
    design_matrix : jnp.ndarray
        Design matrix M, shape (n_toas, n_params)
    errors : jnp.ndarray
        TOA uncertainties (seconds)
    lambda_lm : float
        Levenberg-Marquardt damping parameter
        
    Returns
    -------
    delta_params : jnp.ndarray
        Parameter updates (in original scale)
    covariance : jnp.ndarray
        Covariance matrix (M^T W M)^-1 (in original scale)
    scales : jnp.ndarray
        Column scaling factors used
    """
    n_toas, n_params = design_matrix.shape
    
    # Scale design matrix for numerical stability
    M_scaled, scales = scale_design_matrix(design_matrix)
    
    # Weight matrix
    weights = 1.0 / errors**2
    W = jnp.diag(weights)
    
    # Normal equations with scaled M: A = M_scaled^T W M_scaled
    A = M_scaled.T @ W @ M_scaled
    
    # Add Levenberg-Marquardt damping: A + λI
    A_damped = A + lambda_lm * jnp.eye(n_params)
    
    # Right hand side: b = M_scaled^T W r
    b = M_scaled.T @ W @ residuals
    
    # Solve for scaled parameter updates: (A + λI) Δp_scaled = b
    delta_params_scaled = jax.scipy.linalg.solve(A_damped, b, assume_a='pos')
    
    # Unscale parameter updates
    delta_params = delta_params_scaled / scales
    
    # Covariance matrix (without damping, in original scale)
    # Cov = (M^T W M)^-1 = S * (M_scaled^T W M_scaled)^-1 * S
    # where S = diag(scales)
    cov_scaled = jnp.linalg.inv(A)
    covariance = cov_scaled / (scales[:, jnp.newaxis] * scales[jnp.newaxis, :])
    
    return delta_params, covariance, scales


def gauss_newton_fit_jax(
    residuals_fn: Callable,
    params: Dict[str, float],
    fit_params: List[str],
    design_matrix_fn: Callable,
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    errors_us: np.ndarray,
    max_iter: int = 20,
    lambda_init: float = 1e-3,
    lambda_factor: float = 10.0,
    convergence_threshold: float = 1e-6,
    verbose: bool = True
) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """Fit timing model using JAX-accelerated Gauss-Newton.
    
    This is a hybrid implementation:
    - Parameter updates: Pure NumPy (small arrays)
    - Matrix operations: JAX JIT-compiled (fast)
    - Residual computation: User-provided function
    
    Parameters
    ----------
    residuals_fn : callable
        Function: residuals_fn(params) -> residuals (μs)
    params : dict
        Initial parameter values
    fit_params : list of str
        Parameters to fit
    design_matrix_fn : callable
        Function to compute design matrix
    toas_mjd : ndarray
        TOA times (MJD)
    freq_mhz : ndarray
        Observing frequencies (MHz)
    errors_us : ndarray
        TOA uncertainties (μs)
    max_iter : int
        Maximum iterations
    lambda_init : float
        Initial Levenberg-Marquardt damping
    lambda_factor : float
        Factor to increase/decrease lambda
    convergence_threshold : float
        Convergence threshold for chi2 and parameter changes
    verbose : bool
        Print progress
        
    Returns
    -------
    fitted_params : dict
        Best-fit parameter values
    uncertainties : dict
        1-sigma parameter uncertainties
    info : dict
        Fitting statistics
    """
    # Convert to JAX arrays
    toas_jax = jnp.array(toas_mjd)
    freq_jax = jnp.array(freq_mhz)
    errors_jax = jnp.array(errors_us * 1e-6)  # Convert to seconds
    
    # Initialize
    current_params = params.copy()
    lambda_lm = lambda_init
    
    # Initial residuals and chi2
    residuals_us = residuals_fn(current_params)
    residuals_sec = jnp.array(residuals_us * 1e-6)
    chi2 = float(compute_weighted_chi2_jax(residuals_sec, errors_jax))
    dof = len(toas_mjd) - len(fit_params)
    reduced_chi2 = chi2 / dof if dof > 0 else np.nan
    
    if verbose:
        print(f"{'='*80}")
        print(f"JAX-Accelerated Gauss-Newton Fitting")
        print(f"{'='*80}")
        print(f"Initial chi2: {chi2:.2f}, reduced chi2: {reduced_chi2:.4f}")
        print(f"DOF: {dof}, N_TOAs: {len(toas_mjd)}, N_params: {len(fit_params)}")
        print(f"Fitting: {', '.join(fit_params)}")
        print(f"\n{'Iter':<6} {'Chi2':<12} {'Reduced χ²':<12} {'RMS (μs)':<12} {'Lambda':<12} {'Status'}")
        print(f"{'-'*80}")
    
    # Fitting loop
    iteration_info = []
    
    for iteration in range(max_iter):
        # Compute design matrix (returns weighted matrix)
        M_weighted = design_matrix_fn(
            current_params, toas_mjd, freq_mhz, errors_us, fit_params
        )
        M_weighted_jax = jnp.array(M_weighted)
        
        # The design matrix function returns M * diag(1/errors)
        # We need unweighted M for the step computation
        weights_inv = errors_jax
        M_unweighted = M_weighted_jax * weights_inv[:, jnp.newaxis]
        
        # Compute parameter update (this applies weighting internally)
        delta_params_jax, covariance_jax, scales_jax = gauss_newton_step_jax(
            residuals_sec, M_unweighted, errors_jax, lambda_lm
        )
        delta_params = np.array(delta_params_jax)
        
        # Apply update
        trial_params = current_params.copy()
        for i, param_name in enumerate(fit_params):
            trial_params[param_name] = current_params[param_name] - delta_params[i]
        
        # Evaluate trial parameters
        trial_residuals_us = residuals_fn(trial_params)
        trial_residuals_sec = jnp.array(trial_residuals_us * 1e-6)
        trial_chi2 = float(compute_weighted_chi2_jax(trial_residuals_sec, errors_jax))
        trial_reduced_chi2 = trial_chi2 / dof if dof > 0 else np.nan
        
        # Check if step improved fit
        delta_chi2 = chi2 - trial_chi2
        rms_us = float(jnp.std(trial_residuals_sec)) * 1e6
        
        if trial_chi2 < chi2:
            # Accept step
            current_params = trial_params
            residuals_sec = trial_residuals_sec
            chi2 = trial_chi2
            reduced_chi2 = trial_reduced_chi2
            
            # Reduce damping
            lambda_lm /= lambda_factor
            status = "✓ Accept"
            
            # Check convergence
            max_delta_param = np.max(np.abs(delta_params))
            converged = (abs(delta_chi2) < convergence_threshold and 
                        max_delta_param < convergence_threshold)
            
        else:
            # Reject step
            lambda_lm *= lambda_factor
            status = "✗ Reject"
            converged = False
        
        if verbose:
            print(f"{iteration+1:<6} {chi2:<12.2f} {reduced_chi2:<12.6f} {rms_us:<12.3f} "
                  f"{lambda_lm:<12.3e} {status}")
        
        iteration_info.append({
            'iteration': iteration + 1,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'rms_us': rms_us,
            'lambda': lambda_lm,
            'accepted': trial_chi2 < chi2
        })
        
        if converged:
            if verbose:
                print(f"\n✓ Converged after {iteration+1} iterations!")
            break
    
    # Compute final uncertainties from covariance matrix
    # Need to recompute with final parameters
    M_weighted_final = design_matrix_fn(
        current_params, toas_mjd, freq_mhz, errors_us, fit_params
    )
    M_weighted_final_jax = jnp.array(M_weighted_final)
    
    # Unweight to get M
    weights_inv = errors_jax
    M_unweighted_final = M_weighted_final_jax * weights_inv[:, jnp.newaxis]
    
    # Recompute covariance without damping
    _, covariance_final, _ = gauss_newton_step_jax(
        residuals_sec, M_unweighted_final, errors_jax, 0.0
    )
    
    # Extract uncertainties (diagonal of covariance matrix)
    uncertainties = {}
    covariance_np = np.array(covariance_final)
    for i, param_name in enumerate(fit_params):
        # Uncertainty is sqrt of diagonal element
        # Multiply by sqrt(reduced_chi2) to account for underestimated errors
        sigma = np.sqrt(covariance_np[i, i]) * np.sqrt(max(1.0, reduced_chi2))
        uncertainties[param_name] = sigma
    
    # Compile info
    info = {
        'iterations': len(iteration_info),
        'converged': converged,
        'final_chi2': chi2,
        'final_reduced_chi2': reduced_chi2,
        'dof': dof,
        'final_rms_us': float(jnp.std(residuals_sec)) * 1e6,
        'covariance_matrix': covariance_np,
        'iteration_history': iteration_info,
        'backend': 'jax'
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Final Results:")
        print(f"  Chi2: {chi2:.2f}")
        print(f"  Reduced Chi2: {reduced_chi2:.6f}")
        print(f"  RMS: {info['final_rms_us']:.3f} μs")
        print(f"\nFitted Parameters:")
        for param_name in fit_params:
            val = current_params[param_name]
            unc = uncertainties[param_name]
            print(f"  {param_name:10s} = {val:20.12e} ± {unc:12.6e}")
        print(f"{'='*80}")
    
    return current_params, uncertainties, info


# Convenience function for hybrid backend selection
def gauss_newton_fit_auto(
    residuals_fn: Callable,
    params: Dict[str, float],
    fit_params: List[str],
    toas_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    errors_us: np.ndarray,
    max_iter: int = 20,
    force_backend: str = None,
    **kwargs
) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    """Automatically select NumPy or JAX backend for fitting.
    
    Parameters
    ----------
    residuals_fn : callable
        Residual computation function
    params : dict
        Initial parameters
    fit_params : list of str
        Parameters to fit
    toas_mjd, freq_mhz, errors_us : ndarray
        TOA data
    max_iter : int
        Maximum iterations
    force_backend : str, optional
        Force 'numpy' or 'jax' backend
    **kwargs
        Additional arguments for fitting
        
    Returns
    -------
    fitted_params : dict
        Best-fit parameters
    uncertainties : dict
        Parameter uncertainties
    info : dict
        Fitting information
    """
    n_toas = len(toas_mjd)
    
    # Decide backend
    if force_backend:
        use_jax = (force_backend.lower() == 'jax')
    else:
        # Use JAX for datasets > 500 TOAs
        use_jax = (n_toas >= 500)
    
    if use_jax:
        # Use JAX design matrix
        from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper
        
        return gauss_newton_fit_jax(
            residuals_fn, params, fit_params,
            compute_design_matrix_jax_wrapper,
            toas_mjd, freq_mhz, errors_us,
            max_iter=max_iter, **kwargs
        )
    else:
        # Use NumPy version
        from jug.fitting.gauss_newton import fit_gauss_newton
        from jug.fitting.design_matrix import compute_design_matrix
        
        return fit_gauss_newton(
            residuals_fn, params, fit_params,
            compute_design_matrix,
            toas_mjd, freq_mhz, errors_us,
            max_iter=max_iter, **kwargs
        )
