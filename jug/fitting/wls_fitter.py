"""
WLS (Weighted Least Squares) Fitter - PINT-compatible implementation.

This implements PINT's fit_wls_svd algorithm using JAX for autodiff while
maintaining full numerical compatibility with PINT's fitter.
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Callable


@jax.jit
def normalize_designmatrix(M: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Normalize each column of the design matrix.
    
    Returns:
        M_normalized: Design matrix with normalized columns
        norm: Normalization factors (column norms)
    """
    norm = jnp.sqrt(jnp.sum(M**2, axis=0))
    # Avoid division by zero
    norm = jnp.where(norm == 0, 1.0, norm)
    return M / norm[None, :], norm


def wls_solve_svd(
    residuals: jnp.ndarray,
    sigma: jnp.ndarray,
    M: jnp.ndarray,
    threshold: float = 1e-14,
    negate_dpars: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Solve weighted least squares using SVD - matches PINT's fit_wls_svd exactly.
    
    Args:
        residuals: Timing residuals in seconds (N,)
        sigma: TOA uncertainties in seconds (N,)
        M: Design matrix (N, n_params) with PINT convention: M = -d(phase)/d(param) / F0
        threshold: SVD threshold for singularity handling
        negate_dpars: If True, negate dpars. Should be False for PINT-style design matrix.
                      Set True only if your design matrix has opposite sign convention.
        
    Returns:
        dpars: Parameter updates
        Sigma: Parameter covariance matrix
        Adiag: Design matrix normalization factors
    """
    # Move data to CPU for stability. SVD on GPU (cuSolver) can be flaky with dense matrices
    # or under thread contention, causing "gpusolverDnCreate" errors.
    # We attempt to move to CPU only if running eagerly (not tracing).
    try:
        # Simple heuristic: concrete arrays usually have a 'device' attribute or buffer interface
        # Tracers might not, or we just rely on try-except.
        # jax.device_put works on Tracers too (inserts valid primitive), so we just try it.
        cpu = jax.devices("cpu")[0]
        residuals = jax.device_put(residuals, cpu)
        sigma = jax.device_put(sigma, cpu)
        M = jax.device_put(M, cpu)
    except Exception:
        # Ignore errors (e.g. no CPU device, or tracing issues) and proceed with default device
        pass

    # Step 1: Weight residuals by uncertainties
    # r1 = N^{-0.5} r
    r1 = residuals / sigma
    
    # Step 2: Weight design matrix by uncertainties
    # M1 = N^{-0.5} M
    M1 = M / sigma[:, None]
    
    # Step 3: Normalize design matrix columns for numerical stability
    # M2 = M1 A^{-1}
    M2, Adiag = normalize_designmatrix(M1)
    
    # Step 4: SVD decomposition
    # M2 = U S V^T
    U, Sdiag, VT = jnp.linalg.svd(M2, full_matrices=False)
    
    # Step 5: Apply threshold to singular values
    # Replace small singular values with inf (equivalent to zero in inverse)
    max_singular = jnp.max(Sdiag)
    Sdiag = jnp.where(Sdiag < threshold * max_singular, jnp.inf, Sdiag)
    
    # Step 6: Compute covariance matrix
    # Sigma = (M2^T M2)^{-1} = V (S^T S)^{-1} V^T
    # Then unnormalize: Sigma_final = A^{-1} Sigma A^{-1}
    Sigma_ = (VT.T / (Sdiag**2)) @ VT
    Sigma = (Sigma_ / Adiag).T / Adiag
    
    # Step 7: Compute parameter updates
    # dpars = V S^{-1} U^T r1
    # Then unnormalize: dpars_final = A^{-1} dpars
    dpars = (VT.T @ ((U.T @ r1) / Sdiag)) / Adiag
    if negate_dpars:
        dpars = -dpars
    
    return dpars, Sigma, Adiag


def wls_iteration_jax(
    residual_fn: Callable,
    params: jnp.ndarray,
    times: jnp.ndarray,
    freqs: jnp.ndarray,
    sigma: jnp.ndarray,
    threshold: float = 1e-14
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Single WLS iteration using JAX autodiff for design matrix.
    
    This version requires JAX-compatible residual functions.
    
    Args:
        residual_fn: JAX-compatible function that computes residuals given params
        params: Current parameter values
        times: TOA times
        freqs: TOA frequencies
        sigma: TOA uncertainties
        threshold: SVD threshold
        
    Returns:
        dpars: Parameter updates
        new_params: Updated parameters
        Sigma: Parameter covariance matrix
    """
    # Compute residuals at current parameters
    residuals = residual_fn(params, times, freqs)
    
    # Compute design matrix via autodiff
    # M[i,j] = d(residual_i) / d(param_j)
    def residuals_for_jac(p):
        return residual_fn(p, times, freqs)
    
    # jacfwd gives us (n_toas, n_params) directly - no transpose needed!
    M = jax.jacfwd(residuals_for_jac)(params)
    
    # Solve WLS problem (negate_dpars=True because M = d(residual)/d(param))
    dpars, Sigma, Adiag = wls_solve_svd(residuals, sigma, M, threshold, negate_dpars=True)
    
    # Update parameters
    new_params = params + dpars
    
    return dpars, new_params, Sigma


def wls_iteration_numerical(
    residual_fn: Callable,
    params: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    sigma: np.ndarray,
    threshold: float = 1e-14,
    eps: float = 1e-8  # Much larger default step for numerical stability!
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single WLS iteration using numerical derivatives for design matrix.
    
    This version works with non-JAX residual functions (like PINT).
    
    Args:
        residual_fn: Function that computes residuals given params  
        params: Current parameter values
        times: TOA times
        freqs: TOA frequencies
        sigma: TOA uncertainties
        threshold: SVD threshold
        eps: Step size for numerical derivatives
        
    Returns:
        dpars: Parameter updates
        new_params: Updated parameters
        Sigma: Parameter covariance matrix
    """
    n_params = len(params)
    n_toas = len(sigma)
    
    # Compute residuals at current parameters
    residuals = residual_fn(params, times, freqs)
    residuals = np.array(residuals, dtype=np.float64)
    
    # Compute design matrix via numerical differentiation
    M = np.zeros((n_toas, n_params), dtype=np.float64)
    for j in range(n_params):
        params_plus = params.copy()
        # Use relative step size based on parameter magnitude
        # This ensures we stay above floating point precision limits
        param_scale = max(abs(params[j]), 1.0)
        step = eps * param_scale
        
        params_plus[j] += step
        residuals_plus = residual_fn(params_plus, times, freqs)
        residuals_plus = np.array(residuals_plus, dtype=np.float64)
        M[:, j] = (residuals_plus - residuals) / step
    
    # Convert to JAX arrays for the linear algebra
    M_jax = jnp.array(M)
    residuals_jax = jnp.array(residuals)
    sigma_jax = jnp.array(sigma)
    
    # Solve WLS problem (negate_dpars=True because M = d(residual)/d(param))
    dpars, Sigma, Adiag = wls_solve_svd(residuals_jax, sigma_jax, M_jax, threshold, negate_dpars=True)
    
    # Update parameters
    new_params = params + np.array(dpars)
    
    return np.array(dpars), new_params, np.array(Sigma)


def fit_wls_numerical(
    residual_fn: Callable,
    initial_params: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    sigma: np.ndarray,
    maxiter: int = 5,
    threshold: float = None,
    eps: float = 1e-8,  # Larger default step
    damping: float = 1.0,
    param_bounds: list = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit parameters using WLS with numerical derivatives.
    
    This version works with non-JAX residual functions (like PINT).
    
    Args:
        residual_fn: Residual function (doesn't need to be JAX-compatible)
        initial_params: Initial parameter guesses
        times: TOA times
        freqs: TOA frequencies  
        sigma: TOA uncertainties
        maxiter: Maximum number of iterations
        threshold: SVD threshold (default: 1e-14 * max(M.shape))
        eps: Step size for numerical derivatives
        damping: Damping factor for parameter updates (1.0 = no damping)
        param_bounds: List of (min, max) tuples for each parameter (None = no bound)
        
    Returns:
        final_params: Fitted parameters
        param_errors: Parameter uncertainties (1-sigma)
        chi2: Final chi-squared value
    """
    params = np.array(initial_params, dtype=np.float64)
    
    if threshold is None:
        threshold = 1e-14 * max(len(times), len(initial_params))
    
    # Iterate
    for i in range(maxiter):
        dpars, params_new, Sigma = wls_iteration_numerical(
            residual_fn, params, times, freqs, sigma, threshold, eps
        )
        
        # Apply damping factor and bounds
        params_test = params + damping * dpars
        
        # Apply bounds if provided
        if param_bounds is not None:
            for j, (pmin, pmax) in enumerate(param_bounds):
                if pmin is not None and params_test[j] < pmin:
                    params_test[j] = pmin
                if pmax is not None and params_test[j] > pmax:
                    params_test[j] = pmax
        
        params = params_test
        
        print(f"  Iteration {i+1}: max |dpars| = {np.max(np.abs(dpars)):.6e}")
        
        # Check convergence (optional)
        if np.max(np.abs(dpars)) < 1e-15:
            print(f"  Converged after {i+1} iterations")
            break
    
    # Compute final chi-squared and covariance at converged parameters
    residuals = residual_fn(params, times, freqs)
    residuals = np.array(residuals, dtype=np.float64)
    chi2 = np.sum((residuals / sigma) ** 2)
    
    # Recompute design matrix and covariance at final parameters
    n_params = len(params)
    n_toas = len(sigma)
    M = np.zeros((n_toas, n_params), dtype=np.float64)
    for j in range(n_params):
        params_plus = params.copy()
        param_scale = max(abs(params[j]), 1.0)
        step = eps * param_scale
        
        params_plus[j] += step
        residuals_plus = residual_fn(params_plus, times, freqs)
        residuals_plus = np.array(residuals_plus, dtype=np.float64)
        M[:, j] = (residuals_plus - residuals) / step
    
    # Compute final covariance
    M_jax = jnp.array(M)
    residuals_jax = jnp.array(residuals)
    sigma_jax = jnp.array(sigma)
    _, Sigma_final, _ = wls_solve_svd(residuals_jax, sigma_jax, M_jax, threshold, negate_dpars=True)
    
    # Extract uncertainties
    param_errors = np.sqrt(np.diag(Sigma_final))
    
    return params, param_errors, float(chi2)


def fit_wls_jax(
    residual_fn: Callable,
    initial_params: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    sigma: np.ndarray,
    maxiter: int = 5,
    threshold: float = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit parameters using WLS with multiple iterations.
    
    Args:
        residual_fn: JAX-compatible residual function
        initial_params: Initial parameter guesses
        times: TOA times
        freqs: TOA frequencies  
        sigma: TOA uncertainties
        maxiter: Maximum number of iterations
        threshold: SVD threshold (default: 1e-14 * max(M.shape))
        
    Returns:
        final_params: Fitted parameters
        param_errors: Parameter uncertainties (1-sigma)
        chi2: Final chi-squared value
    """
    # Convert to JAX arrays
    params = jnp.array(initial_params)
    times_jax = jnp.array(times)
    freqs_jax = jnp.array(freqs)
    sigma_jax = jnp.array(sigma)
    
    if threshold is None:
        threshold = 1e-14 * max(len(times), len(initial_params))
    
    # Iterate
    for i in range(maxiter):
        dpars, params, Sigma = wls_iteration_jax(
            residual_fn, params, times_jax, freqs_jax, sigma_jax, threshold
        )
        
        print(f"  Iteration {i+1}: max |dpars| = {jnp.max(jnp.abs(dpars)):.6e}")
        
        # Check convergence (optional)
        if jnp.max(jnp.abs(dpars)) < 1e-15:
            print(f"  Converged after {i+1} iterations")
            break
    
    # Compute final chi-squared
    residuals = residual_fn(params, times_jax, freqs_jax)
    chi2 = jnp.sum((residuals / sigma_jax) ** 2)
    
    # Extract uncertainties
    param_errors = jnp.sqrt(jnp.diag(Sigma))
    
    return np.array(params), np.array(param_errors), float(chi2)
