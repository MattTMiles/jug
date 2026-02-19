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
from typing import Tuple


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
    
    # Check for NaN in covariance (can occur with ill-conditioned matrices)
    # If NaN detected, use pseudoinverse approach
    if jnp.any(jnp.isnan(Sigma)):
        # Recompute using pseudoinverse: pinv(M2.T @ M2)
        M2TM2 = M2.T @ M2
        Sigma_ = jnp.linalg.pinv(M2TM2)
        Sigma = (Sigma_ / Adiag).T / Adiag
    
    # Step 7: Compute parameter updates
    # dpars = V S^{-1} U^T r1
    # Then unnormalize: dpars_final = A^{-1} dpars
    dpars = (VT.T @ ((U.T @ r1) / Sdiag)) / Adiag
    if negate_dpars:
        dpars = -dpars
    
    return dpars, Sigma, Adiag
