"""Test if we can JAX-compile the entire fitting iteration."""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import time

@jax.jit
def wls_solve_jax(residuals, errors, M):
    """JAX-compiled WLS solver using SVD."""
    # Weight the system
    weights = 1.0 / errors
    M_weighted = M * weights[:, None]
    r_weighted = residuals * weights
    
    # Solve using JAX's lstsq (uses SVD internally)
    delta_params, residuals_lstsq, rank, s = jnp.linalg.lstsq(M_weighted, r_weighted, rcond=None)
    
    # Compute covariance (simplified)
    cov = jnp.linalg.inv(M_weighted.T @ M_weighted)
    
    return delta_params, cov

@jax.jit
def full_iteration_jax(dt_sec, f0, f1, errors, weights):
    """Complete fitting iteration in JAX."""
    # Compute phase
    f1_half = f1 / 2.0
    phase = dt_sec * (f0 + dt_sec * f1_half)
    phase_wrapped = phase - jnp.round(phase)
    residuals = phase_wrapped / f0
    
    # Subtract weighted mean
    weighted_mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    residuals = residuals - weighted_mean
    
    # Compute derivatives
    d_f0 = -(dt_sec / f0)
    d_f1 = -(dt_sec**2 / 2.0) / f0
    
    # Subtract mean from derivatives
    d_f0 = d_f0 - jnp.sum(d_f0 * weights) / jnp.sum(weights)
    d_f1 = d_f1 - jnp.sum(d_f1 * weights) / jnp.sum(weights)
    
    # Build design matrix
    M = jnp.column_stack([d_f0, d_f1])
    
    # WLS solve
    delta_params, cov = wls_solve_jax(residuals, errors, M)
    
    # RMS
    rms = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights)) * 1e6
    
    return delta_params, rms, cov

# Test with dummy data
n_toas = 10408
dt_sec = jnp.array(np.random.randn(n_toas) * 1e8)
errors = jnp.array(np.ones(n_toas) * 1e-6)
weights = 1.0 / errors**2
f0 = 339.3
f1 = -1.6e-15

print("Testing full JAX iteration...")
print("Warming up JIT...")
start = time.time()
delta, rms, cov = full_iteration_jax(dt_sec, f0, f1, errors, weights)
warmup = time.time() - start
print(f"  Warmup (JIT compile): {warmup:.3f}s")

print("\nTiming compiled iterations...")
times = []
for i in range(10):
    start = time.time()
    delta, rms, cov = full_iteration_jax(dt_sec, f0, f1, errors, weights)
    times.append(time.time() - start)

print(f"  Average per iteration: {np.mean(times):.6f}s")
print(f"  Min: {np.min(times):.6f}s")
print(f"  Max: {np.max(times):.6f}s")

print(f"\nIf we can get this down to ~0.01s per iteration:")
print(f"  16 iterations × 0.01s = 0.16s")
print(f"  + 2.7s cache + 0.2s JIT = 3.1s total")
print(f"  = 6.8x speedup vs original!")
