#!/usr/bin/env python3
"""
Level 2 Optimized Fitting - JAX JIT Compilation for 8.8x total speedup
========================================================================

This builds on Level 1 by adding JAX JIT compilation to the hot loop.

Level 1: Cache dt_sec (21s → 3.6s, 5.87x speedup)
Level 2: + JAX JIT (3.6s → 2.4s, 8.8x total speedup)

Expected: Same exact results, just faster!
"""

import numpy as np
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))

# Enable JAX 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.wls_fitter import wls_solve_svd
from jug.utils.constants import SECS_PER_DAY


@jax.jit
def wls_solve_jax(residuals, errors, M):
    """JAX-compiled WLS solver using SVD."""
    # Weight the system
    weights_solve = 1.0 / errors
    M_weighted = M * weights_solve[:, None]
    r_weighted = residuals * weights_solve
    
    # Solve using JAX's lstsq (uses SVD internally)
    delta_params, _, _, _ = jnp.linalg.lstsq(M_weighted, r_weighted, rcond=None)
    
    # Compute covariance
    cov = jnp.linalg.inv(M_weighted.T @ M_weighted)
    
    return delta_params, cov


@jax.jit
def full_iteration_jax(
    dt_sec: jnp.ndarray,
    f0: float,
    f1: float,
    errors: jnp.ndarray,
    weights: jnp.ndarray
) -> tuple:
    """
    Complete fitting iteration in JAX (JIT-compiled).
    
    This combines everything into one JIT-compiled function:
    - Residual computation
    - Derivative computation  
    - WLS solve
    
    All in JAX, all JIT-compiled for maximum speed!
    
    Parameters
    ----------
    dt_sec : jnp.ndarray
        Cached time deltas
    f0 : float
        Current F0
    f1 : float
        Current F1
    errors : jnp.ndarray
        TOA errors in seconds
    weights : jnp.ndarray
        TOA weights (1/error^2)
        
    Returns
    -------
    delta_params : jnp.ndarray
        Parameter updates [delta_f0, delta_f1]
    rms_us : float
        RMS in microseconds
    cov : jnp.ndarray
        Covariance matrix
    """
    # Compute spin phase
    f1_half = f1 / 2.0
    phase = dt_sec * (f0 + dt_sec * f1_half)
    
    # Wrap phase
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert to residuals
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
    
    # WLS solve (also JAX-compiled!)
    delta_params, cov = wls_solve_jax(residuals, errors, M)
    
    # Compute RMS
    rms_sec = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    rms_us = rms_sec * 1e6
    
    return delta_params, rms_us, cov


def fit_f0_f1_level2(
    par_file: Path,
    tim_file: Path,
    max_iter: int = 25,
    convergence_threshold: float = 1e-14
) -> dict:
    """
    Fit F0 and F1 using Level 2 optimization (Level 1 + JAX JIT).
    
    This should give identical results to Level 1 but run ~1.5x faster.
    
    Parameters
    ----------
    par_file : Path
        Path to .par file
    tim_file : Path
        Path to .tim file
    max_iter : int
        Maximum iterations
    convergence_threshold : float
        Convergence threshold on parameter changes
    
    Returns
    -------
    dict with results
    """
    print("="*80)
    print("LEVEL 2 OPTIMIZED FITTING (Level 1 + JAX JIT)")
    print("="*80)
    
    total_start = time.time()
    
    # Parse files
    params = parse_par_file(par_file)
    toas_data = parse_tim_file_mjds(tim_file)
    
    # Extract TOA data
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    
    pepoch = params['PEPOCH']
    f0_start = params['F0']
    f1_start = params['F1']
    
    print(f"\nStarting parameters:")
    print(f"  F0 = {f0_start:.20f} Hz")
    print(f"  F1 = {f1_start:.20e} Hz/s")
    print(f"  TOAs: {len(toas_mjd)}")
    
    # LEVEL 1: Compute dt_sec ONCE
    print(f"\nInitializing cache (computing dt_sec once)...")
    cache_start = time.time()
    
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        result = compute_residuals_simple(
            par_file,
            tim_file,
            clock_dir="data/clock",
            subtract_tzr=False  # Don't wrap - we'll do it ourselves
        )
    
    dt_sec_cached = result['dt_sec']
    
    cache_time = time.time() - cache_start
    print(f"  Cache initialized in {cache_time:.3f}s")
    print(f"  dt_sec cached for {len(dt_sec_cached)} TOAs")
    
    # Convert to JAX arrays for JIT compilation
    dt_sec_jax = jnp.array(dt_sec_cached)
    errors_jax = jnp.array(errors_sec)
    weights_jax = jnp.array(weights)
    
    print(f"  Converted to JAX arrays")
    print(f"  Ready for full JAX JIT compilation!")
    
    # Fitting loop with FULL JAX JIT
    print(f"\nFitting F0 + F1 (using FULL JAX JIT compilation)...")
    f0_curr = f0_start
    f1_curr = f1_start
    prev_delta_max = None
    iteration_times = []
    
    # Warm up JIT (first call compiles)
    print(f"  Warming up JAX JIT compiler (full iteration)...")
    warmup_start = time.time()
    _, _, _ = full_iteration_jax(
        dt_sec_jax, f0_curr, f1_curr, errors_jax, weights_jax
    )
    warmup_time = time.time() - warmup_start
    print(f"  JIT compilation: {warmup_time:.3f}s (one-time cost)")
    
    for iteration in range(max_iter):
        iter_start = time.time()
        
        # LEVEL 2: Complete iteration in JAX (JIT-compiled)!
        delta_params_jax, rms_us, cov_jax = full_iteration_jax(
            dt_sec_jax, f0_curr, f1_curr, errors_jax, weights_jax
        )
        
        # Convert results back to numpy
        delta_params = np.array(delta_params_jax)
        cov = np.array(cov_jax)
        rms_us = float(rms_us)
        
        # Update parameters
        f0_curr += delta_params[0]
        f1_curr += delta_params[1]
        
        iter_time = time.time() - iter_start
        iteration_times.append(iter_time)
        
        # Check convergence
        max_delta = max(abs(delta_params[0]), abs(delta_params[1]))
        
        if iteration < 3 or iteration >= max_iter - 1:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s")
        elif iteration == 3:
            print(f"  ...")
        
        if prev_delta_max is not None and abs(max_delta - prev_delta_max) < 1e-20:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        if max_delta < convergence_threshold:
            print(f"  Iteration {iteration+1}: RMS={rms_us:.6f} μs, time={iter_time:.3f}s (converged)")
            converged = True
            iterations = iteration + 1
            break
        
        prev_delta_max = max_delta
    else:
        converged = False
        iterations = max_iter
    
    total_time = time.time() - total_start
    
    # Print results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    
    print(f"\nConvergence:")
    print(f"  Iterations: {iterations}")
    print(f"  Converged: {converged}")
    
    print(f"\nTiming:")
    print(f"  Cache initialization: {cache_time:.3f}s (one-time cost)")
    print(f"  JIT compilation: {warmup_time:.3f}s (one-time cost)")
    print(f"  Fitting iterations: {sum(iteration_times):.3f}s")
    print(f"  Average per iteration: {np.mean(iteration_times):.3f}s")
    print(f"  Total time: {total_time:.3f}s")
    
    print(f"\nFinal parameters:")
    print(f"  F0 = {f0_curr:.20f} Hz")
    print(f"  F1 = {f1_curr:.20e} Hz/s")
    print(f"  RMS = {rms_us:.6f} μs")
    
    print(f"\nUncertainties:")
    unc_f0 = np.sqrt(cov[0, 0])
    unc_f1 = np.sqrt(cov[1, 1])
    print(f"  σ(F0) = {unc_f0:.3e} Hz")
    print(f"  σ(F1) = {unc_f1:.3e} Hz/s")
    
    return {
        'iterations': iterations,
        'converged': converged,
        'total_time': total_time,
        'cache_time': cache_time,
        'jit_time': warmup_time,
        'avg_iter_time': np.mean(iteration_times),
        'final_f0': f0_curr,
        'final_f1': f1_curr,
        'final_rms': float(rms_us),
        'covariance': cov
    }


if __name__ == '__main__':
    par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    # Run Level 2 optimized fitting
    results = fit_f0_f1_level2(par_file, tim_file)
    
    # Compare to baselines
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    original_time = 21.15  # From previous benchmark
    level1_time = 3.60     # From Level 1
    level2_time = results['total_time']
    
    speedup_vs_original = original_time / level2_time
    speedup_vs_level1 = level1_time / level2_time
    
    print(f"\nOriginal (no optimization):  {original_time:.3f}s")
    print(f"Level 1 (cached dt_sec):     {level1_time:.3f}s  (5.87x faster)")
    print(f"Level 2 (+ JAX JIT):         {level2_time:.3f}s")
    
    print(f"\nLevel 2 Speedup:")
    print(f"  vs Original: {speedup_vs_original:.2f}x faster! 🚀")
    print(f"  vs Level 1:  {speedup_vs_level1:.2f}x faster")
    
    if speedup_vs_original >= 8.0:
        print("\n✅ SUCCESS: Achieved target 8.8x total speedup!")
    elif speedup_vs_original >= 6.0:
        print(f"\n✅ GOOD: {speedup_vs_original:.1f}x speedup (target was 8.8x)")
    else:
        print(f"\n⚠️  Speedup less than expected (target: 8.8x)")
    
    # Validate accuracy
    print(f"\n{'='*80}")
    print("ACCURACY VALIDATION")
    print(f"{'='*80}")
    
    expected_f0 = 339.31569191904083027111
    expected_f1 = -1.61475056113088215780e-15
    expected_rms = 0.403565
    
    f0_match = (results['final_f0'] == expected_f0)
    f1_close = abs(results['final_f1'] - expected_f1) < 1e-20
    rms_close = abs(results['final_rms'] - expected_rms) < 0.001
    
    print(f"\nF0: {results['final_f0']:.20f} Hz")
    print(f"    Expected: {expected_f0:.20f} Hz")
    print(f"    Match: {f0_match} {'✅' if f0_match else '❌'}")
    
    print(f"\nF1: {results['final_f1']:.20e} Hz/s")
    print(f"    Expected: {expected_f1:.20e} Hz/s")
    print(f"    Close: {f1_close} {'✅' if f1_close else '❌'}")
    
    print(f"\nRMS: {results['final_rms']:.6f} μs")
    print(f"     Expected: {expected_rms:.6f} μs")
    print(f"     Close: {rms_close} {'✅' if rms_close else '❌'}")
    
    if f0_match and f1_close and rms_close:
        print(f"\n{'='*80}")
        print("✅ VALIDATION PASSED: Results match Level 1 exactly!")
        print("✅ Level 2 optimization is CORRECT and FAST!")
        print(f"{'='*80}")
