#!/usr/bin/env python3
"""
Detailed breakdown of JUG workflow with precise timing of each component.
"""

import time
import numpy as np
from pathlib import Path
import sys
import os

# Suppress warnings
os.environ['JAX_LOG_COMPILES'] = '0'

print("="*80)
print("JUG WORKFLOW DETAILED BREAKDOWN")
print("="*80)

par_file = Path("data/pulsars/J1909-3744_tdb_wrong.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

# We'll run the workflow twice to see cold vs warm performance
N_RUNS = 2

all_results = []

for run_idx in range(N_RUNS):
    print(f"\n{'='*80}")
    print(f"RUN {run_idx+1}: {'COLD START' if run_idx == 0 else 'WARM START (JIT cached)'}")
    print("="*80)
    
    times = {}
    total_start = time.time()
    
    # ========================================================================
    # STEP 1: File Parsing
    # ========================================================================
    print("\n[1/7] File Parsing")
    t0 = time.time()
    
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    
    params = parse_par_file(par_file)
    toas_data = parse_tim_file_mjds(tim_file)
    
    times['parse'] = time.time() - t0
    print(f"  ✓ Parsed .par and .tim files: {times['parse']:.3f}s")
    print(f"    - Parameters: {len(params)} entries")
    print(f"    - TOAs: {len(toas_data)} measurements")
    
    # ========================================================================
    # STEP 2: Initialize Fitter (Cache Building)
    # ========================================================================
    print("\n[2/7] Cache Initialization")
    t0 = time.time()
    
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    # This computes dt_sec (all delays) once and caches it
    result = compute_residuals_simple(
        par_file, tim_file, 
        verbose=False,
        subtract_tzr=False
    )
    dt_sec_cached = result['dt_sec']
    
    times['cache_build'] = time.time() - t0
    print(f"  ✓ Built delay cache: {times['cache_build']:.3f}s")
    print(f"    - Cached dt_sec for {len(dt_sec_cached)} TOAs")
    
    # ========================================================================
    # STEP 3: Extract TOA Data
    # ========================================================================
    print("\n[3/7] Extract TOA Data")
    t0 = time.time()
    
    errors_us = np.array([toa.error_us for toa in toas_data])
    errors_sec = errors_us * 1e-6
    weights = 1.0 / errors_sec**2
    freq_mhz = np.array([toa.freq_mhz for toa in toas_data])
    
    times['extract_data'] = time.time() - t0
    print(f"  ✓ Extracted TOA data: {times['extract_data']:.3f}s")
    print(f"    - Errors, frequencies, weights")
    
    # ========================================================================
    # STEP 4: Setup JAX Functions (JIT Compilation)
    # ========================================================================
    print("\n[4/7] JAX Function Setup")
    t0 = time.time()
    
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    
    from jug.fitting.derivatives_spin import compute_spin_derivatives
    from jug.residuals.phase import compute_residuals_from_dt
    
    # Convert to JAX arrays
    dt_sec_jax = jnp.array(dt_sec_cached)
    freq_mhz_jax = jnp.array(freq_mhz)
    errors_jax = jnp.array(errors_sec)
    weights_jax = jnp.array(weights)
    
    times['jax_setup'] = time.time() - t0
    print(f"  ✓ JAX arrays created: {times['jax_setup']:.3f}s")
    
    # ========================================================================
    # STEP 5: First Residual Computation (Triggers JIT)
    # ========================================================================
    print("\n[5/7] Initial Residual Computation")
    t0 = time.time()
    
    f0_start = float(params['F0'])
    f1_start = float(params.get('F1', 0.0))
    
    # This will trigger JIT compilation on first run
    residuals_sec = compute_residuals_from_dt(
        dt_sec_jax,
        freq_mhz_jax,
        f0_start,
        f1_start,
        0.0,  # f2
        float(params.get('DM', 0.0)),
        0.0,  # dm1
        0.0,  # dm2
        subtract_mean=True
    )
    residuals_sec = np.array(residuals_sec)
    
    times['first_residual'] = time.time() - t0
    rms_prefit = np.sqrt(np.average(residuals_sec**2, weights=weights))
    print(f"  ✓ Initial residuals computed: {times['first_residual']:.3f}s")
    print(f"    - Prefit RMS: {rms_prefit*1e6:.3f} μs")
    
    # ========================================================================
    # STEP 6: Fitting Loop
    # ========================================================================
    print("\n[6/7] Fitting Iterations (F0 + F1)")
    t0 = time.time()
    
    f0 = f0_start
    f1 = f1_start
    max_iter = 15
    
    for iteration in range(max_iter):
        # Compute derivatives
        derivs = compute_spin_derivatives(
            {'F0': f0, 'F1': f1},
            dt_sec_jax,
            ['F0', 'F1'],
            freq_mhz_jax
        )
        
        # Build design matrix
        M = np.column_stack([derivs['F0'], derivs['F1']])
        
        # WLS solve
        from jug.fitting.wls_fitter import wls_solve_svd
        delta_params, cov, _ = wls_solve_svd(
            residuals_sec,
            errors_sec,
            M,
            negate_dpars=False
        )
        
        # Update parameters
        f0 += delta_params[0]
        f1 += delta_params[1]
        
        # Recompute residuals
        residuals_sec = compute_residuals_from_dt(
            dt_sec_jax,
            freq_mhz_jax,
            f0, f1, 0.0,
            float(params.get('DM', 0.0)),
            0.0, 0.0,
            subtract_mean=True
        )
        residuals_sec = np.array(residuals_sec)
        
        # Check convergence
        if np.max(np.abs(delta_params)) < 1e-15:
            print(f"    ✓ Converged at iteration {iteration+1}")
            break
    
    times['fitting'] = time.time() - t0
    rms_postfit = np.sqrt(np.average(residuals_sec**2, weights=weights))
    print(f"  ✓ Fitting completed: {times['fitting']:.3f}s")
    print(f"    - Iterations: {iteration+1}")
    print(f"    - Postfit RMS: {rms_postfit*1e6:.3f} μs")
    
    # ========================================================================
    # STEP 7: Summary Statistics
    # ========================================================================
    print("\n[7/7] Final Results")
    t0 = time.time()
    
    # Compute final uncertainties from covariance
    uncertainties = np.sqrt(np.diag(cov))
    
    times['finalize'] = time.time() - t0
    print(f"  ✓ Computed uncertainties: {times['finalize']:.3f}s")
    print(f"    - F0: {f0:.20e} ± {uncertainties[0]:.2e} Hz")
    print(f"    - F1: {f1:.20e} ± {uncertainties[1]:.2e} Hz/s")
    
    # ========================================================================
    # TOTAL TIME
    # ========================================================================
    times['total'] = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"TOTAL TIME: {times['total']:.3f}s")
    print("="*80)
    
    # ========================================================================
    # BREAKDOWN
    # ========================================================================
    print(f"\nDetailed Breakdown:")
    print(f"  1. File parsing:            {times['parse']:.3f}s  ({100*times['parse']/times['total']:5.1f}%)")
    print(f"  2. Cache initialization:    {times['cache_build']:.3f}s  ({100*times['cache_build']/times['total']:5.1f}%)")
    print(f"  3. Extract TOA data:        {times['extract_data']:.3f}s  ({100*times['extract_data']/times['total']:5.1f}%)")
    print(f"  4. JAX setup:               {times['jax_setup']:.3f}s  ({100*times['jax_setup']/times['total']:5.1f}%)")
    print(f"  5. First residual (JIT):    {times['first_residual']:.3f}s  ({100*times['first_residual']/times['total']:5.1f}%)")
    print(f"  6. Fitting iterations:      {times['fitting']:.3f}s  ({100*times['fitting']/times['total']:5.1f}%)")
    print(f"  7. Finalize:                {times['finalize']:.3f}s  ({100*times['finalize']/times['total']:5.1f}%)")
    print(f"  {'─'*60}")
    print(f"  TOTAL:                      {times['total']:.3f}s  (100.0%)")
    
    all_results.append(times)

# ============================================================================
# COMPARISON: Cold vs Warm
# ============================================================================
print("\n" + "="*80)
print("COLD START vs WARM START COMPARISON")
print("="*80)

print(f"\n{'Component':<30} {'Cold (Run 1)':<15} {'Warm (Run 2)':<15} {'Speedup':<10}")
print("-" * 80)

cold = all_results[0]
warm = all_results[1]

for key in ['parse', 'cache_build', 'extract_data', 'jax_setup', 
            'first_residual', 'fitting', 'finalize', 'total']:
    speedup = cold[key] / warm[key] if warm[key] > 0 else 1.0
    print(f"{key.replace('_', ' ').title():<30} {cold[key]:>8.3f}s      {warm[key]:>8.3f}s      {speedup:>6.2f}x")

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

speedup_total = cold['total'] / warm['total']
print(f"\n1. Overall speedup (warm vs cold): {speedup_total:.2f}x")
print(f"   - Cold start: {cold['total']:.3f}s")
print(f"   - Warm start: {warm['total']:.3f}s")

print(f"\n2. Biggest improvement: First residual computation")
speedup_residual = cold['first_residual'] / warm['first_residual']
print(f"   - Cold: {cold['first_residual']:.3f}s (JIT compiling)")
print(f"   - Warm: {warm['first_residual']:.3f}s (JIT cached)")
print(f"   - Speedup: {speedup_residual:.2f}x")

print(f"\n3. Cache initialization is consistent:")
print(f"   - Cold: {cold['cache_build']:.3f}s")
print(f"   - Warm: {warm['cache_build']:.3f}s")
print(f"   - (One-time cost, not affected by JIT)")

print(f"\n4. Fitting is faster when warm:")
speedup_fitting = cold['fitting'] / warm['fitting']
print(f"   - Cold: {cold['fitting']:.3f}s")
print(f"   - Warm: {warm['fitting']:.3f}s")
print(f"   - Speedup: {speedup_fitting:.2f}x")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print(f"\nAfter JIT warmup, JUG runs at {warm['total']:.3f}s per fit.")
print(f"This is the typical production performance.")
print(f"\nThe cold start overhead of {cold['total'] - warm['total']:.3f}s is paid only once,")
print(f"then amortized across all subsequent fits.")
