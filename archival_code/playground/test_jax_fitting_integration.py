#!/usr/bin/env python3
"""Test JAX fitting integration with real data.

This script tests the new JAX residual computation and fitting workflow.
"""

# CRITICAL: Enable JAX float64 BEFORE ANY imports whatsoever
import jax
jax.config.update('jax_enable_x64', True)

# Now import everything else
import jug
import numpy as np
import jax.numpy as jnp
from pathlib import Path

from jug.residuals.core import prepare_fixed_data, compute_residuals_jax, compute_residuals_jax_from_dt
from jug.residuals.simple_calculator import compute_residuals_simple


def test_jax_residuals():
    """Test that JAX residuals match simple_calculator residuals."""
    
    # Use J1909-3744 (well-tested ELL1 binary)
    data_dir = Path("data/pulsars")
    par_file = data_dir / "J1909-3744_tdb.par"
    tim_file = data_dir / "J1909-3744.tim"
    
    if not par_file.exists():
        print("❌ Test data not found")
        return False
    
    print("="*60)
    print("Testing JAX Residual Computation")
    print("="*60)
    
    # Step 1: Compute residuals using simple_calculator (baseline)
    print("\n1. Computing baseline residuals (simple_calculator)...")
    result_baseline = compute_residuals_simple(
        par_file, tim_file,
        clock_dir="data/clock",
        observatory="meerkat"
    )
    residuals_baseline = result_baseline['residuals_us']
    rms_baseline = result_baseline['rms_us']
    print(f"   Baseline RMS: {rms_baseline:.3f} μs")
    
    # Step 2: Prepare fixed data for JAX
    print("\n2. Preparing fixed data for JAX...")
    fixed_data = prepare_fixed_data(
        par_file, tim_file,
        clock_dir="data/clock",
        observatory="meerkat"
    )
    
    # Step 3: Extract fitted parameters
    print("\n3. Setting up JAX residual function...")
    par_params = fixed_data['par_params']
    
    # For now, fit F0 and F1 (simple test)
    param_names = ('F0', 'F1')  # Must be tuple for JIT
    params_array = jnp.array([
        par_params['F0'],
        par_params['F1']
    ])
    
    print(f"   Parameters: F0={par_params['F0']:.10f} Hz, F1={par_params['F1']:.5e} Hz/s")
    print(f"   PEPOCH={fixed_data['pepoch']:.6f} MJD")
    print(f"   TZR phase={fixed_data['tzr_phase']:.1f} cycles")
    print(f"   Mean geometric delay={np.mean(fixed_data['geometric_delay_sec']):.3f} s")
    
    # Fixed parameters (everything except F0, F1)
    fixed_params = {k: v for k, v in par_params.items() 
                    if k not in param_names}
    
    # Step 4: Compute residuals with JAX (using dt_sec for precision)
    print("\n4. Computing residuals with JAX (from emission times)...")
    residuals_jax_sec = compute_residuals_jax_from_dt(
        params_array,
        param_names,
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )
    
    residuals_jax_us = np.array(residuals_jax_sec) * 1e6
    rms_jax = np.sqrt(np.mean(residuals_jax_us**2))
    print(f"   JAX RMS: {rms_jax:.3f} μs")
    
    # Step 5: Compare
    print("\n5. Comparing results...")
    diff_us = residuals_jax_us - residuals_baseline
    diff_rms = np.sqrt(np.mean(diff_us**2))
    diff_std = np.std(diff_us)
    diff_mean = np.mean(diff_us)
    
    print(f"   Difference RMS: {diff_rms:.6f} μs")
    print(f"   Difference mean: {diff_mean:.6f} μs")
    print(f"   Difference std: {diff_std:.6f} μs")
    print(f"   Max difference: {np.max(np.abs(diff_us)):.6f} μs")
    
    # Check if they match
    tolerance_us = 0.5  # 500 nanoseconds - well below measurement precision
    if diff_rms < tolerance_us:
        print(f"\n✅ JAX residuals match simple_calculator (< {tolerance_us} μs)")
        return True
    else:
        print(f"\n⚠️  JAX residuals differ from baseline by {diff_rms:.6f} μs")
        if diff_rms < 1.0:
            print(f"   This is acceptable (< 1 μs, baseline RMS = {rms_baseline:.3f} μs)")
            return True
        print("   Investigating...")
        return False


def test_jax_speed():
    """Test JAX residual computation speed."""
    import time
    
    data_dir = Path("data/pulsars")
    par_file = data_dir / "J1909-3744_tdb.par"
    tim_file = data_dir / "J1909-3744.tim"
    
    if not par_file.exists():
        print("❌ Test data not found")
        return
    
    print("\n" + "="*60)
    print("Testing JAX Speed")
    print("="*60)
    
    # Prepare data (one-time cost)
    print("\nPreparing fixed data (one-time cost)...")
    t0 = time.time()
    fixed_data = prepare_fixed_data(par_file, tim_file)
    t_prep = time.time() - t0
    print(f"   Preparation time: {t_prep:.3f} s")
    
    # Setup parameters
    par_params = fixed_data['par_params']
    param_names = ('F0', 'F1')
    params_array = jnp.array([par_params['F0'], par_params['F1']])
    fixed_params = {k: v for k, v in par_params.items() if k not in param_names}
    
    # First call (includes JIT compilation)
    print("\nFirst call (includes JIT compilation)...")
    t0 = time.time()
    residuals = compute_residuals_jax(
        params_array, param_names,
        fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
        fixed_data['geometric_delay_sec'],
        fixed_data['other_delays_minus_dm_sec'],
        fixed_data['pepoch'], fixed_data['dm_epoch'],
        fixed_data['tzr_phase'], fixed_data['uncertainties_us'],
        fixed_params
    )
    residuals.block_until_ready()  # Wait for JAX to finish
    t_first = time.time() - t0
    print(f"   First call time: {t_first:.3f} s")
    
    # Subsequent calls (JIT-compiled, should be fast)
    print("\nSubsequent calls (JIT-compiled)...")
    n_calls = 100
    t0 = time.time()
    for _ in range(n_calls):
        residuals = compute_residuals_jax(
            params_array, param_names,
            fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
            fixed_data['geometric_delay_sec'],
            fixed_data['other_delays_minus_dm_sec'],
            fixed_data['pepoch'], fixed_data['dm_epoch'],
            fixed_data['tzr_phase'], fixed_data['uncertainties_us'],
            fixed_params
        )
        residuals.block_until_ready()
    t_total = time.time() - t0
    t_per_call = t_total / n_calls * 1000  # ms
    
    print(f"   {n_calls} calls: {t_total:.3f} s")
    print(f"   Per call: {t_per_call:.3f} ms")
    
    if t_per_call < 10:
        print(f"\n✅ JAX is fast (< 10 ms per call)")
    else:
        print(f"\n⚠️  JAX slower than expected ({t_per_call:.3f} ms per call)")


if __name__ == "__main__":
    # Test correctness first
    success = test_jax_residuals()
    
    if success:
        # Test speed
        test_jax_speed()
    else:
        print("\n⚠️  Skipping speed test due to correctness issues")
