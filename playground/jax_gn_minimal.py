"""
JAX Gauss-Newton Prototype - MINIMAL VERSION
=============================================

Simplest possible test: F0+F1 fitting only with jaxopt.GaussNewton

Goal: Prove that JAX autodiff + GN works before adding complexity.

Author: JUG Development Team
Date: 2025-12-05
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxopt
import time

from jug.io.par_reader import parse_par_file
from jug.residuals.simple_calculator import compute_residuals_simple


def residual_fun_f0f1(params, dt_sec, errors):
    """
    Compute weighted residuals for F0+F1 fitting.
    
    Pure JAX, no loops, no conditionals - just math.
    
    Parameters
    ----------
    params : array [F0, F1]
    dt_sec : array, time from PEPOCH to emission  
    errors : array, TOA uncertainties in seconds
    
    Returns
    -------
    weighted_residuals : array
    """
    F0, F1 = params[0], params[1]
    
    # Phase = F0*dt + F1*dt^2/2
    phase = F0 * dt_sec + (F1 / 2.0) * (dt_sec ** 2)
    
    # Wrap phase
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert to time
    residuals = phase_wrapped / F0
    
    # Subtract weighted mean
    weights = 1.0 / (errors ** 2)
    mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    residuals = residuals - mean
    
    # Weight for WLS
    weighted_residuals = residuals / errors
    
    return weighted_residuals


def test_minimal_gn():
    """Test jaxopt.GaussNewton on F0+F1 fitting."""
    print("="*70)
    print("MINIMAL JAX GAUSS-NEWTON TEST")
    print("="*70)
    print("\nObjective: Prove JAX autodiff + GN works for pulsar timing")
    print("Test case: F0+F1 fitting on J1909-3744")
    
    # Get data
    print("\n1. Computing initial residuals...")
    result = compute_residuals_simple(
        "data/pulsars/J1909-3744_tdb.par",
        "data/pulsars/J1909-3744.tim",
        clock_dir="data/clock",
        verbose=False
    )
    
    # Extract what we need
    dt_sec = jnp.array(result['dt_sec'])
    errors = jnp.array(result['errors_us']) * 1e-6  # μs → seconds
    
    # Initial parameters
    params = parse_par_file("data/pulsars/J1909-3744_tdb.par")
    initial_params = jnp.array([params['F0'], params['F1']])
    
    print(f"   Initial F0: {initial_params[0]:.15f} Hz")
    print(f"   Initial F1: {initial_params[1]:.15e} Hz/s")
    print(f"   Initial RMS: {result['rms_us']:.6f} μs")
    
    # Create solver
    print("\n2. Creating jaxopt.GaussNewton solver...")
    solver = jaxopt.GaussNewton(
        residual_fun=residual_fun_f0f1,
        maxiter=10,
        tol=1e-12
    )
    
    # Run solver
    print("\n3. Running Gauss-Newton...")
    print("   (First call will JIT-compile...)")
    start = time.time()
    result_solver = solver.run(initial_params, dt_sec=dt_sec, errors=errors)
    elapsed = time.time() - start
    
    # Extract results
    final_params = result_solver.params
    final_state = result_solver.state
    
    # Compute final RMS
    final_residuals_weighted = residual_fun_f0f1(final_params, dt_sec, errors)
    final_residuals = final_residuals_weighted * errors  # Unweight
    final_rms = float(jnp.sqrt(jnp.mean(final_residuals ** 2)))
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nConverged: {final_state.error < 1e-12}")
    print(f"Iterations: {final_state.iter_num}")
    print(f"Final error: {final_state.error:.3e}")
    print(f"Total time: {elapsed:.3f}s")
    
    print(f"\nFinal F0: {final_params[0]:.15f} Hz")
    print(f"Final F1: {final_params[1]:.15e} Hz/s")
    print(f"\nΔF0: {(final_params[0] - initial_params[0]):+.3e} Hz")
    print(f"ΔF1: {(final_params[1] - initial_params[1]):+.3e} Hz/s")
    
    print(f"\nInitial RMS: {result['rms_us']:.6f} μs")
    print(f"Final RMS:   {final_rms*1e6:.6f} μs")
    print(f"Improvement: {(result['rms_us'] - final_rms*1e6):.6f} μs")
    
    print(f"\n**Expected final RMS: ~0.404 μs**")
    print(f"**Actual final RMS:   {final_rms*1e6:.6f} μs**")
    
    if abs(final_rms*1e6 - 0.404) < 0.001:
        print("\n✅ SUCCESS! Result matches expected value.")
    else:
        print("\n⚠️  Result differs from expected. Needs investigation.")
    
    print("="*70)
    
    return final_params, final_rms


if __name__ == "__main__":
    test_minimal_gn()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If this test passed:
✅ JAX autodiff works for pulsar timing residuals
✅ jaxopt.GaussNewton converges correctly  
✅ Results match expected RMS

Next steps:
1. Extend to F0+F1+DM+DM1 (add DM delay correction)
2. Add Levenberg-Marquardt fallback
3. Add parameter uncertainties from covariance
4. Benchmark speed vs current fitter
5. Integrate into production code

To run:
    python playground/jax_gn_minimal.py
""")
