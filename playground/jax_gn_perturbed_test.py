"""
JAX Gauss-Newton - Perturbed Parameter Test
============================================

Test that jaxopt.GaussNewton can correct intentionally wrong parameters.

This proves:
1. JAX autodiff correctly computes derivatives
2. GaussNewton finds the minimum
3. The approach works for pulsar timing

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
    
    Parameters
    ----------
    params : array [F0, F1]
    dt_sec : array, time from PEPOCH to emission (computed with CORRECT F0/F1)
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


def compute_rms(params, dt_sec, errors):
    """Compute RMS from parameters."""
    weighted_resid = residual_fun_f0f1(params, dt_sec, errors)
    residuals = weighted_resid * errors
    weights = 1.0 / (errors ** 2)
    wrms = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    return float(wrms * 1e6)  # Convert to μs


def test_perturbed_parameters():
    """Test GN with intentionally wrong starting parameters."""
    print("="*70)
    print("PERTURBED PARAMETER TEST")
    print("="*70)
    print("\nObjective: Prove GaussNewton can correct wrong parameters")
    print("Method: Start with F0 intentionally wrong, see if GN fixes it")
    
    # Get data with CORRECT parameters
    print("\n1. Computing residuals with CORRECT parameters...")
    result = compute_residuals_simple(
        "data/pulsars/J1909-3744_tdb.par",
        "data/pulsars/J1909-3744.tim",
        clock_dir="data/clock",
        verbose=False
    )
    
    # Extract what we need
    dt_sec = jnp.array(result['dt_sec'])
    errors = jnp.array(result['errors_us']) * 1e-6  # μs → seconds
    
    # Get CORRECT parameters from par file
    params = parse_par_file("data/pulsars/J1909-3744_tdb.par")
    correct_f0 = params['F0']
    correct_f1 = params['F1']
    
    print(f"   Correct F0: {correct_f0:.15f} Hz")
    print(f"   Correct F1: {correct_f1:.15e} Hz/s")
    print(f"   RMS with correct params: {result['rms_us']:.6f} μs")
    
    # Create PERTURBED starting parameters
    # Make F0 wrong by 0.1% (should give ~100 μs RMS error)
    perturb_factor = 1.001  # 0.1% error
    wrong_f0 = correct_f0 * perturb_factor
    wrong_f1 = correct_f1  # Keep F1 correct for now
    
    perturbed_params = jnp.array([wrong_f0, wrong_f1])
    
    print(f"\n2. Creating PERTURBED starting parameters...")
    print(f"   Perturbed F0: {wrong_f0:.15f} Hz (+{(perturb_factor-1)*100:.1f}%)")
    print(f"   Perturbed F1: {wrong_f1:.15e} Hz/s (unchanged)")
    
    # Compute RMS with wrong parameters
    initial_rms = compute_rms(perturbed_params, dt_sec, errors)
    print(f"   RMS with perturbed params: {initial_rms:.6f} μs")
    print(f"   → Worse by {initial_rms - result['rms_us']:.6f} μs ✓")
    
    # Create solver
    print("\n3. Running Gauss-Newton to CORRECT the error...")
    solver = jaxopt.GaussNewton(
        residual_fun=residual_fun_f0f1,
        maxiter=20,
        tol=1e-12
    )
    
    # Run solver
    print("   (First call will JIT-compile...)")
    start = time.time()
    result_solver = solver.run(perturbed_params, dt_sec=dt_sec, errors=errors)
    elapsed = time.time() - start
    
    # Extract results
    final_params = result_solver.params
    final_state = result_solver.state
    
    # Compute final RMS
    final_rms = compute_rms(final_params, dt_sec, errors)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nConvergence:")
    print(f"  Converged: {final_state.error < 1e-12}")
    print(f"  Iterations: {final_state.iter_num}")
    print(f"  Final error: {final_state.error:.3e}")
    print(f"  Total time: {elapsed:.3f}s")
    
    print(f"\nParameter Recovery:")
    print(f"  Correct F0:  {correct_f0:.15f} Hz")
    print(f"  Starting F0: {wrong_f0:.15f} Hz")
    print(f"  Final F0:    {final_params[0]:.15f} Hz")
    print(f"  Error: {abs(final_params[0] - correct_f0):.3e} Hz")
    
    print(f"\n  Correct F1:  {correct_f1:.15e} Hz/s")
    print(f"  Final F1:    {final_params[1]:.15e} Hz/s")
    print(f"  Error: {abs(final_params[1] - correct_f1):.3e} Hz/s")
    
    print(f"\nRMS Evolution:")
    print(f"  Optimal RMS:    {result['rms_us']:.6f} μs")
    print(f"  Initial RMS:    {initial_rms:.6f} μs (perturbed)")
    print(f"  Final RMS:      {final_rms:.6f} μs")
    print(f"  Improvement:    {initial_rms - final_rms:.6f} μs")
    
    # Check success
    f0_recovered = abs(final_params[0] - correct_f0) / correct_f0 < 1e-12
    rms_recovered = abs(final_rms - result['rms_us']) < 0.01  # Within 0.01 μs
    
    print("\n" + "="*70)
    print("SUCCESS CRITERIA")
    print("="*70)
    
    if f0_recovered:
        print("✅ F0 recovered to within machine precision")
    else:
        print(f"❌ F0 recovery failed (error: {abs(final_params[0] - correct_f0):.3e})")
    
    if rms_recovered:
        print("✅ RMS recovered to optimal value")
    else:
        print(f"❌ RMS recovery failed (difference: {abs(final_rms - result['rms_us']):.3f} μs)")
    
    if f0_recovered and rms_recovered:
        print("\n🎉 TEST PASSED! JAX Gauss-Newton works correctly!")
    else:
        print("\n⚠️  TEST FAILED - Needs investigation")
    
    print("="*70)
    
    return final_params, final_rms, f0_recovered and rms_recovered


def test_larger_perturbation():
    """Test with even larger perturbation (1% error)."""
    print("\n\n" + "="*70)
    print("LARGE PERTURBATION TEST")
    print("="*70)
    print("\nObjective: Test GN robustness with 1% F0 error")
    
    # Get data
    result = compute_residuals_simple(
        "data/pulsars/J1909-3744_tdb.par",
        "data/pulsars/J1909-3744.tim",
        clock_dir="data/clock",
        verbose=False
    )
    
    dt_sec = jnp.array(result['dt_sec'])
    errors = jnp.array(result['errors_us']) * 1e-6
    
    params = parse_par_file("data/pulsars/J1909-3744_tdb.par")
    correct_f0 = params['F0']
    correct_f1 = params['F1']
    
    # 1% error in F0
    wrong_f0 = correct_f0 * 1.01
    perturbed_params = jnp.array([wrong_f0, correct_f1])
    
    initial_rms = compute_rms(perturbed_params, dt_sec, errors)
    print(f"   Initial RMS (1% F0 error): {initial_rms:.3f} μs")
    
    # Run solver
    solver = jaxopt.GaussNewton(
        residual_fun=residual_fun_f0f1,
        maxiter=20,
        tol=1e-12,
        verbose=False
    )
    
    result_solver = solver.run(perturbed_params, dt_sec=dt_sec, errors=errors)
    final_params = result_solver.params
    final_rms = compute_rms(final_params, dt_sec, errors)
    
    print(f"   Final RMS:                 {final_rms:.6f} μs")
    print(f"   Iterations:                {result_solver.state.iter_num}")
    
    recovered = abs(final_params[0] - correct_f0) / correct_f0 < 1e-10
    
    if recovered:
        print("   ✅ Recovered from 1% error!")
    else:
        print("   ⚠️  Failed to fully recover")
    
    return recovered


if __name__ == "__main__":
    # Test 1: Small perturbation (0.1%)
    final_params, final_rms, success1 = test_perturbed_parameters()
    
    # Test 2: Large perturbation (1%)
    success2 = test_larger_perturbation()
    
    # Summary
    print("\n\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    if success1 and success2:
        print("""
✅ BOTH TESTS PASSED!

Conclusions:
1. JAX autodiff correctly computes timing model derivatives
2. jaxopt.GaussNewton successfully minimizes residuals
3. Recovery works for both small (0.1%) and large (1%) errors
4. Convergence is fast (~1-10 iterations)

Next steps:
→ The GaussNewton approach is VALIDATED
→ Ready to extend to DM parameter fitting
→ Then benchmark speed vs current fitter
→ Then integrate into production

To run:
    python playground/jax_gn_perturbed_test.py
""")
    else:
        print("""
⚠️  SOME TESTS FAILED

Results:
- Small perturbation (0.1%): {}
- Large perturbation (1.0%): {}

This needs investigation before proceeding.
""".format("✅ PASS" if success1 else "❌ FAIL", 
           "✅ PASS" if success2 else "❌ FAIL"))
    
    print("="*70)
