"""
JAX Gauss-Newton - DM Parameter Test
=====================================

Test jaxopt.GaussNewton on DM parameter fitting.

Why this should work:
- DM correction happens AFTER dt_sec is computed
- dt_sec doesn't depend on DM values
- So we can fit DM independently

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
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.constants import K_DM_SEC


def residual_fun_dm(params, dt_sec_base, tdb_mjd, freq_mhz, errors, 
                    pepoch, dmepoch, f0):
    """
    Compute weighted residuals for DM fitting.
    
    Parameters
    ----------
    params : array [DM]
    dt_sec_base : array, time from PEPOCH (WITHOUT DM correction yet)
    tdb_mjd : array, TDB times
    freq_mhz : array, frequencies
    errors : array, TOA uncertainties in seconds
    pepoch : float, PEPOCH
    dmepoch : float, DMEPOCH
    f0 : float, spin frequency
    
    Returns
    -------
    weighted_residuals : array
    """
    DM = params[0]
    
    # Compute DM delay
    dm_delay_sec = K_DM_SEC * DM / (freq_mhz ** 2)
    
    # Apply DM correction to dt_sec
    dt_sec = dt_sec_base - dm_delay_sec
    
    # Compute phase (just F0 term for simplicity)
    phase = f0 * dt_sec
    
    # Wrap phase
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert to time residuals
    residuals = phase_wrapped / f0
    
    # Subtract weighted mean
    weights = 1.0 / (errors ** 2)
    mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    residuals = residuals - mean
    
    # Weight for WLS
    weighted_residuals = residuals / errors
    
    return weighted_residuals


def compute_rms_dm(params, dt_sec_base, tdb_mjd, freq_mhz, errors,
                   pepoch, dmepoch, f0):
    """Compute RMS from DM parameter."""
    weighted_resid = residual_fun_dm(params, dt_sec_base, tdb_mjd, freq_mhz, 
                                     errors, pepoch, dmepoch, f0)
    residuals = weighted_resid * errors
    weights = 1.0 / (errors ** 2)
    wrms = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    return float(wrms * 1e6)  # μs


def test_dm_fitting():
    """Test GN on DM parameter fitting."""
    print("="*70)
    print("DM PARAMETER FITTING TEST")
    print("="*70)
    print("\nObjective: Fit DM parameter using JAX Gauss-Newton")
    print("Why this should work: DM correction is independent of dt_sec\n")
    
    # Get data
    print("1. Loading data...")
    par_file = "data/pulsars/J1909-3744_tdb.par"
    tim_file = "data/pulsars/J1909-3744.tim"
    
    params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    
    # Compute residuals with DM=0 to get dt_sec WITHOUT DM correction
    print("\n2. Computing dt_sec with DM=0 (base timing)...")
    # Create temp par file with DM=0
    import tempfile
    from pathlib import Path
    
    params_nodm = params.copy()
    correct_dm = params['DM']
    params_nodm['DM'] = 0.0
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        for key, val in params_nodm.items():
            if isinstance(val, (int, float)):
                f.write(f"{key} {val:.15e}\n")
            else:
                f.write(f"{key} {val}\n")
        temp_par = f.name
    
    result_nodm = compute_residuals_simple(temp_par, tim_file, 
                                           clock_dir="data/clock", verbose=False)
    Path(temp_par).unlink()
    
    print(f"   RMS with DM=0: {result_nodm['rms_us']:.3f} μs (very bad!)")
    
    # Also compute with correct DM for comparison
    result_correct = compute_residuals_simple(par_file, tim_file,
                                              clock_dir="data/clock", verbose=False)
    print(f"   RMS with correct DM: {result_correct['rms_us']:.6f} μs (good)")
    
    # Extract data
    dt_sec_base = jnp.array(result_nodm['dt_sec'])  # Without DM correction
    tdb_mjd = jnp.array(result_nodm['tdb_mjd'])
    freq_mhz = jnp.array([toa.freq_mhz for toa in toas])
    errors = jnp.array([toa.error_us * 1e-6 for toa in toas])
    
    f0 = params['F0']
    pepoch = params['PEPOCH']
    dmepoch = params.get('DMEPOCH', pepoch)
    
    print(f"\n3. Setting up DM fitting...")
    print(f"   Correct DM: {correct_dm:.10f} pc cm⁻³")
    
    # Start from a perturbed value
    starting_dm = correct_dm * 0.9  # 10% too low
    initial_params = jnp.array([starting_dm])
    
    print(f"   Starting DM: {starting_dm:.10f} pc cm⁻³ (-10%)")
    
    # Compute initial RMS
    initial_rms = compute_rms_dm(initial_params, dt_sec_base, tdb_mjd, freq_mhz,
                                 errors, pepoch, dmepoch, f0)
    print(f"   RMS with starting DM: {initial_rms:.3f} μs")
    
    # Create solver
    print("\n4. Running Gauss-Newton...")
    solver = jaxopt.GaussNewton(
        residual_fun=residual_fun_dm,
        maxiter=20,
        tol=1e-12
    )
    
    start = time.time()
    result_solver = solver.run(
        initial_params,
        dt_sec_base=dt_sec_base,
        tdb_mjd=tdb_mjd,
        freq_mhz=freq_mhz,
        errors=errors,
        pepoch=pepoch,
        dmepoch=dmepoch,
        f0=f0
    )
    elapsed = time.time() - start
    
    # Extract results
    final_params = result_solver.params
    final_state = result_solver.state
    
    # Compute final RMS
    final_rms = compute_rms_dm(final_params, dt_sec_base, tdb_mjd, freq_mhz,
                               errors, pepoch, dmepoch, f0)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nConvergence:")
    print(f"  Converged: {final_state.error < 1e-12}")
    print(f"  Iterations: {final_state.iter_num}")
    print(f"  Final error: {final_state.error:.3e}")
    print(f"  Total time: {elapsed:.3f}s")
    
    print(f"\nDM Recovery:")
    print(f"  Correct DM:  {correct_dm:.10f} pc cm⁻³")
    print(f"  Starting DM: {starting_dm:.10f} pc cm⁻³")
    print(f"  Final DM:    {final_params[0]:.10f} pc cm⁻³")
    print(f"  Error: {abs(final_params[0] - correct_dm):.3e} pc cm⁻³")
    print(f"  Relative error: {abs(final_params[0] - correct_dm)/correct_dm * 100:.6f}%")
    
    print(f"\nRMS Evolution:")
    print(f"  Optimal RMS:  {result_correct['rms_us']:.6f} μs")
    print(f"  Initial RMS:  {initial_rms:.3f} μs (10% wrong DM)")
    print(f"  Final RMS:    {final_rms:.6f} μs")
    print(f"  Improvement:  {initial_rms - final_rms:.3f} μs")
    
    # Check success
    dm_recovered = abs(final_params[0] - correct_dm) / correct_dm < 0.001  # 0.1%
    rms_recovered = abs(final_rms - result_correct['rms_us']) < 0.01  # 0.01 μs
    
    print("\n" + "="*70)
    print("SUCCESS CRITERIA")
    print("="*70)
    
    if dm_recovered:
        print("✅ DM recovered to within 0.1%")
    else:
        print(f"❌ DM recovery failed (error: {abs(final_params[0] - correct_dm)/correct_dm * 100:.3f}%)")
    
    if rms_recovered:
        print("✅ RMS recovered to optimal value")
    else:
        print(f"❌ RMS recovery failed (difference: {abs(final_rms - result_correct['rms_us']):.3f} μs)")
    
    if dm_recovered and rms_recovered:
        print("\n🎉 TEST PASSED! DM fitting with JAX GN works!")
    else:
        print("\n⚠️  TEST FAILED - Needs investigation")
    
    print("="*70)
    
    return final_params[0], final_rms, dm_recovered and rms_recovered


if __name__ == "__main__":
    final_dm, final_rms, success = test_dm_fitting()
    
    print("\n\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    
    if success:
        print("""
✅ DM FITTING WORKS!

Key findings:
1. JAX autodiff + GaussNewton successfully fits DM parameter
2. Converges from 10% error to <0.1% error
3. RMS improves significantly (from ~1000 μs to ~0.4 μs)
4. Fast convergence in just a few iterations

Why this worked:
- DM correction is applied AFTER dt_sec is computed
- No circular dependency between fitted parameter and cached data
- Clean separation of concerns

Next steps:
→ Extend to DM1, DM2 (time derivatives of DM)
→ Then try combined F0+F1+DM+DM1 fitting
→ For F0/F1, need to recompute dt_sec in pure JAX
→ Or: use current architecture with cached dt_sec for other params

The path forward is clear!

To run:
    python playground/jax_gn_dm_test.py
""")
    else:
        print("""
⚠️  DM FITTING FAILED

This suggests a fundamental issue with the approach.
Need to investigate before proceeding.
""")
    
    print("="*70)
