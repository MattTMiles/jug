"""
JAX Gauss-Newton Prototype
===========================

Testing jaxopt.GaussNewton with pure JAX residual function for pulsar timing.

Goal: Replace manual derivatives + numpy WLS with JAX autodiff + GN solver.

Status: PROTOTYPE - testing before integration into jug/fitting/

Author: JUG Development Team
Date: 2025-12-05
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxopt
import math
from pathlib import Path
import time

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple


# ============================================================================
# Step 1: Pure JAX Residual Function
# ============================================================================

def residual_function_jax(params_array, dt_sec_base, tdb_mjd, freq_mhz, errors,
                          pepoch, dmepoch, K_DM, fit_f0, fit_f1, fit_f2, 
                          fit_dm, fit_dm1, fit_dm2):
    """
    Compute timing residuals in pure JAX.
    
    This is the key function that jaxopt will differentiate.
    
    All inputs are arrays or scalars - no Python lists or strings.
    Boolean flags indicate which parameters are being fitted.
    
    Parameters
    ----------
    params_array : jnp.ndarray
        Parameter values in order: [F0?, F1?, F2?, DM?, DM1?, DM2?]
        (only fitted params included)
    dt_sec_base : jnp.ndarray
        Time from PEPOCH to emission (without DM correction)
    tdb_mjd : jnp.ndarray
        TDB times in MJD
    freq_mhz : jnp.ndarray
        Observing frequencies in MHz
    errors : jnp.ndarray
        TOA uncertainties in seconds
    pepoch : float
        PEPOCH in MJD
    dmepoch : float
        DMEPOCH in MJD
    K_DM : float
        DM constant
    fit_* : bool
        Flags indicating which parameters are being fitted
        
    Returns
    -------
    residuals : jnp.ndarray
        Phase residuals in seconds, shape (N_TOA,)
    """
    # Extract parameters from array based on flags
    idx = 0
    F0 = params_array[idx] if fit_f0 else jnp.float64(0.0)
    idx += int(fit_f0)
    F1 = params_array[idx] if fit_f1 else jnp.float64(0.0)
    idx += int(fit_f1)
    F2 = params_array[idx] if fit_f2 else jnp.float64(0.0)
    idx += int(fit_f2)
    DM = params_array[idx] if fit_dm else jnp.float64(0.0)
    idx += int(fit_dm)
    DM1 = params_array[idx] if fit_dm1 else jnp.float64(0.0)
    idx += int(fit_dm1)
    DM2 = params_array[idx] if fit_dm2 else jnp.float64(0.0)
    
    # Compute DM delay correction if DM parameters are being fitted
    dm_delay_correction = jnp.zeros_like(dt_sec_base)
    if fit_dm or fit_dm1 or fit_dm2:
        dt_years = (tdb_mjd - dmepoch) / 365.25
        dm_eff = DM
        if fit_dm1:
            dm_eff += DM1 * dt_years
        if fit_dm2:
            dm_eff += DM2 * (dt_years ** 2) / 2.0
        # DM delay in seconds
        dm_delay_correction = K_DM * dm_eff / (freq_mhz ** 2)
    
    # Update dt_sec
    dt_sec = dt_sec_base - dm_delay_correction
    
    # Compute spin phase
    phase = F0 * dt_sec
    if fit_f1:
        phase += F1 * (dt_sec ** 2) / 2.0
    if fit_f2:
        phase += F2 * (dt_sec ** 3) / 6.0
    
    # Wrap phase
    phase_wrapped = phase - jnp.round(phase)
    
    # Convert to time residuals
    residuals_sec = phase_wrapped / F0
    
    # Weighted mean subtraction
    weights = 1.0 / (errors ** 2)
    weighted_mean = jnp.sum(residuals_sec * weights) / jnp.sum(weights)
    residuals_sec = residuals_sec - weighted_mean
    
    return residuals_sec


# ============================================================================
# Step 2: Weighted Residual Function (for WLS)
# ============================================================================

def weighted_residual_function(params_array, static_data):
    """
    Weighted residuals for WLS fitting.
    
    This is what GaussNewton should minimize: weighted residuals.
    """
    residuals = residual_function_jax(params_array, static_data)
    errors = static_data['errors']
    weighted_residuals = residuals / errors
    return weighted_residuals


# ============================================================================
# Step 3: Setup Function (prepare static data)
# ============================================================================

def setup_static_data(par_file, tim_file, fit_params, clock_dir=None):
    """
    Compute all the expensive stuff once and cache it.
    
    Returns
    -------
    static_data : dict
        Everything needed for residual_function_jax
    initial_params : np.ndarray
        Starting parameter values
    """
    print("Setting up static data...")
    start = time.time()
    
    # Parse files
    params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    
    # Compute full residuals to get dt_sec and TDB times
    result = compute_residuals_simple(
        par_file, tim_file, 
        clock_dir=clock_dir
    )
    
    # Extract data
    dt_sec_base = np.array(result['dt_sec'])
    tdb_mjd = np.array(result['tdb_mjd'])
    freq_mhz = np.array([toa.freq_mhz for toa in toas])
    errors_sec = np.array([toa.error_us * 1e-6 for toa in toas])  # μs → seconds
    
    # Get parameter values
    initial_params = np.array([params[p] for p in fit_params])
    
    # Constants
    from jug.utils.constants import K_DM_SEC
    
    # Build static data dict
    static_data = {
        'dt_sec_base': dt_sec_base,
        'tdb_mjd': tdb_mjd,
        'freq_mhz': freq_mhz,
        'errors': errors_sec,
        'param_names': fit_params,
        'pepoch': params['PEPOCH'],
        'dmepoch': params.get('DMEPOCH', params['PEPOCH']),
        'K_DM': K_DM_SEC,
    }
    
    elapsed = time.time() - start
    print(f"  Static data setup: {elapsed:.3f}s")
    
    return static_data, initial_params


# ============================================================================
# Step 4: Fitting Function (using jaxopt)
# ============================================================================

def fit_with_jaxopt_gn(par_file, tim_file, fit_params, 
                       max_iter=20, tol=1e-12, clock_dir=None,
                       verbose=True):
    """
    Fit timing parameters using jaxopt.GaussNewton with JAX autodiff.
    
    Parameters
    ----------
    par_file : str or Path
        Path to .par file
    tim_file : str or Path
        Path to .tim file
    fit_params : list of str
        Parameters to fit (e.g., ['F0', 'F1', 'DM'])
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    clock_dir : str, optional
        Path to clock files
    verbose : bool
        Print progress
        
    Returns
    -------
    result : dict
        Fitting results
    """
    if verbose:
        print("="*70)
        print("JAX GAUSS-NEWTON FITTER (Prototype)")
        print("="*70)
        print(f"\nFitting {len(fit_params)} parameters: {fit_params}")
    
    # Setup
    static_data, initial_params = setup_static_data(
        par_file, tim_file, fit_params, clock_dir
    )
    
    if verbose:
        print(f"\nInitial parameter values:")
        for name, val in zip(fit_params, initial_params):
            print(f"  {name} = {val:.15e}")
    
    # Create solver
    solver = jaxopt.GaussNewton(
        residual_fun=weighted_residual_function,
        maxiter=max_iter,
        tol=tol,
        implicit_diff=False,  # We don't need implicit differentiation
    )
    
    # Run solver (first call will JIT-compile)
    if verbose:
        print("\nRunning Gauss-Newton solver...")
        print("(First iteration will JIT-compile, please wait...)")
    
    jit_start = time.time()
    result = solver.run(initial_params, static_data=static_data)
    total_time = time.time() - jit_start
    
    # Extract results
    final_params = result.params
    final_state = result.state
    
    # Compute final residuals and RMS
    final_residuals = residual_function_jax(final_params, static_data)
    errors = static_data['errors']
    weights = 1.0 / (errors ** 2)
    final_rms = float(jnp.sqrt(jnp.sum(final_residuals**2 * weights) / jnp.sum(weights)))
    
    # Compute initial RMS for comparison
    initial_residuals = residual_function_jax(initial_params, static_data)
    initial_rms = float(jnp.sqrt(jnp.sum(initial_residuals**2 * weights) / jnp.sum(weights)))
    
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"\nConverged: {final_state.error < tol}")
        print(f"Iterations: {final_state.iter_num}")
        print(f"Final error: {final_state.error:.3e}")
        print(f"\nInitial RMS: {initial_rms*1e6:.6f} μs")
        print(f"Final RMS:   {final_rms*1e6:.6f} μs")
        print(f"Improvement: {(initial_rms - final_rms)*1e6:.6f} μs")
        print(f"\nTotal time: {total_time:.3f}s")
        print(f"\nFitted parameters:")
        for name, val_init, val_final in zip(fit_params, initial_params, final_params):
            delta = val_final - val_init
            print(f"  {name}: {val_final:.15e} (Δ = {delta:+.3e})")
        print("="*70)
    
    return {
        'final_params': {name: float(val) for name, val in zip(fit_params, final_params)},
        'initial_params': {name: float(val) for name, val in zip(fit_params, initial_params)},
        'final_rms_us': final_rms * 1e6,
        'initial_rms_us': initial_rms * 1e6,
        'converged': final_state.error < tol,
        'iterations': int(final_state.iter_num),
        'final_error': float(final_state.error),
        'total_time': total_time,
        'residuals_sec': np.array(final_residuals),
    }


# ============================================================================
# Step 5: Test Cases
# ============================================================================

def test_f0_f1_fitting():
    """Test F0+F1 fitting (should converge quickly)."""
    print("\n" + "="*70)
    print("TEST 1: F0+F1 Fitting")
    print("="*70)
    
    result = fit_with_jaxopt_gn(
        par_file="data/pulsars/J1909-3744_tdb.par",
        tim_file="data/pulsars/J1909-3744.tim",
        fit_params=['F0', 'F1'],
        max_iter=10,
        clock_dir="data/clock",
        verbose=True
    )
    
    print(f"\nExpected final RMS: ~0.404 μs")
    print(f"Actual final RMS:   {result['final_rms_us']:.6f} μs")
    
    return result


def test_f0_f1_dm_dm1_fitting():
    """Test F0+F1+DM+DM1 fitting (the problematic case)."""
    print("\n" + "="*70)
    print("TEST 2: F0+F1+DM+DM1 Fitting")
    print("="*70)
    
    result = fit_with_jaxopt_gn(
        par_file="data/pulsars/J1909-3744_tdb.par",
        tim_file="data/pulsars/J1909-3744.tim",
        fit_params=['F0', 'F1', 'DM', 'DM1'],
        max_iter=20,
        clock_dir="data/clock",
        verbose=True
    )
    
    print(f"\nExpected final RMS: ~0.404 μs")
    print(f"Actual final RMS:   {result['final_rms_us']:.6f} μs")
    
    return result


def test_dm_only_fitting():
    """Test DM-only fitting."""
    print("\n" + "="*70)
    print("TEST 3: DM-only Fitting")
    print("="*70)
    
    result = fit_with_jaxopt_gn(
        par_file="data/pulsars/J1909-3744_tdb.par",
        tim_file="data/pulsars/J1909-3744.tim",
        fit_params=['DM'],
        max_iter=10,
        clock_dir="data/clock",
        verbose=True
    )
    
    return result


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check jaxopt is installed
    try:
        import jaxopt
        print(f"jaxopt imported successfully")
    except ImportError:
        print("ERROR: jaxopt not installed. Install with: pip install jaxopt")
        sys.exit(1)
    
    # Run tests
    print("\n" + "="*70)
    print("JAX GAUSS-NEWTON PROTOTYPE - TEST SUITE")
    print("="*70)
    print("\nThis will test jaxopt.GaussNewton on three cases:")
    print("1. F0+F1 (simple, should converge in ~5 iterations)")
    print("2. F0+F1+DM+DM1 (complex, tests DM delay computation)")
    print("3. DM-only (tests DM-only path)")
    
    # Test 1: F0+F1
    result1 = test_f0_f1_fitting()
    
    # Test 2: F0+F1+DM+DM1
    result2 = test_f0_f1_dm_dm1_fitting()
    
    # Test 3: DM-only
    result3 = test_dm_only_fitting()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTest 1 (F0+F1):        {result1['final_rms_us']:.6f} μs in {result1['iterations']} iterations")
    print(f"Test 2 (F0+F1+DM+DM1): {result2['final_rms_us']:.6f} μs in {result2['iterations']} iterations")
    print(f"Test 3 (DM-only):      {result3['final_rms_us']:.6f} μs in {result3['iterations']} iterations")
    
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("""
If all tests pass with expected RMS values:
1. Add Levenberg-Marquardt fallback
2. Add parameter uncertainties (from Jacobian/covariance)
3. Compare speed vs current optimized_fitter.py
4. Integrate into jug/fitting/ as production code

To test:
    python playground/jax_gauss_newton_prototype.py
""")
