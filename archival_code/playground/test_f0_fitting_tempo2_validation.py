#!/usr/bin/env python3
"""
Test spin derivatives with controlled tempo2 validation
=========================================================

This uses your pre-made files:
- Start: J1909-3744_tdb_wrong.par (F0 is slightly wrong)
- Target: J1909-3744_tdb_refit_F0.par (tempo2 fitted F0)

We fit ONLY F0 (not F1) to match tempo2's setup.
"""

import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd

def load_data():
    """Load J1909-3744 data with your controlled files."""
    data_dir = Path("data/pulsars")
    par_wrong = data_dir / "J1909-3744_tdb_wrong.par"
    par_correct = data_dir / "J1909-3744_tdb_refit_F0.par"
    
    # Use the tim file from the other location
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    if not par_wrong.exists() or not par_correct.exists() or not tim_file.exists():
        print(f"ERROR: Files not found")
        return None, None, None, None
    
    params_wrong = parse_par_file(par_wrong)
    params_correct = parse_par_file(par_correct)
    
    return params_wrong, params_correct, par_wrong, tim_file


def compute_residuals_for_f0(par_file, tim_file, f0_value):
    """Compute residuals with updated F0."""
    # Read par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    # Update F0
    updated_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'F0':
            updated_lines.append(f"F0             {f0_value:.20f}     1\n")
        else:
            updated_lines.append(line)
    
    # Write temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        temp_par = f.name
        f.writelines(updated_lines)
    
    try:
        # Compute residuals (suppress output)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            result = compute_residuals_simple(
                temp_par, tim_file, clock_dir="data/clock",
                subtract_tzr=True  # Use PINT's default (TZR + mean subtraction)
            )
        residuals_sec = result['residuals_us'] * 1e-6
        rms_us = result['rms_us']
    finally:
        Path(temp_par).unlink()
    
    return residuals_sec, rms_us


def test_f0_only_fitting():
    """Test fitting F0 only to match tempo2."""
    print("="*80)
    print("Testing F0-Only Fitting (Tempo2 Validation)")
    print("="*80)
    
    # Load data
    print("\n1. Loading J1909-3744 with controlled files...")
    params_wrong, params_correct, par_wrong, tim_file = load_data()
    if params_wrong is None:
        return False
    
    # Get TOA times
    from jug.io.tim_reader import parse_tim_file_mjds
    toas_data = parse_tim_file_mjds(tim_file)
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    
    n_toas = len(toas_mjd)
    print(f"   Loaded {n_toas} TOAs")
    
    # Starting and target F0
    f0_wrong = params_wrong['F0']
    f0_correct = params_correct['F0']
    
    print(f"\n2. F0 values:")
    print(f"   Starting (wrong):  {f0_wrong:.20f} Hz")
    print(f"   Target (tempo2):   {f0_correct:.20f} Hz")
    print(f"   Difference:        {f0_correct - f0_wrong:.3e} Hz")
    
    # Compute RMS with wrong and correct F0
    print(f"\n3. Computing RMS with each F0...")
    res_wrong, rms_wrong = compute_residuals_for_f0(par_wrong, tim_file, f0_wrong)
    res_correct, rms_correct = compute_residuals_for_f0(par_wrong, tim_file, f0_correct)
    
    print(f"   RMS with wrong F0:   {rms_wrong:.3f} μs")
    print(f"   RMS with correct F0: {rms_correct:.3f} μs")
    print(f"   Improvement:         {rms_wrong/rms_correct:.2f}x")
    
    print(f"\n4. Fitting F0 using analytical derivatives...")
    print(f"\n   DEBUG: Checking design matrix and residuals...")
    
    # Compute analytical derivative for F0 at wrong value
    params_wrong_copy = params_wrong.copy()
    params_wrong_copy['F0'] = f0_wrong
    
    derivs_test = compute_spin_derivatives(
        params_wrong_copy,
        toas_mjd,
        ['F0']
    )
    
    M_test = derivs_test['F0']
    
    print(f"   Design matrix (d(phase)/d(F0)):")
    print(f"     Mean: {np.mean(M_test):.3e} s/Hz")
    print(f"     RMS:  {np.sqrt(np.mean(M_test**2)):.3e} s/Hz")
    print(f"     Min:  {np.min(M_test):.3e} s/Hz")
    print(f"     Max:  {np.max(M_test):.3e} s/Hz")
    
    print(f"\n   Residuals with wrong F0:")
    print(f"     Mean: {np.mean(res_wrong):.6e} s")
    print(f"     RMS:  {np.sqrt(np.mean(res_wrong**2)):.6e} s")
    
    # Check correlation
    correlation = np.dot(M_test, res_wrong) / (np.linalg.norm(M_test) * np.linalg.norm(res_wrong))
    print(f"\n   Correlation between M and residuals: {correlation:.6f}")
    print(f"   (Should be non-zero if residuals encode F0 error)")
    
    # Predicted step
    M_col = M_test.reshape(-1, 1)
    predicted_step = -np.linalg.lstsq(M_col, res_wrong, rcond=None)[0][0]
    print(f"\n   Predicted ΔF0 from least squares: {predicted_step:.3e} Hz")
    print(f"   Actual F0 error: {f0_correct - f0_wrong:.3e} Hz")
    print(f"   Ratio: {predicted_step / (f0_correct - f0_wrong):.3f}")
    
    print(f"\n5. Running iterative fitting...")
    
    f0_curr = f0_wrong
    errors_sec = errors_us * 1e-6
    
    max_iter = 20
    min_iterations_for_stagnation = 3  # Need this many identical iterations
    
    print(f"   Iteration progress:")
    print(f"   (Showing ΔF0, RMS change, and expected ΔF0 from error)")
    print(f"   Expected ΔF0 to reach target: {f0_correct - f0_curr:.3e} Hz\n")
    
    # Track history for stagnation detection
    f0_history = []
    converged = False
    
    for iteration in range(max_iter):
        # Compute residuals
        residuals, rms_old = compute_residuals_for_f0(par_wrong, tim_file, f0_curr)
        
        # Compute analytical derivative for F0
        params_current = params_wrong.copy()
        params_current['F0'] = f0_curr
        
        derivs = compute_spin_derivatives(
            params_current,
            toas_mjd,
            ['F0']
        )
        
        # Design matrix is just one column (F0 derivative)
        M = derivs['F0'].reshape(-1, 1)
        
        # WLS iteration
        delta_params, cov, M_scaled = wls_solve_svd(
            residuals, errors_sec, M
        )
        
        # Update F0
        f0_new = f0_curr + delta_params[0]
        
        # Compute new RMS
        residuals_new, rms_new = compute_residuals_for_f0(par_wrong, tim_file, f0_new)
        
        # Uncertainty
        uncertainty = np.sqrt(cov[0, 0])
        
        # Print progress
        print(f"   {iteration + 1:2d}: F0={f0_curr:.20f}  RMS={rms_old:.6f} μs  ΔF0={delta_params[0]:.3e}")
        
        # Add to history
        f0_history.append(f0_new)
        
        # Check for stagnation: has F0 stopped changing?
        if len(f0_history) >= min_iterations_for_stagnation:
            # Check if last N iterations produced identical F0 (to floating point precision)
            recent_f0s = f0_history[-min_iterations_for_stagnation:]
            if all(f0 == recent_f0s[0] for f0 in recent_f0s):
                print(f"\n   ✓ Converged! F0 unchanged for {min_iterations_for_stagnation} iterations")
                converged = True
                f0_curr = f0_new
                rms_final = rms_new
                break
        
        # Accept step
        f0_curr = f0_new
    else:
        print(f"\n   ⚠️  Did not reach stagnation in {max_iter} iterations")
        rms_final = rms_new
    
    # Final results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n   RMS Progression:")
    print(f"     Starting (wrong F0):  {rms_wrong:.6f} μs")
    print(f"     Target (correct F0):  {rms_correct:.6f} μs")
    print(f"     Fitted:               {rms_final:.6f} μs")
    
    print(f"\n   F0 Values:")
    print(f"     Starting (wrong):  {f0_wrong:.20f} Hz")
    print(f"     Target (tempo2):   {f0_correct:.20f} Hz")
    print(f"     Fitted (JUG):      {f0_curr:.20f} Hz")
    
    print(f"\n   Differences:")
    diff_from_target = f0_curr - f0_correct
    diff_from_start = f0_curr - f0_wrong
    
    print(f"     Fitted - Target:   {diff_from_target:.3e} Hz")
    print(f"     Fitted - Starting: {diff_from_start:.3e} Hz")
    print(f"     Uncertainty:       {uncertainty:.3e} Hz")
    
    if abs(diff_from_target) > 0:
        sigma_diff = abs(diff_from_target / uncertainty)
        print(f"     Difference in σ:   {sigma_diff:.1f}σ")
    
    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    rms_improved = rms_final < rms_wrong * 0.95
    rms_close_to_target = abs(rms_final - rms_correct) < 0.1  # Within 0.1 μs
    
    # F0 should match tempo2 to within a few nHz
    f0_matches = abs(diff_from_target) < 1e-12  # 1 pHz
    
    print(f"\n   {'✅' if converged else '❌'} Converged: {converged}")
    print(f"   {'✅' if rms_improved else '❌'} RMS improved: {rms_wrong/rms_final:.2f}x")
    print(f"   {'✅' if rms_close_to_target else '⚠️ '} RMS close to target: Δ={abs(rms_final-rms_correct):.3f} μs")
    print(f"   {'✅' if f0_matches else '⚠️ '} F0 matches tempo2: Δ={abs(diff_from_target):.3e} Hz")
    
    overall_success = converged and rms_improved and rms_close_to_target
    
    print(f"\n   Overall: {'✅ TEST PASSED' if overall_success else '⚠️  NEEDS INVESTIGATION'}")
    
    if overall_success:
        print(f"\n   ✅ ANALYTICAL DERIVATIVES WORK!")
        print(f"   Fitting recovered tempo2's F0 value successfully.")
        print(f"   Ready to proceed with DM, binary, astrometry derivatives.")
    elif converged and rms_improved:
        print(f"\n   ⚠️  PARTIAL SUCCESS")
        print(f"   Fitter improved RMS but didn't perfectly match tempo2.")
        print(f"   This could be due to:")
        print(f"   - Different fitting setups (JUG vs tempo2)")
        print(f"   - Small residual calculation differences")
        print(f"   - Need to fit more parameters simultaneously")
    else:
        print(f"\n   ❌ TEST FAILED")
        print(f"   Analytical derivatives not working as expected.")
        print(f"   Need to investigate root cause.")
    
    return overall_success


if __name__ == '__main__':
    try:
        success = test_f0_only_fitting()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
