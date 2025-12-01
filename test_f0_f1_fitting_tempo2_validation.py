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


def compute_residuals_for_f0(par_file, tim_file, f0_value, f1_value=None):
    """Compute residuals with updated F0."""
    # Read par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    # Update F0 and optionally F1
    updated_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'F0':
            updated_lines.append(f"F0             {f0_value:.20f}     1\n")
        elif len(parts) > 0 and parts[0] == 'F1' and f1_value is not None:
            updated_lines.append(f"F1             {f1_value:.20f}     1\n")
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


def test_f0_f1_fitting():
    """Test fitting F0 and F1 simultaneously to match tempo2."""
    print("="*80)
    print("Testing F0 and F1 Fitting (Tempo2 Validation)")
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
    
    # Starting and target F0 and F1
    f0_wrong = params_wrong['F0']
    f0_correct = params_correct['F0']
    f1_wrong = params_wrong['F1']
    f1_correct = params_correct['F1']
    
    print(f"\n2. F0 and F1 values:")
    print(f"   Starting (wrong):  F0={f0_wrong:.20f} Hz")
    print(f"                       F1={f1_wrong:.20e} Hz/s")
    print(f"   Target (tempo2):   F0={f0_correct:.20f} Hz")
    print(f"                       F1={f1_correct:.20e} Hz/s")
    print(f"   Difference:        ΔF0={f0_correct - f0_wrong:.3e} Hz")
    print(f"                       ΔF1={f1_correct - f1_wrong:.3e} Hz/s")
    
    # Compute RMS with wrong and correct F0
    print(f"\n3. Computing RMS with each F0...")
    res_wrong, rms_wrong = compute_residuals_for_f0(par_wrong, tim_file, f0_wrong, f1_wrong)
    res_correct, rms_correct = compute_residuals_for_f0(par_wrong, tim_file, f0_correct, f1_correct)
    
    print(f"   RMS with wrong F0:   {rms_wrong:.3f} μs")
    print(f"   RMS with correct F0: {rms_correct:.3f} μs")
    print(f"   Improvement:         {rms_wrong/rms_correct:.2f}x")
    
    print(f"\n4. Checking design matrix and residuals...")
    
    # Compute analytical derivatives at starting values
    params_wrong_copy = params_wrong.copy()
    params_wrong_copy['F0'] = f0_wrong
    params_wrong_copy['F1'] = f1_wrong
    
    derivs_test = compute_spin_derivatives(
        params_wrong_copy,
        toas_mjd,
        ['F0', 'F1']
    )
    
    M_f0 = derivs_test['F0']
    M_f1 = derivs_test['F1']
    
    print(f"   Design matrix (d(residual)/d(F0)):")
    print(f"     Mean: {np.mean(M_f0):.3e} s/Hz")
    print(f"     RMS:  {np.sqrt(np.mean(M_f0**2)):.3e} s/Hz")
    
    print(f"\n   Design matrix (d(residual)/d(F1)):")
    print(f"     Mean: {np.mean(M_f1):.3e} s/(Hz/s)")
    print(f"     RMS:  {np.sqrt(np.mean(M_f1**2)):.3e} s/(Hz/s)")
    
    print(f"\n   Residuals with wrong F0/F1:")
    print(f"     Mean: {np.mean(res_wrong):.6e} s")
    print(f"     RMS:  {np.sqrt(np.mean(res_wrong**2)):.6e} s")
    
    # Check correlation for both parameters
    corr_f0 = np.dot(M_f0, res_wrong) / (np.linalg.norm(M_f0) * np.linalg.norm(res_wrong))
    corr_f1 = np.dot(M_f1, res_wrong) / (np.linalg.norm(M_f1) * np.linalg.norm(res_wrong))
    
    print(f"\n   Correlation (F0, residuals): {corr_f0:.6f}")
    print(f"   Correlation (F1, residuals): {corr_f1:.6f}")
    
    # Predicted steps from multi-parameter fit
    M_both = np.column_stack([M_f0, M_f1])
    predicted_steps = -np.linalg.lstsq(M_both, res_wrong, rcond=None)[0]
    
    print(f"\n   Predicted steps (multi-parameter):")
    print(f"     ΔF0: {predicted_steps[0]:.3e} Hz  (actual error: {f0_correct - f0_wrong:.3e} Hz)")
    print(f"     ΔF1: {predicted_steps[1]:.3e} Hz/s  (actual error: {f1_correct - f1_wrong:.3e} Hz/s)")

    print(f"\n5. Running iterative multi-parameter fitting...")
    
    f0_curr = f0_wrong
    f1_curr = f1_wrong
    errors_sec = errors_us * 1e-6
    
    max_iter = 20
    convergence_threshold = 1e-14  # Relax threshold slightly for numerical stability
    
    print(f"\n   Target values:")
    print(f"     F0: {f0_correct:.20f} Hz")
    print(f"     F1: {f1_correct:.20e} Hz/s\n")
    
    converged = False
    prev_delta_max = None
    
    for iteration in range(max_iter):
        # Compute residuals
        residuals, rms_old = compute_residuals_for_f0(par_wrong, tim_file, f0_curr, f1_curr)
        
        # Compute analytical derivatives
        params_current = params_wrong.copy()
        params_current['F0'] = f0_curr
        params_current['F1'] = f1_curr
        
        derivs = compute_spin_derivatives(
            params_current,
            toas_mjd,
            ['F0', 'F1']
        )
        
        # Build design matrix: [M_F0, M_F1] (n_toas x 2)
        M = np.column_stack([derivs['F0'], derivs['F1']])
        
        # WLS solve for both parameters simultaneously
        delta_params, cov, M_scaled = wls_solve_svd(
            residuals, errors_sec, M
        )
        
        # Update both parameters
        f0_new = f0_curr + delta_params[0]
        f1_new = f1_curr + delta_params[1]
        
        # Compute new RMS
        residuals_new, rms_new = compute_residuals_for_f0(par_wrong, tim_file, f0_new, f1_new)
        
        # Uncertainties
        unc_f0 = np.sqrt(cov[0, 0])
        unc_f1 = np.sqrt(cov[1, 1])
        
        # Print progress
        print(f"   {iteration + 1:2d}: RMS={rms_old:.6f} μs  ΔF0={delta_params[0]:.3e}  ΔF1={delta_params[1]:.3e}")
        
        # Check convergence: both delta params below threshold OR stagnated
        max_delta = max(abs(delta_params[0]), abs(delta_params[1]))
        
        # Check for stagnation (delta stopped changing)
        if prev_delta_max is not None and abs(max_delta - prev_delta_max) < 1e-20:
            print(f"\n   ✓ Converged! Delta stagnated at {max_delta:.3e}")
            converged = True
            f0_curr = f0_new
            f1_curr = f1_new
            rms_final = rms_new
            break
        
        if max_delta < convergence_threshold:
            print(f"\n   ✓ Converged! Max delta = {max_delta:.3e}")
            converged = True
            f0_curr = f0_new
            f1_curr = f1_new
            rms_final = rms_new
            break
        
        # Accept step
        f0_curr = f0_new
        f1_curr = f1_new
        prev_delta_max = max_delta
    else:
        print(f"\n   ⚠️  Did not converge in {max_iter} iterations (max_delta={max_delta:.3e})")
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
    
    print(f"\n   F1 Values:")
    print(f"     Starting (wrong):  {f1_wrong:.20e} Hz/s")
    print(f"     Target (tempo2):   {f1_correct:.20e} Hz/s")
    print(f"     Fitted (JUG):      {f1_curr:.20e} Hz/s")
    
    print(f"\n   Differences:")
    diff_f0 = f0_curr - f0_correct
    diff_f1 = f1_curr - f1_correct
    
    print(f"     F0: Fitted - Target = {diff_f0:.3e} Hz")
    print(f"     F1: Fitted - Target = {diff_f1:.3e} Hz/s")
    print(f"     F0 Uncertainty:       {unc_f0:.3e} Hz")
    print(f"     F1 Uncertainty:       {unc_f1:.3e} Hz/s")
    
    if abs(diff_f0) > 0:
        sigma_f0 = abs(diff_f0 / unc_f0)
        print(f"     F0 Difference in σ:   {sigma_f0:.1f}σ")
    if abs(diff_f1) > 0:
        sigma_f1 = abs(diff_f1 / unc_f1)
        print(f"     F1 Difference in σ:   {sigma_f1:.1f}σ")
    
    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    rms_improved = rms_final < rms_wrong * 0.95
    rms_close_to_target = abs(rms_final - rms_correct) < 0.1  # Within 0.1 μs
    
    # Parameters should match tempo2 to high precision
    f0_matches = abs(diff_f0) < 1e-12  # 1 pHz
    f1_matches = abs(diff_f1) < 1e-20  # Very tight for F1
    
    print(f"\n   {'✅' if converged else '❌'} Converged: {converged}")
    print(f"   {'✅' if rms_improved else '❌'} RMS improved: {rms_wrong/rms_final:.2f}x")
    print(f"   {'✅' if rms_close_to_target else '⚠️ '} RMS close to target: Δ={abs(rms_final-rms_correct):.3f} μs")
    print(f"   {'✅' if f0_matches else '⚠️ '} F0 matches tempo2: Δ={abs(diff_f0):.3e} Hz")
    print(f"   {'✅' if f1_matches else '⚠️ '} F1 matches tempo2: Δ={abs(diff_f1):.3e} Hz/s")
    
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
        success = test_f0_f1_fitting()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
