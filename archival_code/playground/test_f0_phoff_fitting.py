#!/usr/bin/env python3
"""
Test F0 fitting with PHOFF (phase offset) parameter.

This implements PINT's approach: fit both F0 and PHOFF simultaneously.
PHOFF absorbs the arbitrary phase zero-point.
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


def compute_residuals_with_phoff(par_file, tim_file, f0_value, phoff_cycles):
    """Compute residuals with F0 and phase offset.
    
    residual = (phase - mean_phase) - PHOFF
    
    This matches PINT's PhaseOffset component.
    """
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
        # Compute residuals WITHOUT TZR (use mean subtraction instead)
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            result = compute_residuals_simple(
                temp_par, tim_file, clock_dir="data/clock",
                subtract_tzr=False  # Use mean instead of TZR
            )
        
        residuals_us = result['residuals_us']
        rms_us = result['rms_us']
        
        # Apply PHOFF (phase offset in cycles)
        # Convert residuals to cycles first
        from jug.io.par_reader import parse_par_file
        params = parse_par_file(temp_par)
        f0 = params['F0']
        
        residuals_cycles = residuals_us * 1e-6 * f0
        
        # Subtract PHOFF
        residuals_cycles_offset = residuals_cycles - phoff_cycles
        
        # Convert back to seconds
        residuals_sec = residuals_cycles_offset / f0
        
        # Recompute RMS
        rms_us_new = np.sqrt(np.mean(residuals_sec**2)) * 1e6
        
    finally:
        Path(temp_par).unlink()
    
    return residuals_sec, rms_us_new


def test_f0_phoff_fitting():
    """Test fitting F0 and PHOFF together."""
    print("="*80)
    print("Testing F0 + PHOFF Fitting (PINT-Style)")
    print("="*80)
    
    # Load data
    print("\n1. Loading J1909-3744...")
    data_dir = Path("data/pulsars")
    par_wrong = data_dir / "J1909-3744_tdb_wrong.par"
    par_correct = data_dir / "J1909-3744_tdb_refit_F0.par"
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    params_wrong = parse_par_file(par_wrong)
    params_correct = parse_par_file(par_correct)
    
    # Get TOA times
    from jug.io.tim_reader import parse_tim_file_mjds
    toas_data = parse_tim_file_mjds(tim_file)
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    
    f0_wrong = params_wrong['F0']
    f0_correct = params_correct['F0']
    
    print(f"   F0 wrong:   {f0_wrong:.20f} Hz")
    print(f"   F0 correct: {f0_correct:.20f} Hz")
    print(f"   Difference: {f0_correct - f0_wrong:.3e} Hz")
    
    # Starting values
    f0_curr = f0_wrong
    phoff_curr = 0.0  # Start with zero offset
    
    errors_sec = errors_us * 1e-6
    
    max_iter = 20
    convergence_threshold = 1e-12
    
    print(f"\n2. Fitting F0 and PHOFF simultaneously...")
    print(f"   Fitting 2 parameters: F0, PHOFF")
    print(f"   PHOFF is the phase offset parameter\n")
    
    for iteration in range(max_iter):
        # Compute residuals with current F0 and PHOFF
        residuals, rms_old = compute_residuals_with_phoff(
            par_wrong, tim_file, f0_curr, phoff_curr
        )
        
        # Compute derivatives
        params_current = params_wrong.copy()
        params_current['F0'] = f0_curr
        
        # d(phase)/d(F0)
        derivs_f0 = compute_spin_derivatives(
            params_current,
            toas_mjd,
            ['F0']
        )
        
        # d(phase)/d(PHOFF) = -1 (constant)
        deriv_phoff = -np.ones(len(toas_mjd))
        
        # Design matrix: [d/dF0, d/dPHOFF]
        M = np.column_stack([derivs_f0['F0'], deriv_phoff])
        
        # WLS solve for both parameters
        delta_params, cov, M_scaled = wls_solve_svd(
            residuals, errors_sec, M
        )
        
        # Update both
        f0_new = f0_curr + delta_params[0]
        phoff_new = phoff_curr + delta_params[1]
        
        # Compute new RMS
        residuals_new, rms_new = compute_residuals_with_phoff(
            par_wrong, tim_file, f0_new, phoff_new
        )
        
        # Uncertainties
        uncertainties = np.sqrt(np.diag(cov))
        
        print(f"   {iteration + 1:2d}: F0={f0_curr:.15e}  PHOFF={phoff_curr:.6e}  RMS={rms_old:.6f} μs")
        print(f"       ΔF0={delta_params[0]:.3e}  ΔPHOFF={delta_params[1]:.3e}")
        
        # Check convergence
        rel_change_f0 = abs(delta_params[0] / f0_curr)
        rel_change_phoff = abs(delta_params[1]) if phoff_curr == 0 else abs(delta_params[1] / phoff_curr)
        max_rel_change = max(rel_change_f0, rel_change_phoff)
        
        if max_rel_change < convergence_threshold:
            print(f"\n   ✓ Converged! (max relative change: {max_rel_change:.2e})")
            f0_curr = f0_new
            phoff_curr = phoff_new
            rms_final = rms_new
            break
        
        # Accept step
        f0_curr = f0_new
        phoff_curr = phoff_new
    else:
        print(f"\n   ⚠️  Did not converge in {max_iter} iterations")
        rms_final = rms_new
    
    # Results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    print(f"\n   Fitted Values:")
    print(f"     F0    = {f0_curr:.20f} ± {uncertainties[0]:.3e} Hz")
    print(f"     PHOFF = {phoff_curr:.6e} ± {uncertainties[1]:.3e} cycles")
    
    print(f"\n   Target Values:")
    print(f"     F0 = {f0_correct:.20f} Hz")
    
    diff_f0 = f0_curr - f0_correct
    sigma_f0 = abs(diff_f0 / uncertainties[0]) if uncertainties[0] > 0 else 0
    
    print(f"\n   F0 Difference:")
    print(f"     Fitted - Target: {diff_f0:.3e} Hz")
    print(f"     In sigma: {sigma_f0:.2f}σ")
    
    print(f"\n   RMS: {rms_final:.6f} μs")
    
    # Assessment
    print("\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    converged = max_rel_change < convergence_threshold if iteration < max_iter - 1 else False
    f0_matches = abs(diff_f0) < 1e-12
    
    print(f"\n   {'✅' if converged else '❌'} Converged: {converged}")
    print(f"   {'✅' if f0_matches else '⚠️ '} F0 matches target: Δ={abs(diff_f0):.3e} Hz")
    print(f"   {'✅' if sigma_f0 < 3 else '⚠️ '} Within 3σ: {sigma_f0:.2f}σ")
    
    overall_success = converged and (f0_matches or sigma_f0 < 3)
    
    print(f"\n   Overall: {'✅ TEST PASSED' if overall_success else '⚠️  NEEDS INVESTIGATION'}")
    
    if overall_success:
        print(f"\n   ✅ F0 + PHOFF FITTING WORKS!")
        print(f"   PINT-style approach successful.")
        print(f"   PHOFF absorbed the phase offset, F0 recovered correctly.")
    
    return overall_success


if __name__ == '__main__':
    try:
        success = test_f0_phoff_fitting()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
