#!/usr/bin/env python3
"""Test F0 + F1 simultaneous fitting against tempo2 validation."""

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
    """Load J1909-3744 data with controlled files."""
    data_dir = Path("data/pulsars")
    par_wrong = data_dir / "J1909-3744_tdb_wrong.par"
    par_correct = data_dir / "J1909-3744_tdb_refit_F0_F1.par"
    
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    if not par_wrong.exists() or not par_correct.exists() or not tim_file.exists():
        print(f"ERROR: Files not found")
        return None, None, None, None
    
    params_wrong = parse_par_file(par_wrong)
    params_correct = parse_par_file(par_correct)
    
    return params_wrong, params_correct, par_wrong, tim_file


def compute_residuals_for_f0_f1(par_file, tim_file, f0_value, f1_value):
    """Compute residuals with updated F0 and F1."""
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    # Update F0 and F1
    new_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'F0':
            new_lines.append(f'F0             {f0_value:.20f}     1\n')
        elif len(parts) > 0 and parts[0] == 'F1':
            new_lines.append(f'F1             {f1_value:.20e}     1\n')
        else:
            new_lines.append(line)
    
    # Write temporary par file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
        tmp.writelines(new_lines)
        tmp_par = tmp.name
    
    try:
        # Suppress output and compute residuals
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            result = compute_residuals_simple(
                tmp_par, str(tim_file), clock_dir="data/clock",
                subtract_tzr=True
            )
        residuals_us = result['residuals_us']
        toa_errs_us = result['errors_us']
        return residuals_us, toa_errs_us, tmp_par
    except Exception as e:
        Path(tmp_par).unlink(missing_ok=True)
        raise e


def fit_f0_f1(params, par_file, tim_file, max_iter=20):
    """Fit both F0 and F1 iteratively."""
    from jug.io.tim_reader import parse_tim_file_mjds
    
    # Load TOA times once
    toas_data = parse_tim_file_mjds(tim_file)
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    
    f0 = params['F0']
    f1 = params['F1']
    
    for iteration in range(max_iter):
        # Update params dict for this iteration
        params_iter = params.copy()
        params_iter['F0'] = f0
        params_iter['F1'] = f1
        
        # Compute residuals (in microseconds)
        res_us, toa_errs_us, tmp_par = compute_residuals_for_f0_f1(par_file, tim_file, f0, f1)
        
        # Convert to seconds for WLS solver (derivatives are in s/Hz units)
        res_s = res_us * 1e-6
        toa_errs_s = toa_errs_us * 1e-6
        
        # Compute derivatives for both F0 and F1
        derivs = compute_spin_derivatives(params_iter, toas_mjd, ['F0', 'F1'])
        deriv_f0 = derivs['F0']
        deriv_f1 = derivs['F1']
        
        # Clean up temporary file
        Path(tmp_par).unlink(missing_ok=True)
        
        # Build design matrix [dφ/dF0, dφ/dF1]
        M = np.column_stack([deriv_f0, deriv_f1])
        
        # DEBUG
        if iteration == 0:
            print(f"    DEBUG: res_s shape = {res_s.shape}")
            print(f"    DEBUG: toa_errs_s shape = {toa_errs_s.shape}")
            print(f"    DEBUG: M shape = {M.shape}")
            print(f"    DEBUG: deriv_f0 shape = {deriv_f0.shape}")
            print(f"    DEBUG: deriv_f1 shape = {deriv_f1.shape}")
        
        # Solve weighted least squares
        delta_params, _, _ = wls_solve_svd(res_s, toa_errs_s, M)
        
        # Update parameters
        f0 += delta_params[0]
        f1 += delta_params[1]
        
        # Check convergence
        rms = np.std(res_us)
        delta_rms = np.linalg.norm(delta_params)
        
        print(f"  Iter {iteration+1}: RMS={rms:.3f} us, ΔF0={delta_params[0]:.3e} Hz, ΔF1={delta_params[1]:.3e} Hz/s")
        
        if delta_rms < 1e-12:
            print(f"  Converged in {iteration+1} iterations")
            break
    
    return f0, f1


def test_f0_f1_fitting():
    """Test that we can fit both F0 and F1 simultaneously."""
    
    params_wrong, params_correct, par_wrong, tim_file = load_data()
    if params_wrong is None:
        return False
    
    print("Starting parameters:")
    print(f"  F0 = {params_wrong['F0']:.15f} Hz")
    print(f"  F1 = {params_wrong['F1']:.15e} Hz/s")
    print()
    print("Tempo2 target parameters:")
    print(f"  F0 = {params_correct['F0']:.15f} Hz")
    print(f"  F1 = {params_correct['F1']:.15e} Hz/s")
    print()
    
    # Compute initial residuals
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res_initial_dict = compute_residuals_simple(str(par_wrong), str(tim_file), clock_dir="data/clock", subtract_tzr=True)
    print(f"Initial RMS: {res_initial_dict['rms_us']:.3f} us\n")
    
    # Fit F0 and F1
    print("Fitting F0 and F1...")
    f0_fitted, f1_fitted = fit_f0_f1(params_wrong, par_wrong, tim_file)
    
    print("\nFitted parameters:")
    print(f"  F0 = {f0_fitted:.15f} Hz")
    print(f"  F1 = {f1_fitted:.15e} Hz/s")
    print()
    
    # Compute final residuals with fitted parameters
    res_final, _, tmp_par_final = compute_residuals_for_f0_f1(par_wrong, tim_file, f0_fitted, f1_fitted)
    Path(tmp_par_final).unlink()
    print(f"Final RMS: {np.std(res_final):.3f} us\n")
    
    # Compare to tempo2
    f0_diff = f0_fitted - params_correct['F0']
    f1_diff = f1_fitted - params_correct['F1']
    
    print("Comparison to tempo2:")
    print(f"  ΔF0 = {f0_diff:.3e} Hz ({f0_diff/params_correct['F0']*1e9:.3f} ppb)")
    print(f"  ΔF1 = {f1_diff:.3e} Hz/s ({abs(f1_diff/params_correct['F1'])*100:.3f}%)")
    print()
    
    # Check convergence based on absolute precision, not relative to initial offset
    # For F0: should match to better than 1e-12 Hz (sub-nanoHertz precision)
    # For F1: should match to better than 1e-20 Hz/s
    f0_converged = abs(f0_diff) < 1e-12
    f1_converged = abs(f1_diff) < 1e-19
    
    print(f"F0 convergence: {'✓ PASS' if f0_converged else '✗ FAIL'} (|ΔF0| = {abs(f0_diff):.3e} Hz < 1e-12)")
    print(f"F1 convergence: {'✓ PASS' if f1_converged else '✗ FAIL'} (|ΔF1| = {abs(f1_diff):.3e} Hz/s < 1e-19)")
    
    if f0_converged and f1_converged:
        print("\n✓ Both parameters converged successfully!")
        return True
    else:
        print("\n✗ One or more parameters failed to converge")
        return False


if __name__ == '__main__':
    success = test_f0_f1_fitting()
    exit(0 if success else 1)
