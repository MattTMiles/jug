#!/usr/bin/env python3
"""
Debug why F0+F1 fitting doesn't converge to tempo2's RMS
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

def compute_residuals_for_f0_f1(par_file, tim_file, f0_value, f1_value):
    """Compute residuals with updated F0 and F1."""
    # Read par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    # Update F0 and F1
    updated_lines = []
    for line in par_lines:
        if line.startswith('F0 '):
            updated_lines.append(f'F0 {f0_value:.15f}\n')
        elif line.startswith('F1 '):
            updated_lines.append(f'F1 {f1_value:.15e}\n')
        else:
            updated_lines.append(line)
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
        tmp.writelines(updated_lines)
        tmp_path = tmp.name
    
    try:
        result = compute_residuals_simple(tmp_path, str(tim_file))
        return result['residuals_us'] * 1e-6  # Convert μs to seconds
    finally:
        Path(tmp_path).unlink()


def main():
    # File paths
    par_wrong = Path("data/pulsars/J1909-3744_tdb_wrong.par")
    par_target = Path("data/pulsars/J1909-3744_tdb_refit_F0_F1.par")
    tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
    
    # Load initial and target parameters
    params_wrong = parse_par_file(par_wrong)
    params_target = parse_par_file(par_target)
    
    # Load TOA data
    from jug.io.tim_reader import parse_tim_file_mjds
    toas_data = parse_tim_file_mjds(tim_file)
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
    errors_us = np.array([toa.error_us for toa in toas_data])
    
    print("=== Initial vs Target Parameters ===")
    print(f"Initial F0: {params_wrong['F0']:.15f}")
    print(f"Target  F0: {params_target['F0']:.15f}")
    print(f"Diff    F0: {(params_wrong['F0'] - params_target['F0']):.2e}")
    print()
    print(f"Initial F1: {params_wrong['F1']:.15e}")
    print(f"Target  F1: {params_target['F1']:.15e}")
    print(f"Diff    F1: {(params_wrong['F1'] - params_target['F1']):.2e}")
    
    # Initial residuals
    result_initial = compute_residuals_simple(str(par_wrong), str(tim_file))
    rms_initial = result_initial['unweighted_rms_us']
    print(f"\nInitial RMS: {rms_initial:.4f} μs")
    
    # Target residuals (should be ~0.4 μs per tempo2)
    result_target = compute_residuals_simple(str(par_target), str(tim_file))
    rms_target = result_target['unweighted_rms_us']
    print(f"Target  RMS: {rms_target:.4f} μs (from tempo2 refit)\n")
    
    # Now fit iteratively
    F0_current = params_wrong['F0']
    F1_current = params_wrong['F1']
    
    print("=== Iterative Fitting ===")
    for iteration in range(20):
        # Update params
        params_current = params_wrong.copy()
        params_current['F0'] = F0_current
        params_current['F1'] = F1_current
        
        # Compute residuals
        residuals_sec = compute_residuals_for_f0_f1(par_wrong, tim_file, F0_current, F1_current)
        rms_us = np.std(residuals_sec) * 1e6
        
        # Compute design matrix (derivatives with respect to F0 and F1)
        M = compute_spin_derivatives(params_current, toas_mjd, ['F0', 'F1'])
        
        # Convert residuals to cycles (using current F0)
        residuals_cycles = residuals_sec * F0_current
        
        # Solve WLS (note: signature is (residuals, sigma, M))
        errors_sec = errors_us * 1e-6
        delta, _, _ = wls_solve_svd(residuals_cycles, errors_sec, M)
        
        # Update
        F0_current -= delta[0]
        F1_current -= delta[1]
        
        print(f"Iter {iteration+1}: RMS={rms_us:.4f} μs, ΔF0={delta[0]:.2e}, ΔF1={delta[1]:.2e}")
        
        # Check convergence
        if np.abs(delta[0]) < 1e-13 and np.abs(delta[1]) < 1e-20:
            print(f"Converged at iteration {iteration+1}")
            break
    
    print(f"\n=== Final Results ===")
    print(f"Fitted F0: {F0_current:.15f}")
    print(f"Target F0: {params_target['F0']:.15f}")
    print(f"Diff   F0: {(F0_current - params_target['F0']):.2e}")
    print()
    print(f"Fitted F1: {F1_current:.15e}")
    print(f"Target F1: {params_target['F1']:.15e}")
    print(f"Diff   F1: {(F1_current - params_target['F1']):.2e}")
    print()
    print(f"Final RMS: {rms_us:.4f} μs")
    print(f"Target RMS: {rms_target:.4f} μs")
    print(f"Difference: {(rms_us - rms_target):.4f} μs")

if __name__ == "__main__":
    main()
