#!/usr/bin/env python3
"""
Test F0 fitting with mean-subtracted residuals AND mean-subtracted derivatives.

This matches PINT's approach more closely.
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


def compute_residuals_mean_subtracted(par_file, tim_file, f0_value):
    """Compute residuals with mean subtraction (PINT style)."""
    # Update par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    updated_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'F0':
            updated_lines.append(f"F0             {f0_value:.20f}     1\n")
        else:
            updated_lines.append(line)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        temp_par = f.name
        f.writelines(updated_lines)
    
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            # Use default subtract_tzr=True which subtracts mean
            result = compute_residuals_simple(
                temp_par, tim_file, clock_dir="data/clock",
                subtract_tzr=True  # This subtracts mean!
            )
        residuals_sec = result['residuals_us'] * 1e-6
        rms_us = result['rms_us']
    finally:
        Path(temp_par).unlink()
    
    return residuals_sec, rms_us


# Load data
print("Testing F0 fitting with mean-subtracted residuals...")
data_dir = Path("data/pulsars")
par_wrong = data_dir / "J1909-3744_tdb_wrong.par"
par_correct = data_dir / "J1909-3744_tdb_refit_F0.par"
tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")

params_wrong = parse_par_file(par_wrong)
params_correct = parse_par_file(par_correct)

from jug.io.tim_reader import parse_tim_file_mjds
toas_data = parse_tim_file_mjds(tim_file)
toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
errors_us = np.array([toa.error_us for toa in toas_data])
errors_sec = errors_us * 1e-6

f0_wrong = params_wrong['F0']
f0_correct = params_correct['F0']

print(f"\nF0 wrong:   {f0_wrong:.20f} Hz")
print(f"F0 correct: {f0_correct:.20f} Hz")
print(f"Error:      {f0_correct - f0_wrong:.3e} Hz")

# Get initial residuals (mean-subtracted)
residuals, rms_old = compute_residuals_mean_subtracted(par_wrong, tim_file, f0_wrong)
print(f"\nInitial RMS: {rms_old:.6f} μs")
print(f"Residuals: mean={np.mean(residuals):.3e}, std={np.std(residuals):.3e}")

# Compute derivatives (also mean-subtract them!)
params_current = params_wrong.copy()
params_current['F0'] = f0_wrong

derivs = compute_spin_derivatives(params_current, toas_mjd, ['F0'])
M_raw = derivs['F0']

# Subtract mean from design matrix (match PINT?)
M_mean = np.mean(M_raw)
M = M_raw - M_mean

print(f"\nDesign matrix:")
print(f"  Raw: mean={np.mean(M_raw):.3e}, std={np.std(M_raw):.3e}")
print(f"  Centered: mean={np.mean(M):.3e}, std={np.std(M):.3e}")

# Check correlation
corr = np.dot(M, residuals) / (np.linalg.norm(M) * np.linalg.norm(residuals))
print(f"\nCorrelation: {corr:.6f}")

# WLS solve
M_col = M.reshape(-1, 1)
delta_params, cov, M_scaled = wls_solve_svd(residuals, errors_sec, M_col)

print(f"\nPredicted ΔF0: {delta_params[0]:.3e} Hz")
print(f"Actual error:  {f0_correct - f0_wrong:.3e} Hz")
print(f"Ratio: {delta_params[0] / (f0_correct - f0_wrong):.3f}")

# Apply step
f0_new = f0_wrong + delta_params[0]
residuals_new, rms_new = compute_residuals_mean_subtracted(par_wrong, tim_file, f0_new)

print(f"\nAfter 1 iteration:")
print(f"  F0: {f0_new:.20f}")
print(f"  ΔF0: {delta_params[0]:.3e}")
print(f"  RMS: {rms_old:.6f} → {rms_new:.6f} μs")
print(f"  Change: {rms_new - rms_old:.6f} μs")
