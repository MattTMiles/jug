#!/usr/bin/env python3
"""
Test a single iteration of F0+F1 fitting to see if we're moving in the right direction
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd

# Load data
par_wrong = Path("data/pulsars/J1909-3744_tdb_wrong.par")
par_target = Path("data/pulsars/J1909-3744_tdb_refit_F0_F1.par")
tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")

params_wrong = parse_par_file(par_wrong)
params_target = parse_par_file(par_target)

toas_data = parse_tim_file_mjds(tim_file)
toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
errors_us = np.array([toa.error_us for toa in toas_data])
errors_sec = errors_us * 1e-6

print("=== Starting Parameters ===")
print(f"F0: {params_wrong['F0']:.15f}")
print(f"F1: {params_wrong['F1']:.15e}")

# Get initial residuals (just the array, not full computation)
result = compute_residuals_simple(str(par_wrong), str(tim_file))
residuals_sec = result['residuals_us'] * 1e-6
residuals_cycles = residuals_sec * params_wrong['F0']
rms_us = np.std(residuals_sec) * 1e6

print(f"\nInitial RMS: {rms_us:.4f} μs")
print(f"Residuals (cycles): min={residuals_cycles.min():.6f}, max={residuals_cycles.max():.6f}")

# Compute design matrix
derivs = compute_spin_derivatives(params_wrong, toas_mjd, ['F0', 'F1'])
M = np.column_stack([derivs['F0'], derivs['F1']])
print(f"\nDesign matrix shape: {M.shape}")
print(f"M[:,0] (dφ/dF0): min={M[:,0].min():.6e}, max={M[:,0].max():.6e}")
print(f"M[:,1] (dφ/dF1): min={M[:,1].min():.6e}, max={M[:,1].max():.6e}")
print(f"Condition number: {np.linalg.cond(M):.2e}")

# Solve for parameter updates
# Try with negate_dpars=True (maybe needed for phase derivatives?)
delta_negated, _, _ = wls_solve_svd(residuals_cycles, errors_sec, M, negate_dpars=True)
delta_normal, _, _ = wls_solve_svd(residuals_cycles, errors_sec, M, negate_dpars=False)

print(f"\nParameter updates (negate_dpars=True):")
print(f"ΔF0 = {delta_negated[0]:.15e}")
print(f"ΔF1 = {delta_negated[1]:.15e}")

print(f"\nParameter updates (negate_dpars=False):")
print(f"ΔF0 = {delta_normal[0]:.15e}")
print(f"ΔF1 = {delta_normal[1]:.15e}")

print(f"\n=== Testing with negate_dpars=True ===")
F0_new = params_wrong['F0'] + delta_negated[0]
F1_new = params_wrong['F1'] + delta_negated[1]
print(f"F0_new: {F0_new:.15f} (target: {params_target['F0']:.15f})")
print(f"F1_new: {F1_new:.15e} (target: {params_target['F1']:.15e})")
print(f"F0 error: {abs(F0_new - params_target['F0']):.2e}")
print(f"F1 error: {abs(F1_new - params_target['F1']):.2e}")

print(f"\n=== Testing with negate_dpars=False ===")
F0_new2 = params_wrong['F0'] + delta_normal[0]
F1_new2 = params_wrong['F1'] + delta_normal[1]
print(f"F0_new: {F0_new2:.15f} (target: {params_target['F0']:.15f})")
print(f"F1_new: {F1_new2:.15e} (target: {params_target['F1']:.15e})")
print(f"F0 error: {abs(F0_new2 - params_target['F0']):.2e}")
print(f"F1 error: {abs(F1_new2 - params_target['F1']):.2e}")
