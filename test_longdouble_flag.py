#!/usr/bin/env python3
"""
Test longdouble_spin_pars flag - comparing standard float64 vs longdouble for F0/F1
"""
import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.utils.constants import SECS_PER_DAY
import tempfile

# Load data
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("/home/mattm/projects/HSYMT_dump/partim_real/tdb/J1909-3744.tim")
clock_dir = "data/clock"

params = parse_par_file(par_file)
toas_data = parse_tim_file_mjds(tim_file)
toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])

print("=" * 80)
print("TESTING LONGDOUBLE_SPIN_PARS FLAG: ONE ITERATION OF F0/F1 FITTING")
print("=" * 80)

def fit_one_iteration(use_longdouble):
    """Perform one iteration of F0/F1 fitting."""
    # Get initial parameters
    f0_init = get_longdouble(params, 'F0')
    f1_init = get_longdouble(params, 'F1')
    
    # Compute initial residuals
    result_init = compute_residuals_simple(
        par_file, tim_file, clock_dir=clock_dir, 
        verbose=False, longdouble_spin_pars=use_longdouble
    )
    residuals_us = result_init['residuals_us']
    errors_us = result_init['errors_us']
    
    # Compute derivatives
    derivs = compute_spin_derivatives(
        params, toas_mjd, fit_params=['F0', 'F1'],
        use_longdouble=use_longdouble
    )
    
    # Build design matrix
    design_matrix = np.column_stack([derivs['F0'], derivs['F1']])
    
    # Solve WLS
    delta_params, cov, _ = wls_solve_svd(
        residuals_us, errors_us, design_matrix, negate_dpars=False
    )
    
    # Update parameters - CONVERT TO FLOAT FIRST to avoid JAX/longdouble mixing
    delta_f0 = float(delta_params[0]) / SECS_PER_DAY
    delta_f1 = float(delta_params[1]) / SECS_PER_DAY
    
    f0_new = float(f0_init) + delta_f0
    f1_new = float(f1_init) + delta_f1
    
    # Write updated par file
    with open(par_file, 'r') as f:
        par_lines = f.readlines()
    
    new_lines = []
    for line in par_lines:
        parts = line.split()
        if len(parts) > 0 and parts[0] == 'F0':
            new_lines.append(f'F0             {f0_new:.20f}     1\n')
        elif len(parts) > 0 and parts[0] == 'F1':
            new_lines.append(f'F1             {f1_new:.20e}     1\n')
        else:
            new_lines.append(line)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as tmp:
        tmp.writelines(new_lines)
        tmp_par = tmp.name
    
    # Compute postfit residuals
    result_final = compute_residuals_simple(
        tmp_par, tim_file, clock_dir=clock_dir,
        verbose=False, longdouble_spin_pars=use_longdouble
    )
    
    Path(tmp_par).unlink()
    
    return result_final, f0_new, f1_new

# Test with longdouble
print("\n1. LONGDOUBLE SPIN PARAMETERS (longdouble_spin_pars=True):")
result_ld, f0_ld, f1_ld = fit_one_iteration(use_longdouble=True)
print(f"   F0 = {f0_ld:.20f} Hz")
print(f"   F1 = {f1_ld:.25e} Hz/s")
print(f"   Postfit RMS:  {result_ld['rms_us']:.6f} μs")
print(f"   Postfit WRMS: {result_ld['weighted_rms_us']:.6f} μs")

# Test with standard float64
print("\n2. STANDARD FLOAT64 (longdouble_spin_pars=False):")
result_std, f0_std, f1_std = fit_one_iteration(use_longdouble=False)
print(f"   F0 = {f0_std:.20f} Hz")
print(f"   F1 = {f1_std:.25e} Hz/s")
print(f"   Postfit RMS:  {result_std['rms_us']:.6f} μs")
print(f"   Postfit WRMS: {result_std['weighted_rms_us']:.6f} μs")

# Compare residuals
residuals_ld = result_ld['residuals_us']
residuals_std = result_std['residuals_us']
diff = residuals_ld - residuals_std

print("\n" + "=" * 80)
print("RESIDUAL COMPARISON")
print("=" * 80)
print(f"Difference RMS:   {np.std(diff):.6f} μs = {np.std(diff)*1000:.3f} ns")
print(f"Difference mean:  {np.mean(diff):.6f} μs")
print(f"Difference range: [{np.min(diff)*1000:.3f}, {np.max(diff)*1000:.3f}] ns")

# Create diagnostic plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

mjds = result_ld['toa_mjds']

# Plot 1: Both residuals
axes[0].plot(mjds, residuals_ld, 'o', alpha=0.5, label='Longdouble', markersize=3)
axes[0].plot(mjds, residuals_std, 'x', alpha=0.5, label='Float64', markersize=3)
axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[0].set_ylabel('Residuals (μs)')
axes[0].set_title(f'Postfit Residuals: Longdouble (WRMS={result_ld["wrms_us"]:.3f} μs) vs Float64 (WRMS={result_std["wrms_us"]:.3f} μs)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Difference
axes[1].plot(mjds, diff * 1000, 'o', markersize=2, alpha=0.6, color='C2')
axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
axes[1].set_ylabel('Difference (ns)')
axes[1].set_title(f'Residual Difference (Longdouble - Float64): RMS = {np.std(diff)*1000:.3f} ns')
axes[1].grid(True, alpha=0.3)

# Plot 3: Histogram
axes[2].hist(diff * 1000, bins=50, alpha=0.7, edgecolor='black', color='C2')
axes[2].axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
axes[2].axvline(np.mean(diff)*1000, color='orange', linestyle='--', linewidth=2, label=f'Mean = {np.mean(diff)*1000:.2f} ns')
axes[2].set_xlabel('Difference (ns)')
axes[2].set_ylabel('Count')
axes[2].set_title('Distribution of Residual Differences')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('longdouble_vs_float64_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: longdouble_vs_float64_comparison.png")

print("\n" + "=" * 80)
print("EXPECTED RESULTS:")
print("=" * 80)
print("✓ Both methods should give WRMS ≈ 0.403 μs")
print("✓ Difference should be ~20 ns RMS (per piecewise_precision_comparison.ipynb)")
print("✗ If difference >> 100 ns RMS, longdouble implementation has issues")
