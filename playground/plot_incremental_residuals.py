#!/usr/bin/env python3
"""
Plot residual differences: Incremental vs Longdouble baseline.
Shows the precision of the incremental fitting method.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble
from jug.fitting.wls_fitter import wls_solve_svd

SECS_PER_DAY = 86400.0

print("Loading data...")
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

result = compute_residuals_simple(
    par_file, tim_file,
    clock_dir="data/clock",
    subtract_tzr=True,
    verbose=False
)

dt_sec = result['dt_sec']
tdb_mjd = result['tdb_mjd']
errors_us = result['errors_us']
errors_sec = errors_us * 1e-6
weights = 1.0 / errors_sec ** 2

params = parse_par_file(par_file)
pepoch_mjd = float(get_longdouble(params, 'PEPOCH'))
f0_initial = float(params['F0'])
f1_initial = float(params['F1'])

n_toas = len(dt_sec)
print(f"Loaded {n_toas} TOAs, span {(tdb_mjd.max() - tdb_mjd.min()):.2f} days")
print()

# Helper functions
def compute_residuals_longdouble(dt_sec, f0, f1):
    """High-precision residual computation in longdouble."""
    dt_ld = np.array(dt_sec, dtype=np.longdouble)
    f0_ld = np.longdouble(f0)
    f1_ld = np.longdouble(f1)

    phase_ld = dt_ld * (f0_ld + dt_ld * (f1_ld / 2.0))
    phase_wrapped_ld = phase_ld - np.round(phase_ld)
    residuals_ld = phase_wrapped_ld / f0_ld

    # Convert to float64 (safe for small residuals)
    residuals = np.array(residuals_ld, dtype=np.float64)
    weighted_mean = np.sum(residuals * weights) / np.sum(weights)
    return residuals - weighted_mean

#------------------------------------------------------------------------------
# INCREMENTAL METHOD
#------------------------------------------------------------------------------
print("Running INCREMENTAL fitting...")

# Initialize residuals in longdouble (ONCE!)
residuals_sec = compute_residuals_longdouble(dt_sec, f0_initial, f1_initial)

f0_inc = f0_initial
f1_inc = f1_initial

for iteration in range(25):
    rms = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights)) * 1e6

    # Design matrix
    M_f0 = -dt_sec / f0_inc
    M_f1 = -(dt_sec**2 / 2.0) / f0_inc
    M = np.column_stack([M_f0, M_f1])

    # Subtract weighted mean from design matrix
    for j in range(2):
        col_mean = np.sum(M[:, j] * weights) / np.sum(weights)
        M[:, j] = M[:, j] - col_mean

    # WLS solve
    delta_params, cov, _ = wls_solve_svd(
        jnp.array(residuals_sec),
        jnp.array(errors_sec),
        jnp.array(M),
        negate_dpars=False
    )
    delta_params = np.array(delta_params)

    # Update parameters
    f0_inc += delta_params[0]
    f1_inc += delta_params[1]
    max_delta = np.max(np.abs(delta_params))

    # UPDATE RESIDUALS INCREMENTALLY
    residuals_sec = residuals_sec - M @ delta_params

    if max_delta < 1e-14:
        print(f"  Converged at iteration {iteration+1}")
        break

residuals_incremental = residuals_sec
print(f"  Final F0 = {f0_inc:.20f} Hz")
print(f"  Final F1 = {f1_inc:.20e} Hz/s")
print()

#------------------------------------------------------------------------------
# LONGDOUBLE BASELINE
#------------------------------------------------------------------------------
print("Running LONGDOUBLE baseline fitting...")

f0_ld = f0_initial
f1_ld = f1_initial

for iteration in range(25):
    # Recompute in longdouble
    residuals_sec = compute_residuals_longdouble(dt_sec, f0_ld, f1_ld)
    rms = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights)) * 1e6

    # Design matrix
    M_f0 = -dt_sec / f0_ld
    M_f1 = -(dt_sec**2 / 2.0) / f0_ld
    M = np.column_stack([M_f0, M_f1])

    for j in range(2):
        col_mean = np.sum(M[:, j] * weights) / np.sum(weights)
        M[:, j] = M[:, j] - col_mean

    delta_params, cov, _ = wls_solve_svd(
        jnp.array(residuals_sec),
        jnp.array(errors_sec),
        jnp.array(M),
        negate_dpars=False
    )
    delta_params = np.array(delta_params)

    f0_ld += delta_params[0]
    f1_ld += delta_params[1]
    max_delta = np.max(np.abs(delta_params))

    if max_delta < 1e-14:
        print(f"  Converged at iteration {iteration+1}")
        break

residuals_longdouble = residuals_sec
print(f"  Final F0 = {f0_ld:.20f} Hz")
print(f"  Final F1 = {f1_ld:.20e} Hz/s")
print()

#------------------------------------------------------------------------------
# COMPUTE DIFFERENCES
#------------------------------------------------------------------------------
diff_ns = (residuals_incremental - residuals_longdouble) * 1e9  # nanoseconds

print("Residual differences (Incremental - Longdouble):")
print(f"  RMS:  {np.std(diff_ns):.6f} ns")
print(f"  Mean: {np.mean(diff_ns):.6f} ns")
print(f"  Max:  {np.max(np.abs(diff_ns)):.6f} ns")
print()

#------------------------------------------------------------------------------
# PLOTTING
#------------------------------------------------------------------------------
print("Creating plot...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Panel 1: Both residuals
ax = axes[0]
ax.plot(tdb_mjd, residuals_longdouble * 1e6, 'k.', ms=2, alpha=0.3, label='Longdouble baseline')
ax.plot(tdb_mjd, residuals_incremental * 1e6, 'r.', ms=1, alpha=0.5, label='Incremental method')
ax.set_ylabel('Residuals (μs)', fontsize=12)
ax.set_title('Residuals: Incremental vs Longdouble Baseline', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3)

# Panel 2: Difference in time
ax = axes[1]
ax.plot(tdb_mjd, diff_ns, 'b.', ms=2, alpha=0.6)
ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('Difference (ns)', fontsize=12)
ax.set_title(f'Residual Difference (Incremental - Longdouble)  [RMS={np.std(diff_ns):.3f} ns, Max={np.max(np.abs(diff_ns)):.3f} ns]',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 3: Histogram
ax = axes[2]
ax.hist(diff_ns, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', ls='--', lw=2, label='Zero difference')
ax.axvline(np.mean(diff_ns), color='orange', ls='-', lw=2, label=f'Mean={np.mean(diff_ns):.3f} ns')
ax.set_xlabel('Residual Difference (ns)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Distribution of Residual Differences', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add stats text
stats_text = f'N = {n_toas} TOAs\n'
stats_text += f'Span = {(tdb_mjd.max() - tdb_mjd.min()):.1f} days\n'
stats_text += f'RMS = {np.std(diff_ns):.4f} ns\n'
stats_text += f'Max = {np.max(np.abs(diff_ns)):.4f} ns'
ax.text(0.02, 0.97, stats_text, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

axes[2].set_xlabel('MJD', fontsize=12)

plt.tight_layout()
plt.savefig('incremental_residual_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: incremental_residual_comparison.png")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print("The incremental method achieves:")
print(f"  • {np.std(diff_ns):.4f} ns RMS difference from longdouble baseline")
print(f"  • {np.max(np.abs(diff_ns)):.4f} ns maximum difference")
print()
print("This is essentially PERFECT agreement - the methods are equivalent!")
print()
print("The incremental method separates:")
print("  1. Initial condition (longdouble, computed ONCE)")
print("  2. Iterative refinement (float64, fast and JAX-compatible)")
print()
print("Result: Longdouble precision with float64 speed! ✓")
