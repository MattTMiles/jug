#!/usr/bin/env python3
"""
Investigate the systematic drift in incremental fitting.

Test 1: Run longdouble baseline to FULL convergence
Test 2: Force identical final parameters and recompute residuals
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

print("="*80)
print("DRIFT INVESTIGATION: Incremental vs Longdouble")
print("="*80)
print()

# Load data
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

print("Loading data...")
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
print(f"  Loaded {n_toas} TOAs")
print(f"  Initial F0 = {f0_initial:.15f} Hz")
print(f"  Initial F1 = {f1_initial:.6e} Hz/s")
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
# TEST 1: Run INCREMENTAL to convergence
#------------------------------------------------------------------------------
print("TEST 1: INCREMENTAL METHOD (full convergence)")
print("-"*80)

# Initialize residuals in longdouble (ONCE!)
residuals_sec = compute_residuals_longdouble(dt_sec, f0_initial, f1_initial)

f0_inc = f0_initial
f1_inc = f1_initial

for iteration in range(50):  # More iterations to ensure full convergence
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

    if iteration < 5 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e}, "
              f"ΔF0={delta_params[0]:.2e}, ΔF1={delta_params[1]:.2e}")

    if max_delta < 1e-20:  # Much tighter convergence
        print(f"  → CONVERGED at iteration {iteration+1} (threshold: 1e-20)")
        break

residuals_incremental = residuals_sec.copy()
print(f"\nFinal (INCREMENTAL):")
print(f"  F0 = {f0_inc:.20f} Hz")
print(f"  F1 = {f1_inc:.25e} Hz/s")
print()

#------------------------------------------------------------------------------
# TEST 2: Run LONGDOUBLE to FULL convergence
#------------------------------------------------------------------------------
print("TEST 2: LONGDOUBLE BASELINE (full convergence)")
print("-"*80)

f0_ld = f0_initial
f1_ld = f1_initial

for iteration in range(50):  # More iterations
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

    if iteration < 5 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e}, "
              f"ΔF0={delta_params[0]:.2e}, ΔF1={delta_params[1]:.2e}")

    if max_delta < 1e-20:  # Much tighter convergence
        print(f"  → CONVERGED at iteration {iteration+1} (threshold: 1e-20)")
        break

residuals_longdouble = residuals_sec.copy()
print(f"\nFinal (LONGDOUBLE):")
print(f"  F0 = {f0_ld:.20f} Hz")
print(f"  F1 = {f1_ld:.25e} Hz/s")
print()

#------------------------------------------------------------------------------
# COMPARISON 1: Different convergence states
#------------------------------------------------------------------------------
print("="*80)
print("COMPARISON 1: Tighter convergence impact")
print("="*80)
print()

print("Parameter differences:")
print(f"  ΔF0 = {f0_inc - f0_ld:+.3e} Hz")
print(f"  ΔF1 = {f1_inc - f1_ld:+.3e} Hz/s")
print()

diff_ns_1 = (residuals_incremental - residuals_longdouble) * 1e9
print("Residual differences (after full convergence):")
print(f"  RMS:  {np.std(diff_ns_1):.6f} ns")
print(f"  Mean: {np.mean(diff_ns_1):.6f} ns")
print(f"  Max:  {np.max(np.abs(diff_ns_1)):.6f} ns")
print()

# Check for drift
early_idx = tdb_mjd < np.percentile(tdb_mjd, 25)
late_idx = tdb_mjd > np.percentile(tdb_mjd, 75)
drift_1 = np.mean(diff_ns_1[late_idx]) - np.mean(diff_ns_1[early_idx])
print(f"Systematic drift (late - early): {drift_1:.6f} ns")
print()

#------------------------------------------------------------------------------
# TEST 3: Force IDENTICAL parameters
#------------------------------------------------------------------------------
print("="*80)
print("TEST 3: Forcing identical final parameters")
print("="*80)
print()

print("Using incremental final parameters for both methods...")
print(f"  F0 = {f0_inc:.20f} Hz")
print(f"  F1 = {f1_inc:.25e} Hz/s")
print()

# Recompute incremental residuals (should be same as current)
residuals_inc_final = residuals_incremental.copy()

# Recompute longdouble residuals with SAME parameters
residuals_ld_same_params = compute_residuals_longdouble(dt_sec, f0_inc, f1_inc)

diff_ns_2 = (residuals_inc_final - residuals_ld_same_params) * 1e9
print("Residual differences (identical parameters):")
print(f"  RMS:  {np.std(diff_ns_2):.6f} ns")
print(f"  Mean: {np.mean(diff_ns_2):.6f} ns")
print(f"  Max:  {np.max(np.abs(diff_ns_2)):.6f} ns")
print()

# Check for drift
drift_2 = np.mean(diff_ns_2[late_idx]) - np.mean(diff_ns_2[early_idx])
print(f"Systematic drift (late - early): {drift_2:.6f} ns")
print()

#------------------------------------------------------------------------------
# PLOTTING
#------------------------------------------------------------------------------
print("Creating diagnostic plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Difference after full convergence
ax = axes[0, 0]
ax.plot(tdb_mjd, diff_ns_1, 'b.', ms=2, alpha=0.6)
ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('Difference (ns)', fontsize=11)
ax.set_xlabel('MJD', fontsize=11)
ax.set_title(f'After Full Convergence\nRMS={np.std(diff_ns_1):.3f} ns, Drift={drift_1:.3f} ns',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 2: Histogram after full convergence
ax = axes[0, 1]
ax.hist(diff_ns_1, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(diff_ns_1), color='orange', ls='-', lw=2, label=f'Mean={np.mean(diff_ns_1):.3f} ns')
ax.set_xlabel('Residual Difference (ns)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution (Full Convergence)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Panel 3: Difference with identical parameters
ax = axes[1, 0]
ax.plot(tdb_mjd, diff_ns_2, 'g.', ms=2, alpha=0.6)
ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
ax.set_ylabel('Difference (ns)', fontsize=11)
ax.set_xlabel('MJD', fontsize=11)
ax.set_title(f'Identical Parameters (F0, F1)\nRMS={np.std(diff_ns_2):.3f} ns, Drift={drift_2:.3f} ns',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Histogram with identical parameters
ax = axes[1, 1]
ax.hist(diff_ns_2, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(diff_ns_2), color='orange', ls='-', lw=2, label=f'Mean={np.mean(diff_ns_2):.3f} ns')
ax.set_xlabel('Residual Difference (ns)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution (Identical Parameters)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('incremental_drift_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: incremental_drift_analysis.png")
print()

#------------------------------------------------------------------------------
# SUMMARY
#------------------------------------------------------------------------------
print("="*80)
print("SUMMARY")
print("="*80)
print()

print("Test 1 - Full convergence both methods:")
print(f"  • Residual difference: {np.std(diff_ns_1):.3f} ns RMS")
print(f"  • Systematic drift: {drift_1:.3f} ns")
print(f"  • Parameter differences: ΔF0={f0_inc - f0_ld:.2e} Hz, ΔF1={f1_inc - f1_ld:.2e} Hz/s")
print()

print("Test 2 - Identical final parameters:")
print(f"  • Residual difference: {np.std(diff_ns_2):.3f} ns RMS")
print(f"  • Systematic drift: {drift_2:.3f} ns")
print()

if np.std(diff_ns_2) < 0.01:
    print("✓ SUCCESS! With identical parameters, methods agree to <0.01 ns!")
    print()
    print("Conclusion: The drift was due to slight parameter differences from")
    print("different convergence paths. The incremental method is mathematically")
    print("equivalent to longdouble recomputation!")
elif np.std(diff_ns_2) < 1.0:
    print("✓ GOOD! With identical parameters, methods agree to sub-ns precision.")
    print()
    print(f"Remaining {np.std(diff_ns_2):.3f} ns is likely from accumulated float64")
    print("rounding in incremental updates. Still excellent for pulsar timing!")
else:
    print("⚠ ISSUE: Even with identical parameters, there's residual drift.")
    print()
    print("This suggests incremental updates accumulate errors. May need:")
    print("  1. Periodic longdouble recomputation")
    print("  2. Higher precision for residual storage")
    print("  3. Different update strategy")
