#!/usr/bin/env python3
"""
Test solutions to eliminate drift from incremental updates.

Solution 1: Store residual updates in longdouble, convert to float64 only for WLS
Solution 2: Periodic longdouble recomputation to reset accumulated errors
Solution 3: Compute final residuals with longdouble using converged parameters
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
print("DRIFT ELIMINATION TEST")
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
# SOLUTION 1: Store residuals in LONGDOUBLE during updates
#------------------------------------------------------------------------------
print("SOLUTION 1: Longdouble residual storage during iteration")
print("-"*80)

# Initialize residuals in longdouble
dt_ld = np.array(dt_sec, dtype=np.longdouble)
f0_ld = np.longdouble(f0_initial)
f1_ld = np.longdouble(f1_initial)

phase_ld = dt_ld * (f0_ld + dt_ld * (f1_ld / 2.0))
phase_wrapped_ld = phase_ld - np.round(phase_ld)
residuals_ld = phase_wrapped_ld / f0_ld
weighted_mean = np.sum(np.array(residuals_ld, dtype=np.float64) * weights) / np.sum(weights)
residuals_ld = residuals_ld - np.longdouble(weighted_mean)

f0_sol1 = f0_initial
f1_sol1 = f1_initial

for iteration in range(50):
    # Convert to float64 for WLS
    residuals_f64 = np.array(residuals_ld, dtype=np.float64)
    rms = np.sqrt(np.sum(residuals_f64**2 * weights) / np.sum(weights)) * 1e6

    # Design matrix in float64
    M_f0 = -dt_sec / f0_sol1
    M_f1 = -(dt_sec**2 / 2.0) / f0_sol1
    M = np.column_stack([M_f0, M_f1])

    for j in range(2):
        col_mean = np.sum(M[:, j] * weights) / np.sum(weights)
        M[:, j] = M[:, j] - col_mean

    # WLS solve
    delta_params, cov, _ = wls_solve_svd(
        jnp.array(residuals_f64),
        jnp.array(errors_sec),
        jnp.array(M),
        negate_dpars=False
    )
    delta_params = np.array(delta_params)

    # Update parameters
    f0_sol1 += delta_params[0]
    f1_sol1 += delta_params[1]
    max_delta = np.max(np.abs(delta_params))

    # UPDATE RESIDUALS IN LONGDOUBLE!
    M_ld = np.column_stack([
        np.array(M_f0, dtype=np.longdouble),
        np.array(M_f1, dtype=np.longdouble)
    ])
    delta_ld = np.array(delta_params, dtype=np.longdouble)
    residuals_ld = residuals_ld - M_ld @ delta_ld

    if iteration < 3 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e}")

    if max_delta < 1e-20:
        print(f"  → CONVERGED at iteration {iteration+1}")
        break

residuals_sol1 = np.array(residuals_ld, dtype=np.float64)
print(f"\nFinal (SOLUTION 1):")
print(f"  F0 = {f0_sol1:.20f} Hz")
print(f"  F1 = {f1_sol1:.25e} Hz/s")
print()

#------------------------------------------------------------------------------
# SOLUTION 2: Final longdouble recomputation with converged parameters
#------------------------------------------------------------------------------
print("SOLUTION 2: Incremental fitting + final longdouble recomputation")
print("-"*80)

# Use regular incremental method (fast)
residuals_sec = compute_residuals_longdouble(dt_sec, f0_initial, f1_initial)

f0_sol2 = f0_initial
f1_sol2 = f1_initial

for iteration in range(50):
    rms = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights)) * 1e6

    M_f0 = -dt_sec / f0_sol2
    M_f1 = -(dt_sec**2 / 2.0) / f0_sol2
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

    f0_sol2 += delta_params[0]
    f1_sol2 += delta_params[1]
    max_delta = np.max(np.abs(delta_params))

    # Incremental update (float64, may accumulate error)
    residuals_sec = residuals_sec - M @ delta_params

    if iteration < 3 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e}")

    if max_delta < 1e-20:
        print(f"  → CONVERGED at iteration {iteration+1}")
        break

# FINAL STEP: Recompute residuals with converged parameters in longdouble
print("  → Recomputing final residuals in longdouble...")
residuals_sol2 = compute_residuals_longdouble(dt_sec, f0_sol2, f1_sol2)

print(f"\nFinal (SOLUTION 2):")
print(f"  F0 = {f0_sol2:.20f} Hz")
print(f"  F1 = {f1_sol2:.25e} Hz/s")
print()

#------------------------------------------------------------------------------
# BASELINE: Longdouble recomputation each iteration
#------------------------------------------------------------------------------
print("BASELINE: Longdouble recomputation (ground truth)")
print("-"*80)

f0_ld = f0_initial
f1_ld = f1_initial

for iteration in range(50):
    residuals_sec = compute_residuals_longdouble(dt_sec, f0_ld, f1_ld)
    rms = np.sqrt(np.sum(residuals_sec**2 * weights) / np.sum(weights)) * 1e6

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

    if iteration < 3 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e}")

    if max_delta < 1e-20:
        print(f"  → CONVERGED at iteration {iteration+1}")
        break

residuals_baseline = residuals_sec
print(f"\nFinal (BASELINE):")
print(f"  F0 = {f0_ld:.20f} Hz")
print(f"  F1 = {f1_ld:.25e} Hz/s")
print()

#------------------------------------------------------------------------------
# COMPARISON
#------------------------------------------------------------------------------
print("="*80)
print("COMPARISON VS LONGDOUBLE BASELINE")
print("="*80)
print()

diff1_ns = (residuals_sol1 - residuals_baseline) * 1e9
diff2_ns = (residuals_sol2 - residuals_baseline) * 1e9

early_idx = tdb_mjd < np.percentile(tdb_mjd, 25)
late_idx = tdb_mjd > np.percentile(tdb_mjd, 75)

print("Solution 1 (Longdouble residual storage):")
print(f"  RMS:   {np.std(diff1_ns):.6f} ns")
print(f"  Max:   {np.max(np.abs(diff1_ns)):.6f} ns")
print(f"  Drift: {np.mean(diff1_ns[late_idx]) - np.mean(diff1_ns[early_idx]):.6f} ns")
print()

print("Solution 2 (Final longdouble recomputation):")
print(f"  RMS:   {np.std(diff2_ns):.6f} ns")
print(f"  Max:   {np.max(np.abs(diff2_ns)):.6f} ns")
print(f"  Drift: {np.mean(diff2_ns[late_idx]) - np.mean(diff2_ns[early_idx]):.6f} ns")
print()

#------------------------------------------------------------------------------
# PLOT
#------------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Solution 1
ax = axes[0, 0]
ax.plot(tdb_mjd, diff1_ns, 'b.', ms=2, alpha=0.6)
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_ylabel('Difference (ns)', fontsize=11)
ax.set_title(f'Solution 1: Longdouble Storage\nRMS={np.std(diff1_ns):.4f} ns, Drift={np.mean(diff1_ns[late_idx]) - np.mean(diff1_ns[early_idx]):.4f} ns',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(diff1_ns, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', ls='--', lw=2)
ax.set_xlabel('Difference (ns)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution (Solution 1)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Solution 2
ax = axes[1, 0]
ax.plot(tdb_mjd, diff2_ns, 'g.', ms=2, alpha=0.6)
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_xlabel('MJD', fontsize=11)
ax.set_ylabel('Difference (ns)', fontsize=11)
ax.set_title(f'Solution 2: Final Recomputation\nRMS={np.std(diff2_ns):.4f} ns, Drift={np.mean(diff2_ns[late_idx]) - np.mean(diff2_ns[early_idx]):.4f} ns',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.hist(diff2_ns, bins=50, color='green', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', ls='--', lw=2)
ax.set_xlabel('Difference (ns)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('Distribution (Solution 2)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('drift_elimination_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: drift_elimination_comparison.png")
print()

print("="*80)
print("RECOMMENDATION")
print("="*80)
print()

if np.max(np.abs(diff2_ns)) < 0.1:
    print("✓ SOLUTION 2 is the winner!")
    print()
    print("Strategy:")
    print("  1. Initialize residuals in longdouble (once)")
    print("  2. Update incrementally in float64 (fast iterations)")
    print("  3. Final recomputation in longdouble with converged params")
    print()
    print("Benefits:")
    print("  • Fast iterations (float64 arithmetic)")
    print("  • Perfect final precision (longdouble recomputation)")
    print("  • JAX-compatible iteration loop")
    print("  • No drift in final residuals!")
elif np.max(np.abs(diff1_ns)) < 0.1:
    print("✓ SOLUTION 1 is the winner!")
    print()
    print("Strategy:")
    print("  • Store residuals in longdouble throughout")
    print("  • Convert to float64 only for WLS solve")
    print()
    print("Benefits:")
    print("  • No accumulated errors")
    print("  • Perfect precision")
    print()
    print("Drawbacks:")
    print("  • May be slower (longdouble arithmetic each iteration)")
    print("  • Less JAX-compatible")
else:
    print("Both solutions have residual drift. Need further investigation.")
