#!/usr/bin/env python3
"""
JAX implementation of the incremental fitting method.

Strategy (SOLUTION 2):
  1. Initialize residuals in longdouble (ONCE, outside JAX)
  2. Iterate with JAX float64 (FAST, JIT-compiled, autodiff-capable)
  3. Final recomputation in longdouble (ONCE, perfect precision)

This should achieve:
  ✓ Longdouble precision (0.0009 ns RMS)
  ✓ JAX speed (10-60× faster)
  ✓ Autodiff for derivatives
  ✓ GPU-ready
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad
import matplotlib.pyplot as plt
from pathlib import Path
import time

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble
from jug.fitting.wls_fitter import wls_solve_svd

SECS_PER_DAY = 86400.0

print("="*80)
print("JAX INCREMENTAL FITTER TEST")
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

#------------------------------------------------------------------------------
# Helper functions (outside JAX)
#------------------------------------------------------------------------------

def compute_residuals_longdouble(dt_sec, f0, f1, weights):
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
# JAX implementation
#------------------------------------------------------------------------------

@jit
def wls_solve_jax(residuals, weights, M):
    """Weighted least squares solve using JAX (JIT-compiled)."""
    # Weighted design matrix: W^(1/2) * M
    sqrt_w = jnp.sqrt(weights)
    M_weighted = M * sqrt_w[:, None]
    
    # Weighted residuals: W^(1/2) * r
    r_weighted = residuals * sqrt_w
    
    # Solve using lstsq: M_weighted @ delta = r_weighted
    delta_params, residuals_lstsq, rank, s = jnp.linalg.lstsq(M_weighted, r_weighted, rcond=None)
    
    return delta_params

@jit
def compute_design_matrix(dt_sec, f0, f1, weights):
    """Compute design matrix and apply zero-weighted-mean constraint."""
    # Partial derivatives
    M_f0 = -dt_sec / f0
    M_f1 = -(dt_sec**2 / 2.0) / f0
    M = jnp.column_stack([M_f0, M_f1])
    
    # Zero weighted mean
    sum_weights = jnp.sum(weights)
    for j in range(2):
        col_mean = jnp.sum(M[:, j] * weights) / sum_weights
        M = M.at[:, j].set(M[:, j] - col_mean)
    
    return M

@jit
def fitting_iteration_jax(residuals, dt_sec, f0, f1, weights):
    """
    Single fitting iteration in JAX.
    
    Returns:
        residuals_new: Updated residuals
        f0_new: Updated F0
        f1_new: Updated F1
        delta_params: Parameter updates
        rms: RMS residual
    """
    # Compute design matrix
    M = compute_design_matrix(dt_sec, f0, f1, weights)
    
    # WLS solve
    delta_params = wls_solve_jax(residuals, weights, M)
    
    # Update residuals incrementally
    residuals_new = residuals - M @ delta_params
    
    # Update parameters
    f0_new = f0 + delta_params[0]
    f1_new = f1 + delta_params[1]
    
    # Compute RMS
    rms = jnp.sqrt(jnp.sum(residuals**2 * weights) / jnp.sum(weights))
    
    return residuals_new, f0_new, f1_new, delta_params, rms

#------------------------------------------------------------------------------
# METHOD 1: JAX INCREMENTAL FITTER
#------------------------------------------------------------------------------

print("METHOD 1: JAX Incremental Fitter")
print("-"*80)

# STEP 1: Initialize residuals in longdouble (ONCE)
print("  Step 1: Initializing residuals in longdouble...")
t0 = time.time()
residuals_init = compute_residuals_longdouble(dt_sec, f0_initial, f1_initial, weights)
t_init = time.time() - t0
print(f"    ✓ Initialized in {t_init*1000:.2f} ms")

# Convert to JAX arrays
residuals_jax = jnp.array(residuals_init)
dt_jax = jnp.array(dt_sec)
weights_jax = jnp.array(weights)
f0_jax = f0_initial
f1_jax = f1_initial

# STEP 2: Iterate with JAX (FAST)
print("  Step 2: Iterating with JAX (JIT-compiled)...")
print()

# First iteration will include JIT compilation time
t0 = time.time()
history_jax = []

for iteration in range(50):
    iter_start = time.time()
    
    residuals_jax, f0_jax, f1_jax, delta_params, rms = fitting_iteration_jax(
        residuals_jax, dt_jax, f0_jax, f1_jax, weights_jax
    )
    
    # Block until computation finishes
    delta_params.block_until_ready()
    iter_time = time.time() - iter_start
    
    max_delta = float(jnp.max(jnp.abs(delta_params)))
    rms_us = float(rms) * 1e6
    
    history_jax.append({
        'iteration': iteration + 1,
        'rms': rms_us,
        'max_delta': max_delta,
        'time': iter_time
    })
    
    if iteration == 0:
        print(f"    Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} (time={iter_time*1000:.1f} ms - includes JIT)")
    elif iteration < 3 or max_delta < 1e-14:
        print(f"    Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} (time={iter_time*1000:.1f} ms)")
    
    if max_delta < 1e-20:
        print(f"    → CONVERGED at iteration {iteration+1}")
        break

t_iter = time.time() - t0
residuals_jax_final = np.array(residuals_jax)

# STEP 3: Final recomputation in longdouble (ONCE)
print()
print("  Step 3: Final recomputation in longdouble...")
t0 = time.time()
residuals_jax_ld_final = compute_residuals_longdouble(dt_sec, float(f0_jax), float(f1_jax), weights)
t_final = time.time() - t0
print(f"    ✓ Recomputed in {t_final*1000:.2f} ms")

total_time_jax = t_init + t_iter + t_final

print()
print(f"Final (JAX METHOD):")
print(f"  F0 = {float(f0_jax):.20f} Hz")
print(f"  F1 = {float(f1_jax):.25e} Hz/s")
print(f"  Total time: {total_time_jax*1000:.1f} ms")
print()

#------------------------------------------------------------------------------
# METHOD 2: NUMPY/LONGDOUBLE BASELINE (for comparison)
#------------------------------------------------------------------------------

print("METHOD 2: Numpy/Longdouble Baseline")
print("-"*80)

t0 = time.time()

f0_ld = f0_initial
f1_ld = f1_initial
history_ld = []

for iteration in range(50):
    iter_start = time.time()
    
    # Recompute in longdouble
    residuals_sec = compute_residuals_longdouble(dt_sec, f0_ld, f1_ld, weights)
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
    
    iter_time = time.time() - iter_start
    
    history_ld.append({
        'iteration': iteration + 1,
        'rms': rms,
        'max_delta': max_delta,
        'time': iter_time
    })

    if iteration < 3 or max_delta < 1e-14:
        print(f"  Iter {iteration+1:2d}: RMS={rms:.6f} μs, max|Δ|={max_delta:.2e} (time={iter_time*1000:.1f} ms)")

    if max_delta < 1e-20:
        print(f"  → CONVERGED at iteration {iteration+1}")
        break

total_time_ld = time.time() - t0
residuals_baseline = residuals_sec

print()
print(f"Final (BASELINE):")
print(f"  F0 = {f0_ld:.20f} Hz")
print(f"  F1 = {f1_ld:.25e} Hz/s")
print(f"  Total time: {total_time_ld*1000:.1f} ms")
print()

#------------------------------------------------------------------------------
# COMPARISON
#------------------------------------------------------------------------------

print("="*80)
print("COMPARISON")
print("="*80)
print()

# Precision
diff_ns = (residuals_jax_ld_final - residuals_baseline) * 1e9
early_idx = tdb_mjd < np.percentile(tdb_mjd, 25)
late_idx = tdb_mjd > np.percentile(tdb_mjd, 75)
drift = np.mean(diff_ns[late_idx]) - np.mean(diff_ns[early_idx])

print("Precision (JAX vs Longdouble baseline):")
print(f"  RMS:   {np.std(diff_ns):.6f} ns")
print(f"  Max:   {np.max(np.abs(diff_ns)):.6f} ns")
print(f"  Drift: {drift:.6f} ns")
print()

# Speed
# Exclude first iteration (JIT compilation)
jax_iter_times = [h['time'] for h in history_jax[1:]]
ld_iter_times = [h['time'] for h in history_ld]

print("Speed comparison:")
print(f"  JAX iterations (avg):      {np.mean(jax_iter_times)*1000:.2f} ms")
print(f"  Longdouble iterations (avg): {np.mean(ld_iter_times)*1000:.2f} ms")
print(f"  Speedup per iteration:     {np.mean(ld_iter_times) / np.mean(jax_iter_times):.1f}×")
print()
print(f"  JAX total time:      {total_time_jax*1000:.1f} ms")
print(f"  Longdouble total time: {total_time_ld*1000:.1f} ms")
print(f"  Total speedup:       {total_time_ld / total_time_jax:.1f}×")
print()

#------------------------------------------------------------------------------
# TEST AUTODIFF
#------------------------------------------------------------------------------

print("="*80)
print("AUTODIFF TEST")
print("="*80)
print()

print("Testing automatic differentiation of residuals w.r.t. F0 and F1...")

def compute_residuals_jax(dt_sec, f0, f1, weights):
    """Compute residuals in JAX (for autodiff)."""
    phase = dt_sec * (f0 + dt_sec * (f1 / 2.0))
    phase_wrapped = phase - jnp.round(phase)
    residuals = phase_wrapped / f0
    
    # Zero weighted mean
    weighted_mean = jnp.sum(residuals * weights) / jnp.sum(weights)
    return residuals - weighted_mean

# Compute derivatives at final parameters
f0_test = float(f0_jax)
f1_test = float(f1_jax)

# Autodiff: d(residuals)/d(f0) and d(residuals)/d(f1)
grad_f0_func = jit(grad(lambda f0: jnp.sum(compute_residuals_jax(dt_jax, f0, f1_test, weights_jax)**2)))
grad_f1_func = jit(grad(lambda f1: jnp.sum(compute_residuals_jax(dt_jax, f0_test, f1, weights_jax)**2)))

# Compute gradients
t0 = time.time()
grad_f0 = grad_f0_func(f0_test)
grad_f1 = grad_f1_func(f1_test)
grad_f0.block_until_ready()
t_grad = time.time() - t0

print(f"  ✓ Autodiff successful! (computed in {t_grad*1000:.2f} ms)")
print(f"    ∂(χ²)/∂F0 = {float(grad_f0):.6e}")
print(f"    ∂(χ²)/∂F1 = {float(grad_f1):.6e}")
print()
print("  This proves JAX can automatically compute derivatives!")
print("  → No need for manual analytical derivative implementation")
print()

#------------------------------------------------------------------------------
# PLOTTING
#------------------------------------------------------------------------------

print("Creating diagnostic plots...")

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Panel 1: Residual differences
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(tdb_mjd, diff_ns, 'b.', ms=2, alpha=0.6)
ax1.axhline(0, color='k', ls='--', lw=1)
ax1.set_ylabel('Difference (ns)', fontsize=11)
ax1.set_title(f'JAX Incremental vs Longdouble Baseline\nRMS={np.std(diff_ns):.4f} ns, Max={np.max(np.abs(diff_ns)):.4f} ns, Drift={drift:.4f} ns',
             fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Panel 2: Histogram
ax2 = fig.add_subplot(gs[1, 0])
ax2.hist(diff_ns, bins=50, color='blue', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', ls='--', lw=2, label='Zero')
ax2.axvline(np.mean(diff_ns), color='orange', ls='-', lw=2, label=f'Mean={np.mean(diff_ns):.4f} ns')
ax2.set_xlabel('Difference (ns)', fontsize=11)
ax2.set_ylabel('Count', fontsize=11)
ax2.set_title('Residual Difference Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Iteration times
ax3 = fig.add_subplot(gs[1, 1])
jax_iters = [h['iteration'] for h in history_jax]
jax_times = [h['time'] * 1000 for h in history_jax]
ld_iters = [h['iteration'] for h in history_ld[:len(history_jax)]]
ld_times = [h['time'] * 1000 for h in history_ld[:len(history_jax)]]

ax3.plot(jax_iters, jax_times, 'b.-', label=f'JAX (avg={np.mean(jax_iter_times)*1000:.1f} ms)', lw=2)
ax3.plot(ld_iters, ld_times, 'r.-', label=f'Longdouble (avg={np.mean(ld_iter_times)*1000:.1f} ms)', lw=2)
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Time (ms)', fontsize=11)
ax3.set_title('Iteration Time Comparison', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.annotate('First iter includes\nJIT compilation', xy=(1, jax_times[0]), xytext=(1.5, jax_times[0] + 5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1), fontsize=9)

# Panel 4: Convergence
ax4 = fig.add_subplot(gs[2, :])
jax_deltas = [h['max_delta'] for h in history_jax]
ld_deltas = [h['max_delta'] for h in history_ld[:len(history_jax)]]

ax4.semilogy(jax_iters, jax_deltas, 'b.-', label='JAX incremental', lw=2, ms=8)
ax4.semilogy(ld_iters, ld_deltas, 'r.-', label='Longdouble baseline', lw=2, ms=8)
ax4.axhline(1e-20, color='k', ls='--', lw=1, alpha=0.5, label='Convergence threshold')
ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('max|Δparams|', fontsize=11)
ax4.set_title('Convergence History', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.savefig('jax_incremental_diagnostics.png', dpi=150, bbox_inches='tight')
print("Saved: jax_incremental_diagnostics.png")
print()

#------------------------------------------------------------------------------
# SUMMARY
#------------------------------------------------------------------------------

print("="*80)
print("SUMMARY")
print("="*80)
print()

if np.max(np.abs(diff_ns)) < 0.1 and total_time_jax < total_time_ld:
    print("✓✓✓ COMPLETE SUCCESS! ✓✓✓")
    print()
    print("The JAX incremental fitter achieves:")
    print(f"  ✓ Perfect precision:    {np.std(diff_ns):.4f} ns RMS (matches longdouble!)")
    print(f"  ✓ No drift:             {drift:.4f} ns systematic drift")
    print(f"  ✓ Speed improvement:    {total_time_ld / total_time_jax:.1f}× faster than longdouble")
    print(f"  ✓ Per-iteration speed:  {np.mean(ld_iter_times) / np.mean(jax_iter_times):.1f}× faster iterations")
    print(f"  ✓ Autodiff capable:     Automatic derivatives (no manual implementation!)")
    print(f"  ✓ JIT compiled:         Optimized machine code")
    print()
    print("This is ready for production integration!")
    print()
    print("Next steps:")
    print("  1. Integrate into jug/fitting/optimized_fitter.py")
    print("  2. Test on multiple pulsars")
    print("  3. Benchmark full fitting convergence")
    print("  4. Update documentation")
else:
    print("⚠ Issues found:")
    if np.max(np.abs(diff_ns)) >= 0.1:
        print(f"  • Precision: {np.max(np.abs(diff_ns)):.3f} ns (target: <0.1 ns)")
    if total_time_jax >= total_time_ld:
        print(f"  • Speed: {total_time_jax/total_time_ld:.2f}× slower (target: faster)")
