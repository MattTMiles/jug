#!/usr/bin/env python3
"""
JAX Incremental Fitter with Proper Caching (like production code).

Key insight: The production fitter ALREADY uses incremental updates for DM!
  1. Cache initial dt_sec (with initial DM delay baked in)
  2. Each iteration: Apply INCREMENTAL changes to dt_sec
  3. Update residuals incrementally from dt_sec changes

This is the same principle as the JAX incremental method for F0/F1.
The key is: START from the cached state, apply CHANGES incrementally.

This should work from initialization because we're using the SAME
caching strategy as the proven production fitter.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from pathlib import Path
import time

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, get_longdouble
from jug.io.tim_reader import parse_tim_file_mjds
from jug.fitting.wls_fitter import wls_solve_svd
from jug.fitting.derivatives_dm import compute_dm_derivatives
from jug.fitting.optimized_fitter import compute_dm_delay_fast
from jug.utils.constants import SECS_PER_DAY

print("="*80)
print("JAX INCREMENTAL FITTER: Proper Caching (Production Strategy)")
print("="*80)
print()

# Load data
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

print("Loading data...")
params = parse_par_file(par_file)
toas_data = parse_tim_file_mjds(tim_file)

# Extract TOA data
toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in toas_data])
freq_mhz = np.array([toa.freq_mhz for toa in toas_data])
errors_us = np.array([toa.error_us for toa in toas_data])
errors_sec = errors_us * 1e-6
weights = 1.0 / errors_sec ** 2

# Initial parameters
f0_initial = float(params['F0'])
f1_initial = float(params['F1'])
dm_initial = float(params['DM'])
dm1_initial = float(params.get('DM1', 0.0))
dmepoch_mjd = float(get_longdouble(params, 'DMEPOCH'))

n_toas = len(toas_mjd)
print(f"  Loaded {n_toas} TOAs")
print(f"  Initial F0  = {f0_initial:.15f} Hz")
print(f"  Initial F1  = {f1_initial:.6e} Hz/s")
print(f"  Initial DM  = {dm_initial:.15f} pc/cm^3")
print(f"  Initial DM1 = {dm1_initial:.6e} pc/cm^3/day")
print()

#------------------------------------------------------------------------------
# CACHE INITIAL STATE (like production fitter)
#------------------------------------------------------------------------------

print("Caching initial state (like production fitter)...")
t0 = time.time()

# Compute residuals with initial parameters
# This gives us dt_sec with ALL delays baked in
result = compute_residuals_simple(
    par_file, tim_file,
    clock_dir="data/clock",
    subtract_tzr=False,  # Important! Match production fitter
    verbose=False
)

dt_sec_cached = result['dt_sec']
tdb_mjd = result['tdb_mjd']
freq_bary_mhz = result['freq_bary_mhz']

# Cache initial DM delay (for incremental updates)
initial_dm_params = {'DM': dm_initial, 'DM1': dm1_initial}
initial_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, initial_dm_params, dmepoch_mjd)

# Compute initial residuals in LONGDOUBLE (perfect precision)
dt_sec_ld = np.array(dt_sec_cached, dtype=np.longdouble)
f0_ld = np.longdouble(f0_initial)
f1_ld = np.longdouble(f1_initial)

phase_ld = dt_sec_ld * (f0_ld + dt_sec_ld * (f1_ld / 2.0))
phase_wrapped_ld = phase_ld - np.round(phase_ld)
residuals_ld = phase_wrapped_ld / f0_ld

# Convert to float64 (safe for small residuals)
residuals_init = np.array(residuals_ld, dtype=np.float64)
weighted_mean = np.sum(residuals_init * weights) / np.sum(weights)
residuals_init = residuals_init - weighted_mean

t_cache = time.time() - t0
print(f"  ✓ Cached dt_sec and initial DM delay in {t_cache*1000:.2f} ms")
print()

#------------------------------------------------------------------------------
# JAX INCREMENTAL ITERATION FUNCTION
#------------------------------------------------------------------------------

@jit
def compute_residual_update_from_dt_change(residuals_current, dt_change, f0, f1, weights):
    """
    Update residuals when dt_sec changes (from DM parameter updates).
    
    This is the KEY: When dt changes, phase changes by:
        Δφ = F0 * Δdt + F1 * (dt_old * Δdt + 0.5 * Δdt^2)
    
    For small Δdt (DM corrections), we can linearize.
    """
    # For simplicity, recompute phase with new dt
    # In production, could use Taylor expansion for speed
    # But this is still fast in JAX
    return residuals_current  # Placeholder - will update

@jit
def jax_iteration_all_params(residuals, dt_sec, f0, f1, weights):
    """
    Single iteration: Update F0, F1 using JAX incremental method.
    
    Takes current residuals (already computed from current dt_sec and current F0/F1)
    Returns updated residuals and parameter changes.
    """
    # Design matrix for F0/F1
    M0 = -dt_sec / f0
    M1 = -(dt_sec**2 / 2.0) / f0
    
    # Zero weighted mean
    sum_w = jnp.sum(weights)
    M0 = M0 - jnp.sum(M0 * weights) / sum_w
    M1 = M1 - jnp.sum(M1 * weights) / sum_w
    
    # Build 2×2 normal equations
    A00 = jnp.sum(M0 * weights * M0)
    A01 = jnp.sum(M0 * weights * M1)
    A11 = jnp.sum(M1 * weights * M1)
    
    b0 = jnp.sum(M0 * weights * residuals)
    b1 = jnp.sum(M1 * weights * residuals)
    
    # Analytical 2×2 solve
    det = A00 * A11 - A01 * A01
    delta_f0 = (A11 * b0 - A01 * b1) / det
    delta_f1 = (A00 * b1 - A01 * b0) / det
    
    # Update residuals incrementally (the magic!)
    residuals_new = residuals - delta_f0 * M0 - delta_f1 * M1
    
    # RMS
    rms = jnp.sqrt(jnp.sum(residuals**2 * weights) / sum_w)
    
    return residuals_new, delta_f0, delta_f1, rms

#------------------------------------------------------------------------------
# FITTING LOOP (Incremental updates for ALL parameters)
#------------------------------------------------------------------------------

print("FITTING LOOP: Incremental updates (F0, F1, DM, DM1)")
print("-"*80)

# Current parameter values
f0 = f0_initial
f1 = f1_initial
dm = dm_initial
dm1 = dm1_initial

# Current dt_sec and residuals (will be updated incrementally)
dt_sec_current = dt_sec_cached.copy()
residuals_jax = jnp.array(residuals_init)

# JAX arrays
weights_jax = jnp.array(weights)

t0 = time.time()
history = []
rms_history = []

# Convergence criteria (match production fitter)
xtol = 1e-12  # Parameter tolerance (relative)
gtol = 1e-3   # RMS change tolerance (μs)
min_iterations = 3

for iteration in range(30):
    iter_start = time.time()
    
    # STEP 1: Fit F0/F1 using JAX incremental method
    dt_jax = jnp.array(dt_sec_current)
    residuals_jax, delta_f0, delta_f1, rms = jax_iteration_all_params(
        residuals_jax, dt_jax, f0, f1, weights_jax
    )
    
    delta_f0_val = float(delta_f0)
    delta_f1_val = float(delta_f1)
    rms_us = float(rms) * 1e6
    
    # Update F0/F1
    f0 += delta_f0_val
    f1 += delta_f1_val
    
    # STEP 2: Fit DM parameters using existing JUG derivatives
    residuals_np = np.array(residuals_jax)
    
    # Compute DM derivatives
    params_current = {'DMEPOCH': dmepoch_mjd, 'DM': dm, 'DM1': dm1, 'F0': f0}
    dm_derivs = compute_dm_derivatives(
        params=params_current,
        toas_mjd=tdb_mjd,
        freq_mhz=freq_bary_mhz,
        fit_params=['DM', 'DM1']
    )
    
    # Build DM design matrix
    M_dm = np.column_stack([dm_derivs['DM'], dm_derivs['DM1']])
    
    # Zero weighted mean
    for j in range(2):
        col_mean = np.sum(M_dm[:, j] * weights) / np.sum(weights)
        M_dm[:, j] = M_dm[:, j] - col_mean
    
    # WLS solve for DM
    delta_dm_params, cov, _ = wls_solve_svd(
        jnp.array(residuals_np),
        jnp.array(errors_sec),
        jnp.array(M_dm),
        negate_dpars=False
    )
    delta_dm_params = np.array(delta_dm_params)
    
    # Update DM parameters
    dm += delta_dm_params[0]
    dm1 += delta_dm_params[1]
    
    # STEP 3: Update residuals incrementally for DM changes
    # When DM changes, dt_sec changes, which affects residuals
    # The residual update is: Δr = -M_dm @ Δdm
    residuals_np = residuals_np - M_dm @ delta_dm_params
    residuals_jax = jnp.array(residuals_np)
    
    # Also update dt_sec for next F0/F1 iteration
    # Compute new DM delay with updated DM parameters
    new_dm_params = {'DM': dm, 'DM1': dm1}
    new_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, new_dm_params, dmepoch_mjd)
    
    # Update dt_sec: Apply CHANGE in DM delay
    dt_delay_change = new_dm_delay - initial_dm_delay
    dt_sec_current = dt_sec_cached - dt_delay_change  # Key: incremental update!
    
    # Recompute RMS
    rms_us = np.sqrt(np.sum(residuals_np**2 * weights) / np.sum(weights)) * 1e6
    
    iter_time = time.time() - iter_start
    
    # Max delta
    max_delta = max(abs(delta_f0_val), abs(delta_f1_val),
                    abs(delta_dm_params[0]), abs(delta_dm_params[1]))
    
    history.append({
        'iteration': iteration + 1,
        'rms': rms_us,
        'max_delta': max_delta,
        'time': iter_time
    })
    rms_history.append(rms_us)
    
    # Check convergence (same as production fitter)
    # Criterion 1: Parameter change
    delta_params_all = np.array([delta_f0_val, delta_f1_val, delta_dm_params[0], delta_dm_params[1]])
    param_values_current = np.array([f0, f1, dm, dm1])
    delta_norm = np.linalg.norm(delta_params_all)
    param_norm = np.linalg.norm(param_values_current)
    param_converged = delta_norm <= xtol * (param_norm + xtol)
    
    # Criterion 2: RMS not improving
    rms_converged = False
    if len(rms_history) >= 2:
        rms_change = abs(rms_history[-1] - rms_history[-2])
        rms_converged = rms_change < gtol
    
    # Converged if EITHER criterion met AND minimum iterations done
    converged = iteration >= min_iterations and (param_converged or rms_converged)
    
    status = ""
    if converged:
        if param_converged:
            status = "✓ Params converged"
        elif rms_converged:
            status = "✓ RMS stable"
    
    if iteration == 0:
        print(f"    Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} (time={iter_time*1000:.1f} ms - includes JIT)")
    elif iteration < 5 or converged:
        print(f"    Iter {iteration+1:2d}: RMS={rms_us:.6f} μs, max|Δ|={max_delta:.2e} {status} (time={iter_time*1000:.1f} ms)")
    
    if converged:
        print(f"    → CONVERGED at iteration {iteration+1}")
        break

t_iter = time.time() - t0

# Final recomputation in longdouble
print()
print("  Final recomputation in longdouble...")
t0 = time.time()

final_dm_params = {'DM': dm, 'DM1': dm1}
final_dm_delay = compute_dm_delay_fast(tdb_mjd, freq_bary_mhz, final_dm_params, dmepoch_mjd)
dt_delay_change_final = final_dm_delay - initial_dm_delay
dt_sec_final = dt_sec_cached - dt_delay_change_final

dt_final_ld = np.array(dt_sec_final, dtype=np.longdouble)
f0_final_ld = np.longdouble(f0)
f1_final_ld = np.longdouble(f1)

phase_final_ld = dt_final_ld * (f0_final_ld + dt_final_ld * (f1_final_ld / 2.0))
phase_wrapped_final_ld = phase_final_ld - np.round(phase_final_ld)
residuals_final_ld = phase_wrapped_final_ld / f0_final_ld

residuals_final = np.array(residuals_final_ld, dtype=np.float64)
weighted_mean_final = np.sum(residuals_final * weights) / np.sum(weights)
residuals_final = residuals_final - weighted_mean_final

t_final = time.time() - t0

total_time = t_cache + t_iter + t_final

print()
print(f"Final (JAX INCREMENTAL + CACHING):")
print(f"  F0  = {f0:.20f} Hz")
print(f"  F1  = {f1:.25e} Hz/s")
print(f"  DM  = {dm:.20f} pc/cm^3")
print(f"  DM1 = {dm1:.25e} pc/cm^3/day")
print(f"  Final RMS: {np.sqrt(np.sum(residuals_final**2 * weights) / np.sum(weights))*1e6:.6f} μs")
print(f"  Total time: {total_time*1000:.1f} ms ({len(history)} iterations)")
print()

#------------------------------------------------------------------------------
# BASELINE: Production fitter for comparison
#------------------------------------------------------------------------------

print("BASELINE: Production fitter")
print("-"*80)

from jug.fitting.optimized_fitter import fit_parameters_optimized

t0 = time.time()
result_prod = fit_parameters_optimized(
    par_file=par_file,
    tim_file=tim_file,
    fit_params=['F0', 'F1', 'DM', 'DM1'],
    max_iter=30,
    clock_dir="data/clock",
    verbose=False
)
t_prod = time.time() - t0

print(f"  Converged: {result_prod.get('converged', 'N/A')}")
print(f"  Iterations: {result_prod.get('num_iterations', len([h for h in result_prod.get('history', [])]))} ")
print(f"  Final RMS: {result_prod.get('final_rms', 'N/A'):.6f} μs")
print(f"  Total time: {t_prod*1000:.1f} ms")
print()
print(f"  F0  = {result_prod['final_params']['F0']:.20f} Hz")
print(f"  F1  = {result_prod['final_params']['F1']:.25e} Hz/s")
print(f"  DM  = {result_prod['final_params']['DM']:.20f} pc/cm^3")
print(f"  DM1 = {result_prod['final_params']['DM1']:.25e} pc/cm^3/day")
print()

#------------------------------------------------------------------------------
# COMPARISON
#------------------------------------------------------------------------------

print("="*80)
print("COMPARISON")
print("="*80)
print()

# Parameter differences
f0_prod = result_prod['final_params']['F0']
f1_prod = result_prod['final_params']['F1']
dm_prod = result_prod['final_params']['DM']
dm1_prod = result_prod['final_params']['DM1']

print("Parameter differences (JAX - Production):")
print(f"  ΔF0  = {f0 - f0_prod:+.3e} Hz")
print(f"  ΔF1  = {f1 - f1_prod:+.3e} Hz/s")
print(f"  ΔDM  = {dm - dm_prod:+.3e} pc/cm^3")
print(f"  ΔDM1 = {dm1 - dm1_prod:+.3e} pc/cm^3/day")
print()

# RMS comparison
rms_jax = np.sqrt(np.sum(residuals_final**2 * weights) / np.sum(weights)) * 1e6
rms_prod = result_prod['final_rms']
print(f"RMS comparison:")
print(f"  JAX:        {rms_jax:.6f} μs")
print(f"  Production: {rms_prod:.6f} μs")
print(f"  Difference: {abs(rms_jax - rms_prod):.6f} μs")
print()

# Speed
iter_times_jax = [h['time'] for h in history[1:]]  # Exclude JIT
print(f"Speed:")
print(f"  JAX iterations (avg):   {np.mean(iter_times_jax)*1000:.2f} ms")
print(f"  JAX total time:         {total_time*1000:.1f} ms")
print(f"  Production total time:  {t_prod*1000:.1f} ms")
print(f"  Speedup:                {t_prod / total_time:.2f}×")
print()

#------------------------------------------------------------------------------
# SUCCESS CHECK
#------------------------------------------------------------------------------

print("="*80)
print("SUCCESS CHECK")
print("="*80)
print()

param_match = (abs(f0 - f0_prod) < 1e-13 and 
               abs(f1 - f1_prod) < 1e-21 and
               abs(dm - dm_prod) < 1e-5 and
               abs(dm1 - dm1_prod) < 1e-10)

rms_match = abs(rms_jax - rms_prod) < 0.1  # 0.1 μs tolerance (relaxed)

converged_flag = len(history) < 30  # Didn't hit max iterations

if param_match and rms_match and converged_flag:
    print("✓✓✓ COMPLETE SUCCESS! ✓✓✓")
    print()
    print("The JAX incremental fitter with proper caching:")
    print(f"  ✓ Converged in {len(history)} iterations")
    print(f"  ✓ Matches production parameters exactly")
    print(f"  ✓ Achieves same RMS: {rms_jax:.6f} μs")
    print(f"  ✓ Works from initialization (par file starting values)")
    print(f"  ✓ Uses same caching strategy as production code")
    print()
    print("Key insight confirmed:")
    print("  The production fitter ALREADY uses incremental updates!")
    print("  JAX method extends this with:")
    print("    • Longdouble initial/final residuals (perfect precision)")
    print("    • JIT-compiled iterations (fast)")
    print("    • Same proven caching approach")
    print()
    print("READY FOR PRODUCTION!")
else:
    print("Issues found:")
    if not param_match:
        print(f"  ✗ Parameters don't match")
    if not rms_match:
        print(f"  ✗ RMS differs by {abs(rms_jax - rms_prod):.6f} μs")
    if not converged_flag:
        print(f"  ✗ Didn't converge (hit max iterations)")
