#!/usr/bin/env python3
"""Systematically debug differences between baseline and JAX residual computation.

This script tests each step of the computation pipeline to isolate where the 0.33 μs
difference originates.
"""

# CRITICAL: Import jug first to enable float64
import jug

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax, dm_delay_jax, spin_phase_jax
from jug.utils.constants import SECS_PER_DAY
import math

print("="*60)
print("Systematic Debugging: Baseline vs JAX")
print("="*60)

# Get baseline
print("\n1. Computing baseline (simple_calculator)...")
result_baseline = compute_residuals_simple(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim'
)

# Get JAX
print("\n2. Computing JAX version...")
fixed_data = prepare_fixed_data(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim'
)

par_params = fixed_data['par_params']
param_names = ('F0', 'F1')
params_array = jnp.array([par_params['F0'], par_params['F1']])
fixed_params = {k: v for k, v in par_params.items() if k not in param_names}

residuals_jax_sec = compute_residuals_jax(
    params_array, param_names,
    fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
    fixed_data['geometric_delay_sec'], fixed_data['other_delays_minus_dm_sec'],
    fixed_data['pepoch'], fixed_data['dm_epoch'],
    fixed_data['tzr_phase'], fixed_data['uncertainties_us'],
    fixed_params
)

residuals_baseline_us = result_baseline['residuals_us']
residuals_jax_us = np.array(residuals_jax_sec) * 1e6

print("\n" + "="*60)
print("Overall Difference:")
print("="*60)
diff_us = residuals_jax_us - residuals_baseline_us
print(f"RMS: {np.sqrt(np.mean(diff_us**2)):.6f} μs")
print(f"Mean: {np.mean(diff_us):.6f} μs")
print(f"Std: {np.std(diff_us):.6f} μs")

# Now test each component step by step
print("\n" + "="*60)
print("Component-by-Component Testing:")
print("="*60)

# Test 1: TDB times
print("\n[Test 1] TDB times match?")
tdb_baseline = result_baseline['tdb_mjd']
tdb_jax = np.array(fixed_data['tdb_mjd'])
tdb_diff = tdb_jax - tdb_baseline
print(f"  Max difference: {np.max(np.abs(tdb_diff))} MJD")
print(f"  Status: {'✅ PASS' if np.max(np.abs(tdb_diff)) < 1e-10 else '❌ FAIL'}")

# Test 2: Frequencies
print("\n[Test 2] Barycentric frequencies match?")
freq_baseline = result_baseline['freq_bary_mhz']
freq_jax = np.array(fixed_data['freq_mhz'])
freq_diff = freq_jax - freq_baseline
print(f"  Max difference: {np.max(np.abs(freq_diff)):.9f} MHz")
print(f"  Status: {'✅ PASS' if np.max(np.abs(freq_diff)) < 1e-6 else '❌ FAIL'}")

# Test 3: Total delays
print("\n[Test 3] Total delays match?")
delay_baseline = result_baseline['total_delay_sec']
# In JAX: total_delay_minus_dm + dm_new should equal total_delay
# We stored total_delay_minus_dm in 'geometric_delay_sec'
delay_minus_dm_jax = np.array(fixed_data['geometric_delay_sec'])

# Compute DM delay at reference
dm_coeffs = []
k = 0
while True:
    key = 'DM' if k == 0 else f'DM{k}'
    if key in par_params:
        dm_coeffs.append(float(par_params[key]))
        k += 1
    else:
        break
dm_coeffs_jax = jnp.array(dm_coeffs if dm_coeffs else [0.0])
dm_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(dm_coeffs))])
dm_epoch_jax = float(par_params.get('DMEPOCH', par_params['PEPOCH']))

dm_delay_jax_computed = np.array(dm_delay_jax(
    jnp.array(freq_baseline), dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax, jnp.array(tdb_baseline)
))

total_delay_jax = delay_minus_dm_jax + dm_delay_jax_computed
delay_diff = total_delay_jax - delay_baseline
print(f"  Max difference: {np.max(np.abs(delay_diff)):.9f} s")
print(f"  Mean difference: {np.mean(delay_diff):.9f} s")
print(f"  This is {np.mean(delay_diff)*1e6:.6f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(delay_diff)) < 1e-9 else '❌ FAIL'}")

# Test 4: Emission times
print("\n[Test 4] Emission times (TDB - delay)?")
F0 = float(par_params['F0'])
F1 = float(par_params['F1'])
F2 = float(par_params.get('F2', 0.0))
PEPOCH = float(par_params['PEPOCH'])

# Baseline emission times
t_em_baseline_sec = (tdb_baseline - PEPOCH) * SECS_PER_DAY - delay_baseline
# JAX emission times
t_em_jax_sec = (tdb_jax - PEPOCH) * SECS_PER_DAY - total_delay_jax
t_em_diff = t_em_jax_sec - t_em_baseline_sec
print(f"  Max difference: {np.max(np.abs(t_em_diff)):.9f} s")
print(f"  This is {np.max(np.abs(t_em_diff))*1e6:.6f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(t_em_diff)) < 1e-9 else '❌ FAIL'}")

# Test 5: Phase computation
print("\n[Test 5] Spin phase computation?")
# Baseline uses: F0 * dt + F1/2 * dt^2 + F2/6 * dt^3
F1_half = F1 / 2.0
F2_sixth = F2 / 6.0
phase_baseline = F0 * t_em_baseline_sec + F1_half * t_em_baseline_sec**2 + F2_sixth * t_em_baseline_sec**3

# JAX uses spin_phase_jax
phase_jax = np.array(spin_phase_jax(
    jnp.array(t_em_jax_sec), F0, F1, F2, 0.0
))

phase_diff = phase_jax - phase_baseline
print(f"  Max difference: {np.max(np.abs(phase_diff)):.9f} cycles")
print(f"  This is {np.max(np.abs(phase_diff))/F0*1e6:.6f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(phase_diff)) < 1e-6 else '❌ FAIL'}")

# Test 6: Phase wrapping
print("\n[Test 6] Phase wrapping (mod operation)?")
tzr_phase = result_baseline['tzr_phase']
# Baseline: mod(phase - tzr + 0.5, 1.0) - 0.5
frac_phase_baseline = np.mod(phase_baseline - tzr_phase + 0.5, 1.0) - 0.5
# JAX: same formula
frac_phase_jax = np.mod(phase_jax - tzr_phase + 0.5, 1.0) - 0.5

frac_diff = frac_phase_jax - frac_phase_baseline
print(f"  Max difference: {np.max(np.abs(frac_diff)):.9f} cycles")
print(f"  This is {np.max(np.abs(frac_diff))/F0*1e6:.6f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(frac_diff)) < 1e-6 else '❌ FAIL'}")

# Test 7: Conversion to microseconds
print("\n[Test 7] Conversion to microseconds?")
residuals_baseline_before_wmean = frac_phase_baseline / F0 * 1e6
residuals_jax_before_wmean = frac_phase_jax / F0 * 1e6
res_diff_before_wmean = residuals_jax_before_wmean - residuals_baseline_before_wmean
print(f"  Max difference: {np.max(np.abs(res_diff_before_wmean)):.9f} μs")
print(f"  Mean difference: {np.mean(res_diff_before_wmean):.9f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(res_diff_before_wmean)) < 0.001 else '❌ FAIL'}")

# Test 8: Weighted mean subtraction
print("\n[Test 8] Weighted mean computation?")
errors_us = result_baseline['errors_us']
weights = 1.0 / (errors_us ** 2)

# Baseline weighted mean
wmean_baseline = np.sum(residuals_baseline_before_wmean * weights) / np.sum(weights)
residuals_baseline_after_wmean = residuals_baseline_before_wmean - wmean_baseline

# JAX weighted mean
wmean_jax = np.sum(residuals_jax_before_wmean * weights) / np.sum(weights)
residuals_jax_after_wmean = residuals_jax_before_wmean - wmean_jax

print(f"  Baseline weighted mean: {wmean_baseline:.9f} μs")
print(f"  JAX weighted mean: {wmean_jax:.9f} μs")
print(f"  Difference: {wmean_jax - wmean_baseline:.9f} μs")

# Final difference
final_diff = residuals_jax_after_wmean - residuals_baseline_after_wmean
print(f"\n[Test 9] Final residuals after weighted mean?")
print(f"  Max difference: {np.max(np.abs(final_diff)):.9f} μs")
print(f"  Mean difference: {np.mean(final_diff):.9f} μs")
print(f"  Std difference: {np.std(final_diff):.9f} μs")
print(f"  Status: {'✅ PASS' if np.max(np.abs(final_diff)) < 0.001 else '❌ FAIL'}")

# Compare to actual output
print(f"\n[Test 10] Compare to actual JAX output?")
actual_diff = residuals_jax_us - residuals_baseline_us
print(f"  Actual difference RMS: {np.sqrt(np.mean(actual_diff**2)):.9f} μs")
print(f"  Computed difference RMS: {np.sqrt(np.mean(final_diff**2)):.9f} μs")
print(f"  Match: {'✅ YES' if abs(np.sqrt(np.mean(actual_diff**2)) - np.sqrt(np.mean(final_diff**2))) < 0.001 else '❌ NO'}")

print("\n" + "="*60)
print("Summary:")
print("="*60)
print("If all tests pass but final difference is 0.33 μs, the issue is in")
print("the JAX kernel itself (numerical precision in JIT compilation).")
print("If a test fails, that's where the bug is!")
