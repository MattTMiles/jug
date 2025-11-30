#!/usr/bin/env python3
"""
Test script to debug and fix JUG residuals

Run this after executing the notebook up to cell 233
"""

import numpy as np
import jax.numpy as jnp

# Key diagnostic questions:
print("=" * 80)
print("DIAGNOSTIC CHECKS")
print("=" * 80)

# 1. Check if phase_offset_cycles is being used
print("\n1. Model configuration:")
print(f"   phase_ref_mjd: {model.phase_ref_mjd}")
print(f"   phase_offset_cycles: {model.phase_offset_cycles}")
print(f"   tref_mjd (PEPOCH): {model.tref_mjd}")
print(f"   F0: {model.f0} Hz")
print(f"   F1: {model.f1} Hz/s")

# 2. Check the times being used
print("\n2. Time arrays:")
print(f"   t_mjd (topocentric) range: {t_mjd.min():.6f} to {t_mjd.max():.6f}")
print(f"   t_em_mjd (emission) range: {t_em_mjd.min():.6f} to {t_em_mjd.max():.6f}")
if 't_inf_jax' in dir():
    print(f"   t_inf (infinite-freq) range: {float(t_inf_jax.min()):.6f} to {float(t_inf_jax.max()):.6f}")

# 3. Manually compute residual for first TOA
print("\n3. Manual residual calculation for first TOA:")
t_test = t_em_mjd[0]  # Use emission time
print(f"   t_em[0] = {t_test} MJD")

# Compute DM delay
from jax.numpy import atleast_1d
dt_years = (t_test - model.dm_epoch_mjd) / 365.25
coeffs = atleast_1d(model.dm_coeffs)
facts = atleast_1d(model.dm_factorials)
powers = jnp.arange(len(coeffs))
dm_eff = jnp.sum(coeffs * (dt_years ** powers) / facts)
dm_delay_sec = 4.148808e3 * dm_eff / (freq_mhz[0]**2)

print(f"   DM delay: {dm_delay_sec:.6f} sec")

# Compute infinite-frequency time
t_inf_test = t_test - dm_delay_sec / 86400.0
print(f"   t_inf[0] = {t_inf_test} MJD")

# Compute phase at this time
dt_sec = (t_inf_test - model.tref_mjd) * 86400.0
phase = model.f0 * dt_sec + 0.5 * model.f1 * dt_sec**2

# Compute reference phase
dt_ref = (model.phase_ref_mjd - model.tref_mjd) * 86400.0
phase_ref = model.f0 * dt_ref + 0.5 * model.f1 * dt_ref**2

# Phase difference
phase_diff = phase - phase_ref - model.phase_offset_cycles

print(f"   Phase at t_inf: {phase:.2f} cycles")
print(f"   Phase at ref: {phase_ref:.2f} cycles")
print(f"   Phase difference: {phase_diff:.6f} cycles")

# Fractional phase
frac = np.mod(phase_diff + 0.5, 1.0) - 0.5
print(f"   Fractional phase: {frac:.6f} cycles")

# Residual
res_sec = frac / model.f0
res_us = res_sec * 1e6
print(f"   Residual: {res_us:.3f} μs")

# 4. Compare with tempo2
if t2_res_us is not None:
    print(f"   Tempo2 residual[0]: {t2_res_us[0]:.3f} μs")
    print(f"   Difference: {res_us - t2_res_us[0]:.3f} μs")

# 5. Check for systematic offset
print("\n4. Checking for systematic offset in phase_offset_cycles:")
# If all residuals are offset by ~698 μs, that's ~0.237 cycles at F0=339.3 Hz
# 698e-6 * 339.3 = 0.237 cycles
# This is suspiciously close to 0.08740234375 (the phase_offset) * some factor

offset_cycles = 698e-6 * model.f0
print(f"   698 μs = {offset_cycles:.6f} cycles")
print(f"   phase_offset_cycles = {model.phase_offset_cycles:.6f} cycles")
print(f"   Ratio: {offset_cycles / model.phase_offset_cycles:.2f}")

print("\n" + "=" * 80)
print("POTENTIAL FIX")
print("=" * 80)

# The issue might be in how phase_offset_cycles is being used
# Try recalculating with NEGATIVE phase_offset instead of positive

print("\nTesting fix: Use NEGATIVE phase_offset_cycles")
phase_diff_fixed = phase - phase_ref + model.phase_offset_cycles  # NOTE: + instead of -
frac_fixed = np.mod(phase_diff_fixed + 0.5, 1.0) - 0.5
res_fixed_us = (frac_fixed / model.f0) * 1e6

print(f"Original residual[0]: {res_us:.3f} μs")
print(f"Fixed residual[0]: {res_fixed_us:.3f} μs")
if t2_res_us is not None:
    print(f"Tempo2 residual[0]: {t2_res_us[0]:.3f} μs")
    print(f"Fixed difference: {res_fixed_us - t2_res_us[0]:.3f} μs")
