#!/usr/bin/env python3
"""Trace where precision is lost in delay computation."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
from jug.residuals.simple_calculator import compute_residuals_simple

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get simple_calculator result
simple_result = compute_residuals_simple(par_file, tim_file)

print("Checking what simple_calculator stores:")
print(f"\n1. Total delay statistics:")
total_delay = simple_result['total_delay_sec']
print(f"   dtype: {total_delay.dtype}")
print(f"   mean: {np.mean(total_delay):.12f} s")
print(f"   min: {np.min(total_delay):.12f} s")
print(f"   max: {np.max(total_delay):.12f} s")

print(f"\n2. DM delay statistics:")
if 'dm_delay_sec' in simple_result:
    dm_delay = simple_result.get('dm_delay_sec', np.zeros(len(total_delay)))
    print(f"   dtype: {dm_delay.dtype}")
    print(f"   mean: {np.mean(dm_delay):.12f} s")
    print(f"   min: {np.min(dm_delay):.12f} s") 
    print(f"   max: {np.max(dm_delay):.12f} s")

print(f"\n3. dt_sec (emission time) statistics:")
dt_sec = simple_result['dt_sec']
print(f"   dtype: {dt_sec.dtype}")
print(f"   mean: {np.mean(dt_sec):.12f} s")
print(f"   min: {np.min(dt_sec):.12f} s")
print(f"   max: {np.max(dt_sec):.12f} s")

# Check what happens when we convert to float64
print(f"\n4. Precision loss from longdouble → float64:")

total_delay_f64 = np.array(total_delay, dtype=np.float64)
dt_sec_f64 = np.array(dt_sec, dtype=np.float64)

total_delay_loss = total_delay_f64 - total_delay
dt_sec_loss = dt_sec_f64 - dt_sec

print(f"\n   Total delay conversion error:")
print(f"   mean: {np.mean(total_delay_loss):.6e} s = {np.mean(total_delay_loss)*1e6:.6f} μs")
print(f"   max: {np.max(np.abs(total_delay_loss)):.6e} s = {np.max(np.abs(total_delay_loss))*1e6:.6f} μs")

print(f"\n   dt_sec conversion error:")
print(f"   mean: {np.mean(dt_sec_loss):.6e} s = {np.mean(dt_sec_loss)*1e6:.6f} μs")
print(f"   max: {np.max(np.abs(dt_sec_loss)):.6e} s = {np.max(np.abs(dt_sec_loss))*1e6:.6f} μs")

# The key question: if we use float64 dt_sec directly, does it match?
print(f"\n5. If we use float64 dt_sec from the start:")
print(f"   This is what compute_residuals_jax_from_dt() should use")
print(f"   Max precision loss: {np.max(np.abs(dt_sec_loss))*1e6:.6f} μs")

# But compute_residuals_jax RECOMPUTES dt_sec, which adds more error
print(f"\n6. Key insight:")
print(f"   - Converting pre-computed dt_sec to float64: {np.max(np.abs(dt_sec_loss))*1e6:.3f} μs max error")
print(f"   - Recomputing dt_sec from float64 delays: 3.994 μs max error (from earlier test)")
print(f"   - Difference: recomputing adds {(3.994 - np.max(np.abs(dt_sec_loss))*1e6):.3f} μs extra error!")
