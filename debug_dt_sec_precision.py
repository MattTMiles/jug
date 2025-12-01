#!/usr/bin/env python3  
"""Check dt_sec precision loss."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, dm_delay_jax
from jug.utils.constants import SECS_PER_DAY

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get data
simple_result = compute_residuals_simple(par_file, tim_file)
fixed_data = prepare_fixed_data(par_file, tim_file)

# simple_calculator's dt_sec (computed with longdouble)
dt_sec_simple = simple_result['dt_sec']  # Already in result

# Recompute dt_sec the way compute_residuals_jax does (float64)
tdb_mjd = fixed_data['tdb_mjd']
pepoch = fixed_data['pepoch']
geometric_delay = fixed_data['geometric_delay_sec']
other_delays = fixed_data['other_delays_minus_dm_sec']

# Recompute DM delay
dm_coeffs = jnp.array([float(fixed_data['par_params']['DM'])])
dm_factorials = jnp.array([1.0])
dm_epoch = fixed_data['dm_epoch']
freq_mhz = fixed_data['freq_mhz']

dm_delay_recomputed = dm_delay_jax(freq_mhz, dm_coeffs, dm_factorials, dm_epoch, tdb_mjd)

# Recompute total delay and dt_sec (float64 precision)
total_delay = geometric_delay + other_delays + dm_delay_recomputed
tdb_sec = tdb_mjd * SECS_PER_DAY
pepoch_sec = pepoch * SECS_PER_DAY
dt_sec_recomputed = tdb_sec - pepoch_sec - total_delay

# Compare
dt_diff = np.array(dt_sec_recomputed) - dt_sec_simple

print("dt_sec precision comparison:")
print(f"  simple_calculator (longdouble): mean={np.mean(dt_sec_simple):.6f} s")
print(f"  recomputed (float64): mean={np.mean(np.array(dt_sec_recomputed)):.6f} s")
print(f"\nDifference (recomputed - simple):")
print(f"  Mean: {np.mean(dt_diff):.6e} s = {np.mean(dt_diff)*1e6:.6f} μs")
print(f"  Std: {np.std(dt_diff):.6e} s = {np.std(dt_diff)*1e6:.6f} μs")
print(f"  Max abs: {np.max(np.abs(dt_diff)):.6e} s = {np.max(np.abs(dt_diff))*1e6:.6f} μs")

# This difference propagates to residuals
# Residual ≈ (dt_sec error) * F0 (since residual = phase / F0)
f0 = fixed_data['par_params']['F0']
expected_residual_error = np.max(np.abs(dt_diff)) * f0  # in cycles
expected_residual_error_time = expected_residual_error / f0  # back to seconds

print(f"\nExpected residual error from dt_sec precision loss:")
print(f"  {expected_residual_error_time*1e6:.6f} μs")
print(f"\nActual residual difference observed: 0.792 μs")
