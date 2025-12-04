#!/usr/bin/env python3
"""Debug what prepare_fixed_data() actually stores."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

print("Comparing simple_calculator vs prepare_fixed_data...")

# Get baseline
result = compute_residuals_simple(par_file, tim_file)
print(f"\nBaseline (simple_calculator):")
print(f"  Total delay mean: {np.mean(result['total_delay_sec']):.6f} s")
print(f"  Weighted RMS: {result['rms_us']:.3f} μs")

# Get fixed data
fixed_data = prepare_fixed_data(par_file, tim_file)

print(f"\nFixed data:")
print(f"  geometric_delay_sec mean: {np.mean(np.array(fixed_data['geometric_delay_sec'])):.6f} s")
print(f"  other_delays_minus_dm_sec mean: {np.mean(np.array(fixed_data['other_delays_minus_dm_sec'])):.6f} s")

# Check if they match
geometric = np.array(fixed_data['geometric_delay_sec'])
other = np.array(fixed_data['other_delays_minus_dm_sec'])
total_from_fixed = geometric + other

print(f"\n  Total (geometric + other): {np.mean(total_from_fixed):.6f} s")
print(f"  Match baseline: {np.allclose(total_from_fixed, result['total_delay_sec'] - result.get('dm_delay_sec', 0))}")

# Check DM delay
par_params = fixed_data['par_params']
dm_ref = par_params['DM']

from jug.residuals.core import dm_delay_jax
import jax.numpy as jnp
import math

dm_coeffs = jnp.array([dm_ref])
dm_factorials = jnp.array([1.0])
dm_epoch = fixed_data['dm_epoch']
tdb_mjd = fixed_data['tdb_mjd']
freq_mhz = fixed_data['freq_mhz']

dm_delay_computed = dm_delay_jax(freq_mhz, dm_coeffs, dm_factorials, dm_epoch, tdb_mjd)

print(f"\nDM delay:")
print(f"  Computed by dm_delay_jax: {np.mean(np.array(dm_delay_computed)):.6f} s")
if 'dm_delay_sec' in result:
    print(f"  From simple_calculator: {np.mean(result['dm_delay_sec']):.6f} s")
    print(f"  Match: {np.allclose(dm_delay_computed, result['dm_delay_sec'])}")

# What should total_delay be?
print(f"\nWhat total_delay should be:")
print(f"  From result: {np.mean(result['total_delay_sec']):.6f} s")
print(f"  geometric + other + dm_computed: {np.mean(total_from_fixed + np.array(dm_delay_computed)):.6f} s")

total_reconstructed = total_from_fixed + np.array(dm_delay_computed)
print(f"  Difference: {np.mean(total_reconstructed - result['total_delay_sec']):.6e} s")
print(f"  Max difference: {np.max(np.abs(total_reconstructed - result['total_delay_sec'])):.6e} s")
