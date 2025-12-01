#!/usr/bin/env python3
"""Compare the actual arithmetic in simple_calculator vs compute_residuals_jax."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data
from jug.utils.constants import SECS_PER_DAY

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

simple_result = compute_residuals_simple(par_file, tim_file)
fixed_data = prepare_fixed_data(par_file, tim_file)

# What simple_calculator computed
tdb_mjd_simple = simple_result['tdb_mjd']
total_delay_simple = simple_result['total_delay_sec']
dt_sec_simple = simple_result['dt_sec']
pepoch = fixed_data['pepoch']

# Verify simple's arithmetic
tdb_sec = tdb_mjd_simple * SECS_PER_DAY
pepoch_sec = pepoch * SECS_PER_DAY
dt_sec_check = tdb_sec - pepoch_sec - total_delay_simple

print("Checking simple_calculator's arithmetic:")
print(f"dt_sec from simple: {dt_sec_simple[:5]}")
print(f"dt_sec recomputed:  {dt_sec_check[:5]}")
print(f"Difference: {np.max(np.abs(dt_sec_simple - dt_sec_check))*1e6:.6f} μs")

# What prepare_fixed_data stored
geometric_stored = np.array(fixed_data['geometric_delay_sec'])
other_stored = np.array(fixed_data['other_delays_minus_dm_sec'])

print(f"\nWhat prepare_fixed_data stored:")
print(f"geometric_delay_sec mean: {np.mean(geometric_stored):.6f} s")
print(f"other_delays_minus_dm_sec mean: {np.mean(other_stored):.6f} s")
print(f"Sum: {np.mean(geometric_stored + other_stored):.6f} s")

print(f"\nWhat simple_calculator had:")
# Need to check if simple stores the delay breakdown
if 'dm_delay_sec' in simple_result:
    dm_delay_simple = simple_result['dm_delay_sec']
    total_minus_dm = total_delay_simple - dm_delay_simple
    print(f"total_delay - dm_delay mean: {np.mean(total_minus_dm):.6f} s")
    print(f"Stored geometric should equal this")
    print(f"Difference: {np.max(np.abs(geometric_stored - total_minus_dm))*1e6:.6f} μs")
else:
    print("simple_calculator doesn't return dm_delay_sec separately")

# The issue: when compute_residuals_jax recomputes, it does:
# dt_sec = tdb_sec - pepoch_sec - (geometric + other + dm_new)
# But the stored geometric already had float64 rounding applied
print(f"\nKey question: Is the stored geometric_delay_sec accurate?")
print(f"Or did we lose precision when storing total_delay - dm_delay?")
