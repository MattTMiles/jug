#!/usr/bin/env python3
"""Check if weighted mean subtraction is the issue."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get both results
simple_result = compute_residuals_simple(par_file, tim_file)
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

# Compute JAX residuals
params_array = jnp.array([
    float(par_params['DM']),
    float(par_params['F0']),
    float(par_params['F1'])
], dtype=jnp.float64)
fit_params = ('DM', 'F0', 'F1')
fixed_params = {k: v for k, v in par_params.items() if k not in fit_params}

jax_res_sec = compute_residuals_jax(
    params_array, fit_params,
    fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
    fixed_data['geometric_delay_sec'],
    fixed_data['other_delays_minus_dm_sec'],
    fixed_data['pepoch'], fixed_data['dm_epoch'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    fixed_params
)
jax_res_us = np.array(jax_res_sec) * 1e6

simple_res_us = simple_result['residuals_us']
errors_us = np.array(fixed_data['uncertainties_us'])
weights = 1.0 / (errors_us ** 2)

# Check weighted means
simple_wmean = np.sum(weights * simple_res_us) / np.sum(weights)
jax_wmean = np.sum(weights * jax_res_us) / np.sum(weights)

print("Weighted mean check:")
print(f"  simple_calculator: {simple_wmean:.6e} μs")
print(f"  compute_residuals_jax: {jax_wmean:.6e} μs")

print(f"\nBoth should be ~0 if weighted mean was subtracted")

# Try adding back weighted mean to JAX
jax_res_no_wmean = jax_res_us + jax_wmean

diff_with_wmean = jax_res_no_wmean - simple_res_us

print(f"\nDifference after adding back weighted mean:")
print(f"  Mean: {np.mean(diff_with_wmean):.6f} μs")
print(f"  Std: {np.std(diff_with_wmean):.6f} μs")
print(f"  Max abs: {np.max(np.abs(diff_with_wmean)):.6f} μs")
