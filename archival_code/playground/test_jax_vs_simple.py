#!/usr/bin/env python3
"""Compare compute_residuals_jax vs simple_calculator"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

print("Comparing compute_residuals_jax vs simple_calculator")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get baseline
simple_result = compute_residuals_simple(par_file, tim_file)
print(f"\n1. simple_calculator:")
print(f"   WRMS: {simple_result['rms_us']:.6f} μs")
print(f"   Mean: {np.mean(simple_result['residuals_us']):.6f} μs")

# Prepare for JAX
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

# Use exact same parameters
f0 = par_params['F0']
f1 = par_params['F1']
dm = par_params['DM']

print(f"\n2. Parameters being used:")
print(f"   F0: {f0:.15f} Hz")
print(f"   F1: {f1:.15e} Hz/s")
print(f"   DM: {dm:.15f} pc/cm^3")

# Compute with JAX
params_array = jnp.array([float(dm), float(f0), float(f1)], dtype=jnp.float64)
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

errors_us = np.array(fixed_data['uncertainties_us'])
weights = 1.0 / (errors_us ** 2)
jax_wrms = np.sqrt(np.sum(weights * jax_res_us**2) / np.sum(weights))

print(f"\n3. compute_residuals_jax:")
print(f"   WRMS: {jax_wrms:.6f} μs")
print(f"   Mean: {np.mean(jax_res_us):.6f} μs")

# Compare
diff = jax_res_us - simple_result['residuals_us']

print(f"\n4. Difference (JAX - simple):")
print(f"   Mean: {np.mean(diff):.6f} μs")
print(f"   Std: {np.std(diff):.6f} μs")
print(f"   Min: {np.min(diff):.6f} μs")
print(f"   Max: {np.max(diff):.6f} μs")
print(f"   Max abs: {np.max(np.abs(diff)):.6f} μs")

if np.max(np.abs(diff)) > 0.01:
    print(f"\n❌ JAX differs from simple_calculator by {np.max(np.abs(diff)):.3f} μs")
    print("   This is the problem!")
else:
    print(f"\n✅ JAX matches simple_calculator to <0.01 μs")
