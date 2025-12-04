#!/usr/bin/env python3
"""Diagnose why residual calculations are failing when parameters change."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt

print("="*80)
print("Diagnosing Residual Calculation Issues")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Step 1: Get baseline residuals using simple_calculator
print("\n1. Computing baseline residuals...")
result = compute_residuals_simple(par_file, tim_file)
print(f"   Baseline RMS (unweighted): {np.std(result['residuals_us']):.3f} μs")

# Step 2: Prepare fixed data for JAX
print("\n2. Preparing fixed data for JAX...")
fixed_data = prepare_fixed_data(par_file, tim_file)

# Step 3: Test residual function with EXACT same parameters
print("\n3. Testing JAX residuals with EXACT parameters from .par file...")
par_params = fixed_data['par_params']
f0_true = par_params['F0']
f1_true = par_params['F1']

# Create residual function
fit_params = ['F0', 'F1']
params_array = jnp.array([f0_true, f1_true])
fixed_params = {k: v for k, v in par_params.items() if k not in fit_params}

residuals_sec = compute_residuals_jax_from_dt(
    params_array,
    tuple(fit_params),
    fixed_data['dt_sec'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    fixed_params
)

residuals_us = np.array(residuals_sec) * 1e6
print(f"   JAX RMS (unweighted): {np.std(residuals_us):.3f} μs")
print(f"   JAX mean: {np.mean(residuals_us):.3f} μs")
print(f"   Should match baseline: {abs(np.std(residuals_us) - np.std(result['residuals_us'])) < 0.01}")

# Step 4: Test with tiny F0 perturbation
print("\n4. Testing with TINY F0 perturbation (1e-12 Hz)...")
f0_perturbed = f0_true + 1e-12  # Very small perturbation
params_array_pert = jnp.array([f0_perturbed, f1_true])

residuals_sec_pert = compute_residuals_jax_from_dt(
    params_array_pert,
    tuple(fit_params),
    fixed_data['dt_sec'],
    fixed_data['tzr_phase'],
    fixed_data['uncertainties_us'],
    fixed_params
)

residuals_us_pert = np.array(residuals_sec_pert) * 1e6
print(f"   Perturbed F0: {f0_perturbed:.15f} Hz (Δ = {1e-12:.1e})")
print(f"   RMS: {np.std(residuals_us_pert):.3f} μs")
print(f"   Mean: {np.mean(residuals_us_pert):.3f} μs")
print(f"   Change in RMS: {np.std(residuals_us_pert) - np.std(residuals_us):.3f} μs")

# Step 5: Check what dt_sec values look like
print("\n5. Checking dt_sec (emission times) statistics...")
dt_sec = np.array(fixed_data['dt_sec'])
print(f"   Min dt_sec: {dt_sec.min():.3f} s")
print(f"   Max dt_sec: {dt_sec.max():.3f} s")
print(f"   Mean dt_sec: {dt_sec.mean():.3f} s")
print(f"   Span: {dt_sec.max() - dt_sec.min():.3f} s")

# Step 6: Compute expected phase change from perturbation
print("\n6. Computing expected phase change from perturbation...")
delta_f0 = 1e-12
expected_phase_change = dt_sec * delta_f0
print(f"   Expected phase change (min): {expected_phase_change.min():.6e} cycles")
print(f"   Expected phase change (max): {expected_phase_change.max():.6e} cycles")
print(f"   Expected time residual change: ~{expected_phase_change.max() / f0_true * 1e6:.3f} μs")

# Step 7: Test if residuals include mean offset
print("\n7. Checking if residuals include mean offset...")
print(f"   JAX residuals mean: {np.mean(residuals_us):.3f} μs")
print(f"   Baseline residuals mean: {np.mean(result['residuals_us']):.3f} μs")
print(f"   Note: _compute_residuals_from_dt does NOT subtract mean")
print(f"   This is expected behavior")

# Step 8: Test weighted mean subtraction
print("\n8. Testing weighted mean subtraction...")
errors_us = np.array(fixed_data['uncertainties_us'])
weights = 1.0 / (errors_us ** 2)
weighted_mean = np.sum(weights * residuals_us) / np.sum(weights)
residuals_us_wmean_sub = residuals_us - weighted_mean

print(f"   Weighted mean: {weighted_mean:.3f} μs")
print(f"   RMS after subtracting weighted mean: {np.std(residuals_us_wmean_sub):.3f} μs")
print(f"   Baseline RMS (weighted): {result['rms_us']:.3f} μs")

print("\n" + "="*80)
print("Diagnosis Complete")
print("="*80)
