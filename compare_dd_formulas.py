#!/usr/bin/env python3
"""Deep dive into PINT vs JUG residual calculation differences."""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
import pint.models
import pint.toa
import pint.residuals
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax

print("="*80)
print("Deep Dive: PINT vs JUG - Finding the Source of Difference")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Load both
pint_model = pint.models.get_model(par_file)
pint_toas = pint.toa.get_TOAs(tim_file, ephem='DE440', include_bipm=True, planets=True)
simple_result = compute_residuals_simple(par_file, tim_file)

print(f"\n1. Baseline comparison (simple_calculator):")
print(f"   WRMS: {simple_result['rms_us']:.3f} μs")
print(f"   Mean: {np.mean(simple_result['residuals_us']):.6f} μs")

# PINT residuals
pint_resids = pint.residuals.Residuals(pint_toas, pint_model)
pint_res_us = np.array(pint_resids.time_resids.to_value('us'))
pint_errors_us = np.array(pint_toas.get_errors().to_value('us'))

print(f"\n2. PINT residuals:")
print(f"   Mean: {np.mean(pint_res_us):.6f} μs")
print(f"   Std: {np.std(pint_res_us):.6f} μs")

# Check weighted mean
weights_pint = 1.0 / (pint_errors_us ** 2)
wmean_pint = np.sum(weights_pint * pint_res_us) / np.sum(weights_pint)
print(f"   Weighted mean: {wmean_pint:.6e} μs (should be ~0)")

# PINT WRMS and chi²
pint_wrms = np.sqrt(np.sum(weights_pint * pint_res_us**2) / np.sum(weights_pint))
pint_chi2 = np.sum(weights_pint * pint_res_us**2)
print(f"   WRMS: {pint_wrms:.3f} μs")
print(f"   Chi²: {pint_chi2:.2f}")

print(f"\n3. simple_calculator vs PINT:")
simple_vs_pint = simple_result['residuals_us'] - pint_res_us
print(f"   Difference mean: {np.mean(simple_vs_pint):.6f} μs")
print(f"   Difference std: {np.std(simple_vs_pint):.6f} μs")
print(f"   Difference max: {np.max(np.abs(simple_vs_pint)):.6f} μs")

print("\n" + "="*80)
print("Conclusion:")
print("="*80)

if np.max(np.abs(simple_vs_pint)) < 0.01:
    print("✅ simple_calculator matches PINT to <0.01 μs")
else:
    print(f"⚠️  simple_calculator differs from PINT by {np.max(np.abs(simple_vs_pint)):.3f} μs")
    print("   This is the source of the fitting difference!")
