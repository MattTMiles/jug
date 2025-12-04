#!/usr/bin/env python3
"""Diagnose why PINT and JUG get different fit results.

Check:
1. Do residuals match before fitting?
2. Do residuals match after fitting with SAME parameters?
3. Where does the difference come from?
"""

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
print("Diagnosing PINT vs JUG Residual Differences")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Load PINT
print("\n1. Loading data...")
pint_model = pint.models.get_model(par_file)
pint_toas = pint.toa.get_TOAs(tim_file, ephem='DE440', include_bipm=True, planets=True)

# Get JUG data
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

print(f"   PINT: {len(pint_toas)} TOAs")
print(f"   JUG: {fixed_data['n_toas']} TOAs")

# Create JUG residual function
def jug_residuals(f0, f1, dm):
    """Compute JUG residuals."""
    # Convert to float64 (PINT may return float128)
    params_array = jnp.array([float(dm), float(f0), float(f1)], dtype=jnp.float64)
    fit_params = ('DM', 'F0', 'F1')
    fixed_params = {k: v for k, v in par_params.items() if k not in fit_params}

    residuals_sec = compute_residuals_jax(
        params_array, fit_params,
        fixed_data['tdb_mjd'], fixed_data['freq_mhz'],
        fixed_data['geometric_delay_sec'],
        fixed_data['other_delays_minus_dm_sec'],
        fixed_data['pepoch'], fixed_data['dm_epoch'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )

    return np.array(residuals_sec) * 1e6  # Convert to μs

def pint_residuals(f0, f1, dm):
    """Compute PINT residuals."""
    pint_model.F0.value = f0
    pint_model.F1.value = f1
    pint_model.DM.value = dm

    resids = pint.residuals.Residuals(pint_toas, pint_model)
    return np.array(resids.time_resids.to_value('us'))

# Test 1: Initial parameters from .par file
print("\n" + "="*80)
print("Test 1: Residuals with initial .par file parameters")
print("="*80)

f0_init = pint_model.F0.value
f1_init = pint_model.F1.value
dm_init = pint_model.DM.value

print(f"\nParameters:")
print(f"   F0: {f0_init:.15f} Hz")
print(f"   F1: {f1_init:.15e} Hz/s")
print(f"   DM: {dm_init:.15f} pc/cm^3")

pint_res_init = pint_residuals(f0_init, f1_init, dm_init)
jug_res_init = jug_residuals(f0_init, f1_init, dm_init)

errors_us = np.array(fixed_data['uncertainties_us'])
weights = 1.0 / (errors_us ** 2)

pint_wrms_init = np.sqrt(np.sum(weights * pint_res_init**2) / np.sum(weights))
jug_wrms_init = np.sqrt(np.sum(weights * jug_res_init**2) / np.sum(weights))

print(f"\nPINT residuals:")
print(f"   Mean: {np.mean(pint_res_init):.3f} μs")
print(f"   RMS: {np.std(pint_res_init):.3f} μs")
print(f"   WRMS: {pint_wrms_init:.3f} μs")

print(f"\nJUG residuals:")
print(f"   Mean: {np.mean(jug_res_init):.3f} μs")
print(f"   RMS: {np.std(jug_res_init):.3f} μs")
print(f"   WRMS: {jug_wrms_init:.3f} μs")

# Direct comparison
residual_diff = jug_res_init - pint_res_init
print(f"\nDifference (JUG - PINT):")
print(f"   Mean: {np.mean(residual_diff):.3f} μs")
print(f"   RMS: {np.std(residual_diff):.3f} μs")
print(f"   Max abs: {np.max(np.abs(residual_diff)):.3f} μs")
print(f"   Match: {np.allclose(jug_res_init, pint_res_init, atol=0.01)}")

# Test 2: After fitting - use PINT's fitted parameters in both
print("\n" + "="*80)
print("Test 2: Residuals with PINT's fitted parameters")
print("="*80)

# Fit with PINT first
pint_model.F0.value = f0_init + 1e-10
pint_model.F1.value = f1_init * 1.001
pint_model.DM.value = dm_init * 1.0001

pint_model.free_params = ['F0', 'F1', 'DM']
import pint.fitter
pint_fitter = pint.fitter.WLSFitter(pint_toas, pint_model)
pint_fitter.fit_toas(maxiter=20)

f0_pint = pint_fitter.model.F0.value
f1_pint = pint_fitter.model.F1.value
dm_pint = pint_fitter.model.DM.value

print(f"\nPINT fitted parameters:")
print(f"   F0: {f0_pint:.15f} Hz")
print(f"   F1: {f1_pint:.15e} Hz/s")
print(f"   DM: {dm_pint:.15f} pc/cm^3")

# Compute residuals with these parameters
pint_res_fitted = pint_residuals(f0_pint, f1_pint, dm_pint)
jug_res_fitted = jug_residuals(f0_pint, f1_pint, dm_pint)

pint_wrms_fitted = np.sqrt(np.sum(weights * pint_res_fitted**2) / np.sum(weights))
jug_wrms_fitted = np.sqrt(np.sum(weights * jug_res_fitted**2) / np.sum(weights))

print(f"\nPINT residuals (with PINT params):")
print(f"   Mean: {np.mean(pint_res_fitted):.3f} μs")
print(f"   WRMS: {pint_wrms_fitted:.3f} μs")

print(f"\nJUG residuals (with PINT params):")
print(f"   Mean: {np.mean(jug_res_fitted):.3f} μs")
print(f"   WRMS: {jug_wrms_fitted:.3f} μs")

residual_diff_fitted = jug_res_fitted - pint_res_fitted
print(f"\nDifference (JUG - PINT):")
print(f"   Mean: {np.mean(residual_diff_fitted):.3f} μs")
print(f"   RMS: {np.std(residual_diff_fitted):.3f} μs")
print(f"   Max abs: {np.max(np.abs(residual_diff_fitted)):.3f} μs")

# Test 3: Check if difference is systematic
print("\n" + "="*80)
print("Test 3: Is the difference systematic?")
print("="*80)

print(f"\nDifference statistics:")
print(f"   Initial params difference: {np.mean(jug_res_init - pint_res_init):.3f} ± {np.std(jug_res_init - pint_res_init):.3f} μs")
print(f"   Fitted params difference: {np.mean(jug_res_fitted - pint_res_fitted):.3f} ± {np.std(jug_res_fitted - pint_res_fitted):.3f} μs")

# Check correlation
correlation = np.corrcoef(jug_res_init, pint_res_init)[0, 1]
print(f"\nCorrelation between JUG and PINT residuals: {correlation:.6f}")

if correlation > 0.99:
    print("   → High correlation - differences are likely due to systematic offset or scaling")
else:
    print("   → Low correlation - fundamental difference in residual calculation")

print("\n" + "="*80)
print("Summary")
print("="*80)

if np.max(np.abs(residual_diff)) < 0.1:
    print("\n✅ JUG and PINT residuals match to < 0.1 μs")
    print("   Fitting differences likely due to numerical precision in optimizer")
elif np.max(np.abs(residual_diff)) < 1.0:
    print("\n⚠️  JUG and PINT residuals differ by ~{:.3f} μs".format(np.max(np.abs(residual_diff))))
    print("   This is acceptable but should be investigated")
else:
    print(f"\n❌ JUG and PINT residuals differ significantly ({np.max(np.abs(residual_diff)):.3f} μs max)")
    print("   There is a fundamental difference in the residual calculation")

print("\n" + "="*80)
