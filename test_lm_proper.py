#!/usr/bin/env python3
"""Test LM fitter with the CORRECT residual function that recomputes delays.

This uses compute_residuals_jax() instead of compute_residuals_jax_from_dt(),
which allows DM to be fitted properly.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax
from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt

print("="*80)
print("Testing LM Fitter with Proper Residual Function")
print("="*80)

par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get baseline
print("\n1. Computing baseline...")
result = compute_residuals_simple(par_file, tim_file)
print(f"   Baseline RMS (weighted): {result['rms_us']:.3f} μs")

# Prepare fixed data
print("\n2. Preparing fixed data...")
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

f0_true = par_params['F0']
f1_true = par_params['F1']
dm_true = par_params['DM']

print(f"\n3. True parameters:")
print(f"   F0: {f0_true:.15f} Hz")
print(f"   F1: {f1_true:.15e} Hz/s")
print(f"   DM: {dm_true:.15f} pc/cm^3")

# Small perturbations
dt_span = np.array(fixed_data['dt_sec']).max() - np.array(fixed_data['dt_sec']).min()
max_safe_df0 = 0.25 / dt_span
delta_f0 = max_safe_df0 * 0.05  # 5% of safe limit
delta_f1 = abs(f1_true) * 0.001
delta_dm = dm_true * 0.0001

print(f"\n4. Perturbing parameters...")
print(f"   ΔF0: {delta_f0:.2e} Hz (phase shift: {delta_f0 * dt_span:.3f} cycles)")
print(f"   ΔF1: {delta_f1:.2e} Hz/s")
print(f"   ΔDM: {delta_dm:.2e} pc/cm^3")

f0_start = f0_true + delta_f0
f1_start = f1_true + delta_f1
dm_start = dm_true + delta_dm

# Create residual function using compute_residuals_jax (can fit DM!)
def residual_func(toas, freqs, params_dict):
    """Residual function that recomputes DM delays."""
    fit_params = tuple(sorted(params_dict.keys()))
    params_array = jnp.array([params_dict[name] for name in fit_params])

    fixed_params = {k: v for k, v in par_params.items() if k not in params_dict}

    # Use compute_residuals_jax which recomputes DM delay
    residuals_sec = compute_residuals_jax(
        params_array,
        fit_params,
        fixed_data['tdb_mjd'],
        fixed_data['freq_mhz'],
        fixed_data['geometric_delay_sec'],  # All delays except DM
        fixed_data['other_delays_minus_dm_sec'],  # Currently zeros
        fixed_data['pepoch'],
        fixed_data['dm_epoch'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )

    return residuals_sec

# Helper function for weighted RMS
def weighted_rms(residuals_us, errors_us):
    """Compute weighted RMS."""
    weights = 1.0 / (errors_us ** 2)
    return np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))

errors_us = np.array(fixed_data['uncertainties_us'])

# Test with initial parameters
print(f"\n5. Testing residual function...")
initial_params = {'F0': f0_start, 'F1': f1_start, 'DM': dm_start}
res_initial = residual_func(None, None, initial_params)
res_initial_us = np.array(res_initial) * 1e6

wrms_initial = weighted_rms(res_initial_us, errors_us)
print(f"   Initial WRMS: {wrms_initial:.3f} μs")

# Now test with TRUE parameters (should match baseline)
true_params = {'F0': f0_true, 'F1': f1_true, 'DM': dm_true}
res_true = residual_func(None, None, true_params)
res_true_us = np.array(res_true) * 1e6

wrms_true = weighted_rms(res_true_us, errors_us)
print(f"   True params WRMS: {wrms_true:.3f} μs")
print(f"   Baseline WRMS: {result['rms_us']:.3f} μs")
print(f"   Match: {abs(wrms_true - result['rms_us']) < 0.01}")

# Fit with LM
print(f"\n6. Running Levenberg-Marquardt fitter...")
print("="*80)

fitted_params = fit_levenberg_marquardt(
    residual_func,
    initial_params,
    fixed_data['tdb_mjd'],
    fixed_data['freq_mhz'],
    fixed_data['uncertainties_us'],
    max_iterations=20,
    tolerance=1e-10,
    initial_damping=1e-3,
    verbose=True
)

print("="*80)
print("\n7. Results:")
print("="*80)

print(f"\n{'Parameter':<10} {'True':<20} {'Start':<20} {'Fitted':<20} {'Error'}")
print("-"*95)

for param in ['F0', 'F1', 'DM']:
    true_val = par_params[param]
    start_val = initial_params[param]
    fitted_val = fitted_params[param]
    error = fitted_val - true_val

    print(f"{param:<10} {true_val:.12e} {start_val:.12e} {fitted_val:.12e} {error:+.2e}")

# Final residuals
res_final = residual_func(None, None, fitted_params)
res_final_us = np.array(res_final) * 1e6
final_wrms = weighted_rms(res_final_us, errors_us)

print(f"\nFinal WRMS: {final_wrms:.3f} μs")
print(f"Baseline WRMS: {result['rms_us']:.3f} μs")
print(f"Difference: {abs(final_wrms - result['rms_us']):.3f} μs")

# Success criteria
all_recovered = all([
    abs(fitted_params[p] - par_params[p]) < abs(initial_params[p] - par_params[p]) * 0.1
    for p in ['F0', 'F1', 'DM']
])
rms_match = abs(final_wrms - result['rms_us']) < 0.1

print(f"\n{'='*80}")
print("Success Criteria:")
print(f"{'='*80}")
print(f"All parameters recovered (< 10% of initial perturbation): {all_recovered}")
print(f"RMS matches baseline (< 0.1 μs difference): {rms_match}")

if all_recovered and rms_match:
    print(f"\n{'='*80}")
    print("✅ ✅ ✅  SUCCESS! Fitter works correctly! ✅ ✅ ✅")
    print(f"{'='*80}")
else:
    print(f"\n⚠️  Test failed")

print(f"\n{'='*80}")
print("Test Complete")
print(f"{'='*80}")
