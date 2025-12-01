#!/usr/bin/env python3
"""Minimal test to verify the fitter works correctly.

Strategy: Use real data, start very close to the solution, verify fitter converges.
"""

import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt
from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt

print("="*80)
print("Minimal Fitter Test")
print("="*80)

# Load real data
par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get baseline
print("\n1. Computing baseline...")
result = compute_residuals_simple(par_file, tim_file)
print(f"   Baseline RMS: {result['rms_us']:.3f} μs (weighted)")

# Prepare fixed data
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params = fixed_data['par_params']

f0_true = par_params['F0']
f1_true = par_params['F1']

print(f"\n2. True parameters from .par file:")
print(f"   F0: {f0_true:.15f} Hz")
print(f"   F1: {f1_true:.15e} Hz/s")

# Check observation span to determine safe perturbation
dt_sec = np.array(fixed_data['dt_sec'])
span = dt_sec.max() - dt_sec.min()
max_delta_f0 = 0.25 / span  # 0.25 cycles max shift
print(f"\n3. Observation span: {span:.1e} s")
print(f"   Safe F0 perturbation: < {max_delta_f0:.2e} Hz")

# Use VERY small perturbation to stay in basin of attraction
delta_f0 = max_delta_f0 * 0.1  # Use 10% of safe limit (0.025 cycles)
delta_f1 = f1_true * 0.001  # 0.1% for F1

f0_start = f0_true + delta_f0
f1_start = f1_true + delta_f1

print(f"\n4. Starting parameters (perturbed):")
print(f"   F0: {f0_start:.15f} Hz (Δ = {delta_f0:.2e})")
print(f"   F1: {f1_start:.15e} Hz/s (Δ = {delta_f1:.2e})")

# Create residual function matching LM interface
def residual_func(toas, freqs, params_dict):
    """Residual function for LM fitter."""
    fit_params = tuple(sorted(params_dict.keys()))
    params_array = jnp.array([params_dict[name] for name in fit_params])

    fixed_params = {k: v for k, v in par_params.items() if k not in params_dict}

    return compute_residuals_jax_from_dt(
        params_array,
        fit_params,
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )

# Test residuals with starting parameters
print(f"\n5. Testing residuals with perturbed parameters...")
initial_params = {'F0': f0_start, 'F1': f1_start}
res_initial = residual_func(None, None, initial_params)
res_initial_us = np.array(res_initial) * 1e6
print(f"   RMS: {np.std(res_initial_us):.3f} μs")

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

print(f"\n{'Parameter':<10} {'True':<20} {'Start':<20} {'Fitted':<20} {'Recovery'}")
print("-"*90)

f0_fitted = fitted_params['F0']
f1_fitted = fitted_params['F1']

f0_error = f0_fitted - f0_true
f1_error = f1_fitted - f1_true

print(f"F0         {f0_true:.12e} {f0_start:.12e} {f0_fitted:.12e} {f0_error:+.2e}")
print(f"F1         {f1_true:.12e} {f1_start:.12e} {f1_fitted:.12e} {f1_error:+.2e}")

# Final residuals
res_final = residual_func(None, None, fitted_params)
res_final_us = np.array(res_final) * 1e6
final_rms = np.std(res_final_us)

print(f"\nFinal RMS: {final_rms:.3f} μs")
print(f"Baseline RMS: {result['rms_us']:.3f} μs")
print(f"Difference: {abs(final_rms - result['rms_us']):.3f} μs")

# Success criteria
f0_recovered = abs(f0_error) < abs(delta_f0) * 0.01  # Recovered to 1% of perturbation
f1_recovered = abs(f1_error) < abs(delta_f1) * 0.01

rms_match = abs(final_rms - result['rms_us']) < 0.1

print(f"\n" + "="*80)
print("Success Criteria:")
print("="*80)
print(f"F0 recovered (error < 1% of perturbation): {f0_recovered}")
print(f"F1 recovered (error < 1% of perturbation): {f1_recovered}")
print(f"RMS matches baseline (< 0.1 μs difference): {rms_match}")

if f0_recovered and f1_recovered and rms_match:
    print(f"\n{'='*80}")
    print("✅ ✅ ✅  SUCCESS! Fitter works correctly! ✅ ✅ ✅")
    print(f"{'='*80}")
else:
    print(f"\n⚠️  Test failed - fitter has issues")
    if not f0_recovered:
        print(f"   F0 error: {f0_error:.2e} Hz (should be < {abs(delta_f0) * 0.01:.2e})")
    if not f1_recovered:
        print(f"   F1 error: {f1_error:.2e} Hz/s (should be < {abs(delta_f1) * 0.01:.2e})")
    if not rms_match:
        print(f"   RMS difference: {abs(final_rms - result['rms_us']):.3f} μs (should be < 0.1)")

print(f"\n{'='*80}")
print("Test Complete")
print(f"{'='*80}")
