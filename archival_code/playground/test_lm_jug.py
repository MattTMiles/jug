"""Test Levenberg-Marquardt fitter with JUG data."""

# CRITICAL: Enable JAX float64 BEFORE imports
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from pathlib import Path

from jug.fitting.levenberg_marquardt import fit_levenberg_marquardt
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt

print("="*60)
print("Testing Levenberg-Marquardt Fitter with JUG")
print("="*60)

par_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par')
tim_file = Path('/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim')

# Get baseline
result = compute_residuals_simple(str(par_file), str(tim_file))
print(f"\nBaseline JUG weighted RMS: {result['rms_us']:.3f} μs")

# Prepare fixed data
fixed_data = prepare_fixed_data(str(par_file), str(tim_file))

# Extract parameters
par_params = fixed_data['par_params']
f0_init = par_params['F0']
f1_init = par_params['F1']
dm_init = par_params['DM']
raj_init = par_params['RAJ']
decj_init = par_params['DECJ']

print(f"\nInitial parameters:")
print(f"  F0:   {f0_init:.15f} Hz")
print(f"  F1:   {f1_init:.15e} Hz/s")
print(f"  DM:   {dm_init:.15f} pc/cm^3")

# Create residual function for fitter
# This version uses the pre-computed dt_sec (emission times)
def residual_func(toas, freqs, params_dict):
    """Compute residuals for given parameters using dt_sec method."""
    # Note: DM cannot be fitted with this method since it's baked into dt_sec
    # Only F0, F1 can be fitted

    # Determine which parameters are being fitted
    fit_param_names = tuple(sorted(params_dict.keys()))
    params_array = jnp.array([params_dict[name] for name in fit_param_names])

    # Fixed parameters (not being fitted)
    fixed_params = {}
    for key in par_params.keys():
        if key not in params_dict:
            fixed_params[key] = par_params[key]

    # Use the dt_sec method (emission times pre-computed with longdouble)
    return compute_residuals_jax_from_dt(
        params_array,
        fit_param_names,
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )

# Initial parameters (only F0, F1 can be fitted with dt_sec method - DM is baked in)
initial_params = {
    'F0': f0_init,
    'F1': f1_init,
}

# Test residual function
print("\nTesting residual function with initial parameters...")
res_test = residual_func(None, None, initial_params)  # toas/freqs not used in dt_sec method
res_test_us = np.array(res_test) * 1e6
print(f"  RMS: {np.std(res_test_us):.3f} μs (unweighted)")

# Perturb parameters slightly to give fitter something to do
# CRITICAL: Perturbation must be small enough to not cross cycle boundaries!
# For F0: |ΔF0 * observation_span| should be < 0.5 cycles
# observation_span ~ 4e7 s, so |ΔF0| < 1e-8 Hz
# We use 5e-9 Hz perturbation (0.2 cycles over the span - safe)
perturbed_params = {
    'F0': f0_init + 5e-9,  # Small absolute perturbation (0.2 cycles over observation)
    'F1': f1_init * 1.01,  # 1% perturbation (F1 is less sensitive to cycle wrapping)
}

print("\nPerturbed parameters:")
print(f"  F0:   {perturbed_params['F0']:.15f} Hz (ΔF0 = {perturbed_params['F0'] - f0_init:.6e})")
print(f"  F1:   {perturbed_params['F1']:.15e} Hz/s (ΔF1 = {perturbed_params['F1'] - f1_init:.6e})")
print(f"  Note: DM cannot be fitted with dt_sec method (it's baked into the emission times)")

res_perturbed = residual_func(None, None, perturbed_params)
res_perturbed_us = np.array(res_perturbed) * 1e6
print(f"  RMS after perturbation: {np.std(res_perturbed_us):.3f} μs")

# Now fit with Levenberg-Marquardt
print("\n" + "="*60)
print("Fitting with Levenberg-Marquardt...")
print("="*60)

fitted_params = fit_levenberg_marquardt(
    residual_func,
    perturbed_params,
    fixed_data['tdb_mjd'],  # Not actually used by residual_func, but needed for interface
    fixed_data['freq_mhz'],
    fixed_data['uncertainties_us'],
    max_iterations=10,
    tolerance=1e-9,
    initial_damping=1e-3,
    verbose=True,
)

print("\n" + "="*60)
print("Results")
print("="*60)

print(f"\n{'Parameter':<10} {'Initial':>20} {'Fitted':>20} {'Difference':>15}")
print("-" * 70)

for param_name in ['F0', 'F1']:
    init_val = initial_params[param_name]
    fitted_val = fitted_params[param_name]
    diff = fitted_val - init_val
    print(f"{param_name:<10} {init_val:>20.12e} {fitted_val:>20.12e} {diff:>15.6e}")

# Compute final residuals
res_final = residual_func(None, None, fitted_params)
res_final_us = np.array(res_final) * 1e6
print(f"\nFinal RMS: {np.std(res_final_us):.3f} μs")
print(f"Expected RMS: {result['rms_us']:.3f} μs (from baseline)")

recovery_pct = [
    100 * abs((fitted_params[p] - initial_params[p]) / initial_params[p])
    for p in ['F0', 'F1']
]
print(f"\nParameter recovery:")
for p, pct in zip(['F0', 'F1'], recovery_pct):
    print(f"  {p}: {pct:.6f}% difference from initial")

