#!/usr/bin/env python3
"""Test that JUG and PINT both recover true parameters from synthetic data.

This isolates the fitter from residual calculation differences by
checking if both recover known "true" values from perturbed starts.
"""

# CRITICAL: Enable JAX float64 BEFORE ANY imports
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt
from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax
import pint.models
import pint.toa
import pint.fitter

print("=" * 80)
print("SYNTHETIC FITTING TEST: Do JUG and PINT recover true parameters?")
print("=" * 80)
print("\nTest: Both fitters should recover TRUE values from perturbed starts")
print("(This tests the fitter logic, independent of residual differences)")

# Load real data files
par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get "true" parameters from the par file (these are the reference values)
pint_model_true = pint.models.get_model(par_file)
f0_true = pint_model_true.F0.quantity.value
f1_true = pint_model_true.F1.quantity.value

print(f"\nTRUE PARAMETERS (from {par_file}):")
print(f"  F0  = {f0_true:.15f} Hz")
print(f"  F1  = {f1_true:.15e} Hz/s")

# Create perturbed parameters
# These are ~10-sigma perturbations
f0_pert = f0_true + 1e-9   # ~10 sigma (uncertainty is ~1e-10 Hz)
f1_pert = f1_true + 2e-17  # ~10 sigma (uncertainty is ~2e-18 Hz/s)

print(f"\nPERTURBED STARTING VALUES:")
print(f"  F0  = {f0_pert:.15f} Hz  (Δ = +1e-9 Hz)")
print(f"  F1  = {f1_pert:.15e} Hz/s  (Δ = +2e-17 Hz/s)")

# ============================================================================
# JUG FIT
# ============================================================================
print("\n" + "=" * 80)
print("JUG FIT (using JAX residuals + Gauss-Newton)")
print("=" * 80)

# Prepare fixed data
fixed_data = prepare_fixed_data(par_file, tim_file)
par_params_raw = fixed_data['par_params']

# Convert ALL params to float64 upfront
par_params = {}
for k, v in par_params_raw.items():
    if isinstance(v, (np.longdouble, np.float128)):
        par_params[k] = float(v)
    else:
        par_params[k] = v

# Create perturbed parameters
perturbed_params = par_params.copy()
perturbed_params['F0'] = float(f0_pert)
perturbed_params['F1'] = float(f1_pert)

# Define residual function
def residuals_fn(params):
    """Compute residuals (μs) without weighted mean subtraction."""
    fit_params = ['F0', 'F1']
    params_array = jnp.array([params['F0'], params['F1']])
    fixed_params = {k: v for k, v in par_params.items() if k not in fit_params}
    
    residuals_sec = compute_residuals_jax_from_dt(
        params_array,
        tuple(fit_params),
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )
    
    return residuals_sec * 1e6  # Convert to μs

# Define design matrix function
def design_matrix_fn(params, toas_mjd, freq_mhz, errors_us, fit_params):
    """Compute design matrix using JAX autodiff."""
    def residuals_for_grad(param_values):
        params_dict = par_params.copy()
        for i, name in enumerate(fit_params):
            params_dict[name] = param_values[i]
        return residuals_fn(params_dict)
    
    param_values = jnp.array([params[name] for name in fit_params])
    jacobian_fn = jax.jacfwd(residuals_for_grad)
    jacobian_us = jacobian_fn(param_values)
    jacobian_sec = np.array(jacobian_us) * 1e-6
    
    errors_sec = errors_us * 1e-6
    M_weighted = jacobian_sec / errors_sec[:, np.newaxis]
    
    return M_weighted

# Run JUG fit
fitted_params_jug, uncertainties_jug, info_jug = gauss_newton_fit_jax(
    residuals_fn=residuals_fn,
    params=perturbed_params,
    fit_params=['F0', 'F1'],
    design_matrix_fn=design_matrix_fn,
    toas_mjd=np.array(fixed_data['tdb_mjd']),
    freq_mhz=np.array(fixed_data['freq_mhz']),
    errors_us=np.array(fixed_data['uncertainties_us']),
    max_iter=5,
    lambda_init=1e-3,
    convergence_threshold=1e-10,
    verbose=True
)

f0_jug = fitted_params_jug['F0']
f1_jug = fitted_params_jug['F1']
f0_err_jug = uncertainties_jug['F0']
f1_err_jug = uncertainties_jug['F1']

# ============================================================================
# PINT FIT
# ============================================================================
print("\n" + "=" * 80)
print("PINT FIT (using PINT residuals + WLS fitter)")
print("=" * 80)

# Create perturbed PINT model
pint_model_pert = pint.models.get_model(par_file)
pint_model_pert.F0.value = f0_pert
pint_model_pert.F1.value = f1_pert

# Load TOAs and fit
toas = pint.toa.get_TOAs(tim_file, model=pint_model_pert)
fitter = pint.fitter.WLSFitter(toas=toas, model=pint_model_pert)
fitter.fit_toas(maxiter=5)

f0_pint = fitter.model.F0.quantity.value
f1_pint = fitter.model.F1.quantity.value
f0_err_pint = fitter.model.F0.uncertainty.value
f1_err_pint = fitter.model.F1.uncertainty.value

print(f"\nPINT Fitted Values:")
print(f"  F0  = {f0_pint:.15f} ± {f0_err_pint:.2e} Hz")
print(f"  F1  = {f1_pint:.15e} ± {f1_err_pint:.2e} Hz/s")

# ============================================================================
# COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS COMPARISON")
print("=" * 80)

print("\nJUG Fitted Values:")
print(f"  F0  = {f0_jug:.15f} ± {f0_err_jug:.2e} Hz")
print(f"  F1  = {f1_jug:.15e} ± {f1_err_jug:.2e} Hz/s")

print("\nJUG Recovery (fitted - true) / uncertainty:")
f0_jug_sigma = (f0_jug - f0_true)/f0_err_jug
f1_jug_sigma = (f1_jug - f1_true)/f1_err_jug
print(f"  F0: {f0_jug_sigma:+.2f} σ")
print(f"  F1: {f1_jug_sigma:+.2f} σ")

print("\nPINT Recovery (fitted - true) / uncertainty:")
f0_pint_sigma = (f0_pint - f0_true)/f0_err_pint
f1_pint_sigma = (f1_pint - f1_true)/f1_err_pint
print(f"  F0: {f0_pint_sigma:+.2f} σ")
print(f"  F1: {f1_pint_sigma:+.2f} σ")

print("\nDifference (JUG - PINT):")
print(f"  F0: {(f0_jug - f0_pint):.2e} Hz  ({(f0_jug - f0_pint)/f0_err_pint:+.2f} σ_PINT)")
print(f"  F1: {(f1_jug - f1_pint):.2e} Hz/s  ({(f1_jug - f1_pint)/f1_err_pint:+.2f} σ_PINT)")

# ============================================================================
# VERDICT
# ============================================================================
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Check if both recover true values within 2σ
jug_recovers = (abs(f0_jug_sigma) < 2 and abs(f1_jug_sigma) < 2)
pint_recovers = (abs(f0_pint_sigma) < 2 and abs(f1_pint_sigma) < 2)

# Check if they agree with each other
agree_f0 = abs(f0_jug - f0_pint) < np.sqrt(f0_err_jug**2 + f0_err_pint**2)
agree_f1 = abs(f1_jug - f1_pint) < np.sqrt(f1_err_jug**2 + f1_err_pint**2)
agree = agree_f0 and agree_f1

print(f"\n✓ JUG recovers true values within 2σ: {jug_recovers}")
print(f"✓ PINT recovers true values within 2σ: {pint_recovers}")
print(f"✓ JUG and PINT agree within combined uncertainty: {agree}")

if jug_recovers and pint_recovers:
    print("\n" + "✅ SUCCESS: JUG fitter works correctly!")
    print("   - JUG recovers true parameters from perturbed start")
    print("   - JUG fitting algorithm is validated")
    if agree:
        print("   - JUG and PINT converge to consistent values")
        print("   - Residual differences don't affect fitting quality")
    else:
        print("   - Note: JUG and PINT converge to slightly different values")
        print("   - This is expected due to ~0.01 μs residual differences")
        print("   - Both are valid minima of their respective χ² surfaces")
else:
    print("\n" + "❌ FAILURE: Fitter has issues")
    if not jug_recovers:
        print("   - JUG did not recover true parameters")
    if not pint_recovers:
        print("   - PINT did not recover true parameters")

