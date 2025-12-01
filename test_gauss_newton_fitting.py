#!/usr/bin/env python3
"""Test Gauss-Newton fitting with the new high-precision JAX residuals.

This script verifies that we can fit F0 and F1 using the new compute_residuals_jax_from_dt
function and recover values consistent with PINT.
"""

# CRITICAL: Enable JAX float64 BEFORE ANY imports
import jax
jax.config.update('jax_enable_x64', True)

import jug
import numpy as np
import jax.numpy as jnp
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.core import prepare_fixed_data, compute_residuals_jax_from_dt
from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax

print("="*80)
print("Testing Gauss-Newton Fitting with High-Precision JAX Residuals")
print("="*80)

# Step 1: Prepare fixed data
print("\n1. Preparing fixed data...")
fixed_data = prepare_fixed_data(
    'data/pulsars/J1909-3744_tdb.par',
    'data/pulsars/J1909-3744.tim'
)

par_params = fixed_data['par_params']
print(f"   Reference F0: {par_params['F0']:.15f} Hz")
print(f"   Reference F1: {par_params['F1']:.15e} Hz/s")

# Step 2: Create residual function for fitting
print("\n2. Creating residual function...")

def residuals_fn(params):
    """Compute residuals given parameter dict (returns μs).
    
    NOTE: Does NOT subtract weighted mean - that's arbitrary for fitting!
    """
    # Extract which parameters to update
    fit_params = ['F0', 'F1']
    params_array = jnp.array([params['F0'], params['F1']])
    fixed_params = {k: v for k, v in par_params.items() if k not in fit_params}
    
    # Compute residuals (returns seconds, no weighted mean)
    residuals_sec = compute_residuals_jax_from_dt(
        params_array,
        tuple(fit_params),
        fixed_data['dt_sec'],
        fixed_data['tzr_phase'],
        fixed_data['uncertainties_us'],
        fixed_params
    )
    
    # Convert to microseconds (NO weighted mean subtraction!)
    return residuals_sec * 1e6


# Step 3: Create design matrix function
print("\n3. Creating design matrix function...")

def design_matrix_fn(params, toas_mjd, freq_mhz, errors_us, fit_params):
    """Compute design matrix using JAX autodiff."""
    # Create residual function for autodiff (returns μs)
    def residuals_for_grad(param_values):
        params_dict = par_params.copy()
        for i, name in enumerate(fit_params):
            params_dict[name] = param_values[i]
        return residuals_fn(params_dict)
    
    # Get current parameter values
    param_values = jnp.array([params[name] for name in fit_params])
    
    # Compute Jacobian using JAX (units: μs/parameter)
    jacobian_fn = jax.jacfwd(residuals_for_grad)
    jacobian_us = jacobian_fn(param_values)  # Shape: (n_toas, n_params)
    
    # Convert to seconds
    jacobian_sec = np.array(jacobian_us) * 1e-6
    
    # Apply weights: M_weighted = M / errors (where errors are in seconds)
    errors_sec = errors_us * 1e-6
    M_weighted = jacobian_sec / errors_sec[:, np.newaxis]
    
    return M_weighted


# Step 4: Perturb F0 and F1 to test fitting
print("\n4. Perturbing parameters to test fitting...")
initial_params = par_params.copy()

# Perturb by a reasonable amount (~10 sigma, not millions of sigma!)
# For F0: uncertainty ~1e-10, so perturb by 1e-9
# For F1: uncertainty ~2e-18, so perturb by 2e-17
initial_params['F0'] = par_params['F0'] + 1e-9
initial_params['F1'] = par_params['F1'] + 2e-17

print(f"   Perturbed F0: {initial_params['F0']:.15f} Hz (Δ = +1e-9 Hz)")
print(f"   Perturbed F1: {initial_params['F1']:.15e} Hz/s (Δ = +2e-17 Hz/s)")

# Compute initial residuals
initial_residuals = residuals_fn(initial_params)
initial_rms = np.std(initial_residuals)
print(f"   Initial RMS: {initial_rms:.3f} μs")

# Step 5: Run Gauss-Newton fit
print("\n5. Running Gauss-Newton fit...")
fitted_params, uncertainties, info = gauss_newton_fit_jax(
    residuals_fn=residuals_fn,
    params=initial_params,
    fit_params=['F0', 'F1'],
    design_matrix_fn=design_matrix_fn,
    toas_mjd=np.array(fixed_data['tdb_mjd']),
    freq_mhz=np.array(fixed_data['freq_mhz']),
    errors_us=np.array(fixed_data['uncertainties_us']),
    max_iter=20,
    lambda_init=1e-3,
    convergence_threshold=1e-8,
    verbose=True
)

# Step 6: Compare to reference values
print("\n" + "="*80)
print("Fit Results:")
print("="*80)

print(f"\n{'Parameter':<10} {'Reference':<20} {'Fitted':<20} {'Difference':<20} {'Uncertainty'}")
print("-"*90)

for param in ['F0', 'F1']:
    ref_val = par_params[param]
    fit_val = fitted_params[param]
    diff = fit_val - ref_val
    unc = uncertainties[param]
    sigma = abs(diff / unc) if unc > 0 else np.nan
    
    if param == 'F0':
        print(f"{param:<10} {ref_val:.15f} {fit_val:.15f} {diff:+.3e} {unc:.3e} ({sigma:.1f}σ)")
    else:
        print(f"{param:<10} {ref_val:.6e} {fit_val:.6e} {diff:+.3e} {unc:.3e} ({sigma:.1f}σ)")

print(f"\nFinal chi2: {info['final_chi2']:.2f}")
print(f"Reduced chi2: {info['final_reduced_chi2']:.4f}")
print(f"Final RMS: {info['final_rms_us']:.3f} μs")
print(f"Iterations: {info['iterations']}")
print(f"Converged: {info['converged']}")

# Step 7: Compare to PINT (if available)
print("\n" + "="*80)
print("Comparison to PINT:")
print("="*80)

try:
    from pint.models import get_model
    from pint.toa import get_TOAs
    from pint.fitter import WLSFitter
    from pint.residuals import Residuals
    
    # Load PINT model and TOAs
    model = get_model('data/pulsars/J1909-3744_tdb.par')
    toas = get_TOAs('data/pulsars/J1909-3744.tim', ephem='DE440', planets=True, include_bipm=False)
    
    # Get PINT's values
    pint_f0 = model.F0.value
    pint_f1 = model.F1.value
    
    # Compute PINT residuals at reference values
    residuals_pint = Residuals(toas, model)
    pint_rms = residuals_pint.time_resids.std().to_value('us')
    pint_chi2 = residuals_pint.chi2
    pint_reduced_chi2 = residuals_pint.reduced_chi2
    
    print(f"\nPINT values:")
    print(f"  F0: {pint_f0:.15f} Hz")
    print(f"  F1: {pint_f1:.6e} Hz/s")
    print(f"  RMS: {pint_rms:.3f} μs")
    print(f"  Chi2: {pint_chi2:.2f}")
    print(f"  Reduced chi2: {pint_reduced_chi2:.4f}")
    
    print(f"\nJUG vs PINT:")
    print(f"  F0 difference: {fitted_params['F0'] - pint_f0:.3e} Hz")
    print(f"  F1 difference: {fitted_params['F1'] - pint_f1:.3e} Hz/s")
    print(f"  RMS difference: {info['final_rms_us'] - pint_rms:.3f} μs")
    print(f"  Chi2 difference: {info['final_chi2'] - pint_chi2:.2f}")
    
    # Check if values match within uncertainties
    f0_match = abs(fitted_params['F0'] - pint_f0) < 3 * uncertainties['F0']
    f1_match = abs(fitted_params['F1'] - pint_f1) < 3 * uncertainties['F1']
    
    if f0_match and f1_match:
        print(f"\n✅ JUG fitted values match PINT within 3σ!")
    else:
        print(f"\n⚠️  JUG fitted values differ from PINT by more than 3σ")
        
except ImportError:
    print("\nPINT not available for comparison")
except Exception as e:
    print(f"\nError comparing to PINT: {e}")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
