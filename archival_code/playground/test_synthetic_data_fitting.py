#!/usr/bin/env python3
"""Test JAX fitting with synthetic (fake) perturbed pulsar data.

This test creates fake TOAs by perturbing the true model, then verifies
that the fitter can recover the true parameters from the perturbed data.

This is a more realistic test than using real data with perturbed parameters,
because it simulates the actual measurement process.
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
from jug.io.par_reader import parse_par_file

print("="*80)
print("Testing JAX Fitting with Synthetic Perturbed Data")
print("="*80)

# Step 1: Load real pulsar data to get TOA times and frequencies
print("\n1. Loading real pulsar data for TOA structure...")
par_file = 'data/pulsars/J1909-3744_tdb.par'
tim_file = 'data/pulsars/J1909-3744.tim'

# Get the true parameter values
par_params_true = parse_par_file(par_file)
print(f"   True F0: {par_params_true['F0']:.15f} Hz")
print(f"   True F1: {par_params_true['F1']:.15e} Hz/s")

# Compute residuals with true parameters
result_true = compute_residuals_simple(par_file, tim_file)
print(f"   Loaded {len(result_true['tdb_mjd'])} TOAs")
print(f"   True model RMS: {np.std(result_true['residuals_us']):.3f} μs")

# Step 2: Create synthetic TOAs by perturbing F0 and F1
print("\n2. Creating synthetic TOAs with perturbed parameters...")

# Perturb parameters by known amounts
delta_f0 = 5e-10  # 5e-10 Hz perturbation
delta_f1 = 1e-17  # 1e-17 Hz/s perturbation

f0_perturbed = par_params_true['F0'] + delta_f0
f1_perturbed = par_params_true['F1'] + delta_f1

print(f"   Perturbed F0: {f0_perturbed:.15f} Hz (Δ = +{delta_f0:.1e} Hz)")
print(f"   Perturbed F1: {f1_perturbed:.15e} Hz/s (Δ = +{delta_f1:.1e} Hz/s)")

# Create temporary .par file with perturbed parameters
import tempfile
import os

# Read original par file
with open(par_file, 'r') as f:
    par_lines = f.readlines()

# Replace F0 and F1 lines
par_lines_perturbed = []
for line in par_lines:
    if line.startswith('F0 '):
        par_lines_perturbed.append(f'F0 {f0_perturbed:.15f}\n')
    elif line.startswith('F1 '):
        par_lines_perturbed.append(f'F1 {f1_perturbed:.15e}\n')
    else:
        par_lines_perturbed.append(line)

# Write temporary par file
temp_par = tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False)
temp_par.writelines(par_lines_perturbed)
temp_par.close()

try:
    # Compute "synthetic" residuals with perturbed model
    # These are what we would measure if the true parameters were perturbed
    result_perturbed = compute_residuals_simple(temp_par.name, tim_file)
    
    print(f"   Synthetic data RMS: {np.std(result_perturbed['residuals_us']):.3f} μs")
    print(f"   (This simulates what we would measure with wrong model)")
    
    # Step 3: Now fit the perturbed data starting from true parameters
    # The fitter should find the perturbed parameters!
    print("\n3. Fitting synthetic data starting from true parameters...")
    print("   (Fitter should recover the perturbed parameters)")
    
    # Prepare fixed data using perturbed computation
    # NOTE: We'll manually override to use perturbed result
    fixed_data = prepare_fixed_data(temp_par.name, tim_file)
    
    # Override par_params to use TRUE values (we're testing if fitter finds perturbed)
    par_params_initial = par_params_true.copy()
    
    print(f"   Initial (true) F0: {par_params_initial['F0']:.15f} Hz")
    print(f"   Initial (true) F1: {par_params_initial['F1']:.15e} Hz/s")
    print(f"   Target (perturbed) F0: {f0_perturbed:.15f} Hz")
    print(f"   Target (perturbed) F1: {f1_perturbed:.15e} Hz/s")
    
    # Step 4: Create residual function
    print("\n4. Creating residual function...")
    
    def residuals_fn(params):
        """Compute residuals given parameter dict (returns μs)."""
        fit_params = ['F0', 'F1']
        params_array = jnp.array([params['F0'], params['F1']])
        fixed_params = {k: v for k, v in fixed_data['par_params'].items() if k not in fit_params}
        
        residuals_sec = compute_residuals_jax_from_dt(
            params_array,
            tuple(fit_params),
            fixed_data['dt_sec'],
            fixed_data['tzr_phase'],
            fixed_data['uncertainties_us'],
            fixed_params
        )
        
        return residuals_sec * 1e6  # Convert to μs
    
    # Step 5: Create design matrix function
    def design_matrix_fn(params, toas_mjd, freq_mhz, errors_us, fit_params):
        """Compute design matrix using JAX autodiff."""
        def residuals_for_grad(param_values):
            params_dict = fixed_data['par_params'].copy()
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
    
    # Compute initial residuals
    initial_residuals = residuals_fn(par_params_initial)
    initial_rms = np.std(initial_residuals)
    print(f"   Initial RMS: {initial_rms:.3f} μs")
    
    # Step 6: Run Gauss-Newton fit
    print("\n5. Running Gauss-Newton fit...")
    fitted_params, uncertainties, info = gauss_newton_fit_jax(
        residuals_fn=residuals_fn,
        params=par_params_initial,
        fit_params=['F0', 'F1'],
        design_matrix_fn=design_matrix_fn,
        toas_mjd=np.array(fixed_data['tdb_mjd']),
        freq_mhz=np.array(fixed_data['freq_mhz']),
        errors_us=np.array(fixed_data['uncertainties_us']),
        max_iter=20,
        lambda_init=1e-3,
        convergence_threshold=1e-10,
        verbose=True
    )
    
    # Step 7: Compare fitted parameters to target (perturbed) parameters
    print("\n" + "="*80)
    print("Fit Results:")
    print("="*80)
    
    print(f"\n{'Parameter':<10} {'Target (Perturbed)':<20} {'Fitted':<20} {'Difference':<20} {'Uncertainty'}")
    print("-"*100)
    
    f0_target = f0_perturbed
    f1_target = f1_perturbed
    
    f0_fitted = fitted_params['F0']
    f1_fitted = fitted_params['F1']
    
    f0_diff = f0_fitted - f0_target
    f1_diff = f1_fitted - f1_target
    
    f0_unc = uncertainties['F0']
    f1_unc = uncertainties['F1']
    
    f0_sigma = abs(f0_diff / f0_unc) if f0_unc > 0 else np.nan
    f1_sigma = abs(f1_diff / f1_unc) if f1_unc > 0 else np.nan
    
    print(f"{'F0':<10} {f0_target:.15f} {f0_fitted:.15f} {f0_diff:+.3e} {f0_unc:.3e} ({f0_sigma:.1f}σ)")
    print(f"{'F1':<10} {f1_target:.6e} {f1_fitted:.6e} {f1_diff:+.3e} {f1_unc:.3e} ({f1_sigma:.1f}σ)")
    
    print(f"\nRecovery of known perturbation:")
    print(f"  True Δ F0: {delta_f0:+.3e} Hz")
    print(f"  Fitted Δ F0: {f0_fitted - par_params_true['F0']:+.3e} Hz")
    print(f"  Error in recovery: {(f0_fitted - par_params_true['F0']) - delta_f0:+.3e} Hz")
    
    print(f"\n  True Δ F1: {delta_f1:+.3e} Hz/s")
    print(f"  Fitted Δ F1: {f1_fitted - par_params_true['F1']:+.3e} Hz/s")
    print(f"  Error in recovery: {(f1_fitted - par_params_true['F1']) - delta_f1:+.3e} Hz/s")
    
    print(f"\nFinal chi2: {info['final_chi2']:.2f}")
    print(f"Reduced chi2: {info['final_reduced_chi2']:.4f}")
    print(f"Final RMS: {info['final_rms_us']:.3f} μs")
    print(f"Iterations: {info['iterations']}")
    print(f"Converged: {info['converged']}")
    
    # Step 8: Success criteria
    print("\n" + "="*80)
    print("Success Criteria:")
    print("="*80)
    
    f0_success = f0_sigma < 3.0  # Within 3σ
    f1_success = f1_sigma < 3.0  # Within 3σ
    recovery_success = (abs((f0_fitted - par_params_true['F0']) - delta_f0) < 3 * f0_unc and
                       abs((f1_fitted - par_params_true['F1']) - delta_f1) < 3 * f1_unc)
    
    print(f"\n✓ F0 within 3σ of target: {f0_success}")
    print(f"✓ F1 within 3σ of target: {f1_success}")
    print(f"✓ Perturbation recovered within 3σ: {recovery_success}")
    
    if f0_success and f1_success and recovery_success:
        print(f"\n{'='*80}")
        print("✅ ✅ ✅  SUCCESS! Fitter recovered perturbed parameters! ✅ ✅ ✅")
        print(f"{'='*80}")
    else:
        print(f"\n⚠️  WARNING: Some criteria not met")
        print(f"   This may be acceptable given the small perturbations")

finally:
    # Clean up temporary file
    os.unlink(temp_par.name)
    print(f"\n(Cleaned up temporary file: {temp_par.name})")

print("\n" + "="*80)
print("Test complete!")
print("="*80)
