"""
Simple test of WLS fitter on synthetic data.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from jug.fitting.wls_fitter import fit_wls_jax, fit_wls_numerical

# Generate synthetic data
np.random.seed(42)
n_toas = 1000

# True parameters
true_f0 = 100.0  # Hz
true_f1 = -1e-15  # Hz/s

# Generate times
times = np.linspace(54000, 58000, n_toas)  # MJD
freqs = np.ones(n_toas) * 1400  # MHz

# Compute true residuals (with some systematic error)
def true_model(params, t):
    f0, f1 = params
    phase = f0 * t + 0.5 * f1 * t**2
    return phase

true_phase = true_model([true_f0, true_f1], times - times[0])
# Add noise
noise = np.random.normal(0, 0.01, n_toas)  # 0.01 cycles RMS
observed_phase = true_phase + noise

# Residual function
def residual_fn(params, t, f):
    model_phase = true_model(params, t - t[0])
    return observed_phase - model_phase

# Initial guess (perturbed from truth)
initial_params = np.array([true_f0 * 0.99, true_f1 * 1.1])
sigma = np.ones(n_toas) * 0.01  # 0.01 cycles uncertainty

print("="*80)
print("Testing WLS Fitter on Synthetic Data")
print("="*80)
print(f"\nTrue parameters:")
print(f"  F0: {true_f0:.15e}")
print(f"  F1: {true_f1:.15e}")
print(f"\nInitial parameters:")
print(f"  F0: {initial_params[0]:.15e}")
print(f"  F1: {initial_params[1]:.15e}")

# Test numerical fitter
print("\n" + "="*80)
print("Fitting with numerical derivatives")
print("="*80)
params_num, errors_num, chi2_num = fit_wls_numerical(
    residual_fn,
    initial_params,
    times,
    freqs,
    sigma,
    maxiter=5
)

print(f"\nNumerical Results (chi2 = {chi2_num:.3f}):")
print(f"  F0: {params_num[0]:.15e} +/- {errors_num[0]:.6e}")
print(f"  F1: {params_num[1]:.15e} +/- {errors_num[1]:.6e}")
print(f"\nRecovered vs True:")
print(f"  F0: {(params_num[0] - true_f0)/errors_num[0]:.3f} sigma")
print(f"  F1: {(params_num[1] - true_f1)/errors_num[1]:.3f} sigma")

# Test JAX fitter
print("\n" + "="*80)
print("Fitting with JAX autodiff")
print("="*80)

# Need JAX-compatible residual function
@jax.jit
def residual_fn_jax(params, t, f):
    f0, f1 = params
    t_rel = t - t[0]
    model_phase = f0 * t_rel + 0.5 * f1 * t_rel**2
    return jnp.array(observed_phase) - model_phase

params_jax, errors_jax, chi2_jax = fit_wls_jax(
    residual_fn_jax,
    initial_params,
    times,
    freqs,
    sigma,
    maxiter=5
)

print(f"\nJAX Results (chi2 = {chi2_jax:.3f}):")
print(f"  F0: {params_jax[0]:.15e} +/- {errors_jax[0]:.6e}")
print(f"  F1: {params_jax[1]:.15e} +/- {errors_jax[1]:.6e}")
print(f"\nRecovered vs True:")
print(f"  F0: {(params_jax[0] - true_f0)/errors_jax[0]:.3f} sigma")
print(f"  F1: {(params_jax[1] - true_f1)/errors_jax[1]:.3f} sigma")

# Compare numerical vs JAX
print("\n" + "="*80)
print("Comparison: Numerical vs JAX")
print("="*80)
print(f"\nParameter values:")
print(f"  F0:")
print(f"    Numerical: {params_num[0]:.15e}")
print(f"    JAX:       {params_jax[0]:.15e}")
print(f"    Diff:      {params_jax[0] - params_num[0]:.6e} ({(params_jax[0] - params_num[0])/errors_num[0]:.3f} sigma)")
print(f"  F1:")
print(f"    Numerical: {params_num[1]:.15e}")
print(f"    JAX:       {params_jax[1]:.15e}")
print(f"    Diff:      {params_jax[1] - params_num[1]:.6e} ({(params_jax[1] - params_num[1])/errors_num[1]:.3f} sigma)")

print(f"\nParameter errors:")
print(f"  F0:")
print(f"    Numerical: {errors_num[0]:.6e}")
print(f"    JAX:       {errors_jax[0]:.6e}")
print(f"    Rel diff:  {(errors_jax[0] - errors_num[0])/errors_num[0] * 100:.3f}%")
print(f"  F1:")
print(f"    Numerical: {errors_num[1]:.6e}")
print(f"    JAX:       {errors_jax[1]:.6e}")
print(f"    Rel diff:  {(errors_jax[1] - errors_num[1])/errors_num[1] * 100:.3f}%")

print(f"\nChi-squared:")
print(f"  Numerical: {chi2_num:.6f}")
print(f"  JAX:       {chi2_jax:.6f}")
print(f"  Diff:      {abs(chi2_jax - chi2_num):.6e}")

# Check if they match
params_match = np.allclose(params_jax, params_num, rtol=1e-10)
errors_match = np.allclose(errors_jax, errors_num, rtol=1e-6)
chi2_match = np.isclose(chi2_jax, chi2_num, rtol=1e-6)

print("\n" + "="*80)
if params_match and errors_match and chi2_match:
    print("✓ SUCCESS: JAX and numerical fitters match!")
else:
    print("✗ MISMATCH: JAX and numerical differ")
    print(f"  Parameters match: {params_match}")
    print(f"  Errors match: {errors_match}")
    print(f"  Chi2 match: {chi2_match}")
print("="*80)
