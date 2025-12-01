"""
Test WLS fitter using PINT's actual design matrix (not numerical derivatives).
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter
import jax.numpy as jnp

from jug.fitting.wls_fitter import wls_solve_svd

print("="*80)
print("Testing WLS Fitter: PINT Design Matrix + JUG Solver")
print("="*80)

# Load the data
model = get_model('data/pulsars/J1909-3744_tdb.par')
toas = get_TOAs('data/pulsars/J1909-3744.tim', planets=True)

# Fit with PINT first for reference
print("\n" + "="*80)
print("FITTING WITH PINT WLSFitter")
print("="*80)
fitter = WLSFitter(toas=toas, model=model)
fitter.fit_toas(maxiter=1)

pint_chi2 = fitter.resids.chi2
pint_free_params = list(fitter.model.free_params)
pint_final_values = np.array([getattr(fitter.model, p).value for p in pint_free_params])
pint_final_errors = np.array([getattr(fitter.model, p).uncertainty_value for p in pint_free_params])

print(f"\nPINT Results (chi2 = {pint_chi2:.3f}):")
for i, p in enumerate(pint_free_params):
    print(f"  {p:10s}: {pint_final_values[i]:.15e} +/- {pint_final_errors[i]:.6e}")
print(f"Postfit RMS: {fitter.resids.rms_weighted():.3f} us")

# Now use PINT's design matrix with JUG's solver
print("\n" + "="*80)
print("FITTING WITH PINT Design Matrix + JUG SVD Solver")
print("="*80)

# Reset model to initial state
model_jug = get_model('data/pulsars/J1909-3744_tdb.par')

# Get initial residuals and design matrix from PINT
from pint.residuals import Residuals
resids = Residuals(toas=toas, model=model_jug)
residuals = resids.time_resids.to('s').value
sigma = model_jug.scaled_toa_uncertainty(toas).to('s').value

# Get PINT's design matrix (d(model)/d(params))
M_pint, params, units = model_jug.designmatrix(toas=toas, incfrozen=False, incoffset=True)

print(f"\nDesign matrix shape: {M_pint.shape}")
print(f"Parameters: {params}")

# Use JUG's SVD solver
# NOTE: PINT's M is d(model)/d(param), so we use negate_dpars=False
# Convert to float64 (JAX doesn't support float128)
M_jax = jnp.array(M_pint[:,1:], dtype=jnp.float64)  # Skip Offset column
residuals_jax = jnp.array(residuals, dtype=jnp.float64)
sigma_jax = jnp.array(sigma, dtype=jnp.float64)
params_jug = params[1:]  # Skip Offset

threshold = 1e-14 * max(M_jax.shape)
dpars, Sigma, Adiag = wls_solve_svd(
    residuals_jax, sigma_jax, M_jax, threshold, negate_dpars=False
)

# Apply parameter updates
jug_values = np.array([getattr(model_jug, p).value for p in params_jug])
jug_final_values = jug_values + np.array(dpars)
jug_final_errors = np.sqrt(np.diag(Sigma))

# Update model and compute final chi2
for i, p in enumerate(params_jug):
    param = getattr(model_jug, p)
    param.value = jug_final_values[i]

resids_final = Residuals(toas=toas, model=model_jug)
jug_chi2 = resids_final.chi2

print(f"\nJUG Results (chi2 = {jug_chi2:.3f}):")
for i, p in enumerate(params_jug):
    print(f"  {p:10s}: {jug_final_values[i]:.15e} +/- {jug_final_errors[i]:.6e}")

# Compare
print("\n" + "="*80)
print("COMPARISON: PINT vs JUG (using PINT's design matrix)")
print("="*80)

print(f"\nChi-squared:")
print(f"  PINT: {pint_chi2:.6f}")
print(f"  JUG:  {jug_chi2:.6f}")
print(f"  Diff: {abs(pint_chi2 - jug_chi2):.6e}")

print(f"\nParameter values:")
for i, p in enumerate(params_jug):
    diff = jug_final_values[i] - pint_final_values[i]
    sigma_diff = diff / pint_final_errors[i] if pint_final_errors[i] > 0 else np.nan
    print(f"  {p:10s}:")
    print(f"    PINT: {pint_final_values[i]:.15e}")
    print(f"    JUG:  {jug_final_values[i]:.15e}")
    print(f"    Diff: {diff:.6e} ({sigma_diff:.3f} sigma)")

print(f"\nParameter uncertainties:")
for i, p in enumerate(params_jug):
    diff = jug_final_errors[i] - pint_final_errors[i]
    rel_diff = diff / pint_final_errors[i] * 100
    print(f"  {p:10s}:")
    print(f"    PINT: {pint_final_errors[i]:.6e}")
    print(f"    JUG:  {jug_final_errors[i]:.6e}")
    print(f"    Diff: {diff:.6e} ({rel_diff:.3f}%)")

# Check if they match
params_match = np.allclose(jug_final_values, pint_final_values, rtol=1e-10, atol=1e-15)
errors_match = np.allclose(jug_final_errors, pint_final_errors, rtol=1e-6)
chi2_match = np.isclose(jug_chi2, pint_chi2, rtol=1e-6)

print("\n" + "="*80)
if params_match and errors_match and chi2_match:
    print("✓ SUCCESS: JUG solver matches PINT exactly when using PINT's design matrix!")
else:
    print("✗ MISMATCH: JUG solver differs from PINT")
    print(f"  Parameters match: {params_match}")
    print(f"  Errors match: {errors_match}")
    print(f"  Chi2 match: {chi2_match}")
print("="*80)
