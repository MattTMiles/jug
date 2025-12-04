"""
Test that our WLS fitter matches PINT's WLSFitter exactly.
"""

import numpy as np
import jax
import jax.numpy as jnp
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter
jax.config.update('jax_enable_x64', True)

from jug.fitting.wls_fitter import fit_wls_jax, fit_wls_numerical

# Load J1909 data
par_file = '/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par'
tim_file = '/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, include_bipm=True)

print("="*80)
print("Testing WLS Fitter: JAX vs PINT")
print("="*80)

# Get PINT's pre-fit residuals and uncertainties
fitter = WLSFitter(toas, model)
fitter.update_resids()
pint_prefit_res = fitter.resids.time_resids.to('s').value
pint_sigma = model.scaled_toa_uncertainty(toas).to('s').value

print(f"\nNumber of TOAs: {len(toas)}")
print(f"Prefit RMS: {np.std(pint_prefit_res)*1e6:.3f} us")

# Get free parameters
free_params = model.free_params
print(f"\nFree parameters: {free_params}")
initial_values = np.array([float(getattr(model, p).value) for p in free_params], dtype=np.float64)
print(f"Initial values: {initial_values}")

# Fit with PINT
print("\n" + "="*80)
print("FITTING WITH PINT WLSFitter")
print("="*80)
chi2_pint = fitter.fit_toas(maxiter=5)
pint_final_values = np.array([getattr(model, p).value for p in free_params])
pint_final_errors = np.array([getattr(model, p).uncertainty_value for p in free_params])

print(f"\nPINT Results (chi2 = {chi2_pint:.3f}):")
for i, p in enumerate(free_params):
    print(f"  {p:10s}: {pint_final_values[i]:.15e} +/- {pint_final_errors[i]:.6e}")

# Get postfit residuals
fitter.update_resids()
pint_postfit_res = fitter.resids.time_resids.to('s').value
print(f"Postfit RMS: {np.std(pint_postfit_res)*1e6:.3f} us")

# Now fit with JAX using PINT's residual function
print("\n" + "="*80)
print("FITTING WITH JAX WLS (using PINT residuals)")
print("="*80)

# Reset model to initial values
for i, p in enumerate(free_params):
    getattr(model, p).value = initial_values[i]

# Create JAX-compatible residual function that calls PINT
def residual_fn_pint(params_array, times, freqs):
    """Compute residuals using PINT's model"""
    # Update model parameters
    for i, p in enumerate(free_params):
        getattr(model, p).value = float(params_array[i])
    
    # Compute residuals via PINT's Residuals class
    from pint.residuals import Residuals
    resids = Residuals(toas=toas, model=model)
    residuals = resids.time_resids.to('s').value
    
    return jnp.array(residuals, dtype=jnp.float64)

# Extract times and freqs (not actually used by PINT function, but needed for interface)
times = toas.get_mjds().value
freqs = toas.get_freqs().value

# Fit with JAX using numerical derivatives
jax_final_values, jax_final_errors, chi2_jax = fit_wls_numerical(
    residual_fn_pint,
    initial_values,
    times,
    freqs,
    pint_sigma,
    maxiter=1,  # Try just 1 iteration like PINT
    damping=1.0,  # No damping
    param_bounds=None  # No bounds
)

print(f"\nJAX Results (chi2 = {chi2_jax:.3f}):")
for i, p in enumerate(free_params):
    print(f"  {p:10s}: {jax_final_values[i]:.15e} +/- {jax_final_errors[i]:.6e}")

# Compare results
print("\n" + "="*80)
print("COMPARISON: PINT vs JAX")
print("="*80)

print(f"\nChi-squared:")
print(f"  PINT: {chi2_pint:.6f}")
print(f"  JAX:  {chi2_jax:.6f}")
print(f"  Diff: {abs(chi2_pint - chi2_jax):.6e}")

print(f"\nParameter values:")
for i, p in enumerate(free_params):
    diff = jax_final_values[i] - pint_final_values[i]
    sigma_diff = diff / pint_final_errors[i] if pint_final_errors[i] > 0 else np.nan
    print(f"  {p:10s}:")
    print(f"    PINT: {pint_final_values[i]:.15e}")
    print(f"    JAX:  {jax_final_values[i]:.15e}")
    print(f"    Diff: {diff:.6e} ({sigma_diff:.3f} sigma)")

print(f"\nParameter uncertainties:")
for i, p in enumerate(free_params):
    diff = jax_final_errors[i] - pint_final_errors[i]
    rel_diff = diff / pint_final_errors[i] * 100
    print(f"  {p:10s}:")
    print(f"    PINT: {pint_final_errors[i]:.6e}")
    print(f"    JAX:  {jax_final_errors[i]:.6e}")
    print(f"    Diff: {diff:.6e} ({rel_diff:.3f}%)")

# Check if they match within tolerance
params_match = np.allclose(jax_final_values, pint_final_values, rtol=1e-10, atol=1e-15)
errors_match = np.allclose(jax_final_errors, pint_final_errors, rtol=1e-6)
chi2_match = np.isclose(chi2_jax, chi2_pint, rtol=1e-6)

print("\n" + "="*80)
if params_match and errors_match and chi2_match:
    print("✓ SUCCESS: JAX WLS matches PINT WLS exactly!")
else:
    print("✗ MISMATCH: JAX WLS does not match PINT")
    print(f"  Parameters match: {params_match}")
    print(f"  Errors match: {errors_match}")
    print(f"  Chi2 match: {chi2_match}")
print("="*80)
