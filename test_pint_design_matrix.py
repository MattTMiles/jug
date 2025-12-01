"""
Test using PINT's design matrix directly instead of numerical derivatives.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter
from jug.fitting.wls_fitter import wls_solve_svd

# Load J1909 data
par_file = '/home/mattm/soft/JUG/data/pulsars/J1909-3744_tdb.par'
tim_file = '/home/mattm/soft/JUG/data/pulsars/J1909-3744.tim'

model = get_model(par_file)
toas = get_TOAs(tim_file, planets=True, include_bipm=True)

print("="*80)
print("Testing with PINT's design matrix")
print("="*80)

# Create fitter
fitter = WLSFitter(toas, model)
fitter.update_resids()

# Get design matrix from PINT
M_pint, params_all, units = fitter.get_designmatrix()

# Skip "Offset" parameter (constant term)
params = [p for p in params_all if p != 'Offset']
offset_idx = params_all.index('Offset') if 'Offset' in params_all else None
if offset_idx is not None:
    M_pint = np.delete(M_pint, offset_idx, axis=1)

print(f"\nDesign matrix shape: {M_pint.shape}")
print(f"Parameters: {params}")

# Get residuals and uncertainties
residuals = fitter.resids.time_resids.to('s').value
sigma = model.scaled_toa_uncertainty(toas).to('s').value

print(f"Initial chi2: {np.sum((residuals/sigma)**2):.2f}")
print(f"Initial RMS: {np.std(residuals)*1e6:.3f} us")

# Use our WLS solver with PINT's design matrix
# Convert to float64 since PINT uses float128
dpars, Sigma, Adiag = wls_solve_svd(
    jnp.array(residuals, dtype=jnp.float64),
    jnp.array(sigma, dtype=jnp.float64),
    jnp.array(M_pint, dtype=jnp.float64),
    threshold=1e-14,
    negate_dpars=False  # PINT's M is d(model)/d(param), not d(residual)/d(param)!
)

dpars = np.array(dpars)
Sigma = np.array(Sigma)

# Extract uncertainties
param_errors_jug = np.sqrt(np.diag(Sigma))

print("\nJUG WLS results (using PINT's design matrix):")
for i, p in enumerate(params):
    print(f"  {p:10s}: Δp = {dpars[i]:.6e}, σ = {param_errors_jug[i]:.6e}")

# Now fit with PINT for comparison
print("\n" + "="*80)
print("PINT WLS results (1 iteration)")
print("="*80)

initial_values = {p: getattr(model, p).value for p in params}

chi2_pint = fitter.fit_toas(maxiter=1)
pint_final_values = {p: getattr(model, p).value for p in params}
pint_final_errors = {p: getattr(model, p).uncertainty_value for p in params}

print(f"\nChi2: {chi2_pint:.2f}")
for i, p in enumerate(params):
    dpars_pint = pint_final_values[p] - initial_values[p]
    print(f"  {p:10s}: Δp = {dpars_pint:.6e}, σ = {pint_final_errors[p]:.6e}")

# Compare
print("\n" + "="*80)
print("Comparison")
print("="*80)

print("\nParameter updates:")
for i, p in enumerate(params):
    dpars_pint = pint_final_values[p] - initial_values[p]
    diff = dpars[i] - dpars_pint
    rel_diff = abs(diff / dpars_pint) * 100 if dpars_pint != 0 else 0
    print(f"  {p:10s}: JUG={dpars[i]:.6e}, PINT={dpars_pint:.6e}, diff={rel_diff:.3f}%")

print("\nParameter errors:")
for i, p in enumerate(params):
    diff = param_errors_jug[i] - pint_final_errors[p]
    rel_diff = abs(diff / pint_final_errors[p]) * 100
    print(f"  {p:10s}: JUG={param_errors_jug[i]:.6e}, PINT={pint_final_errors[p]:.6e}, diff={rel_diff:.3f}%")

# Check match
dpars_match = np.allclose([dpars[i] for i in range(len(params))],
                         [pint_final_values[p] - initial_values[p] for p in params],
                         rtol=1e-10)
errors_match = np.allclose(param_errors_jug,
                          [pint_final_errors[p] for p in params],
                          rtol=1e-6)

print("\n" + "="*80)
if dpars_match and errors_match:
    print("✓ SUCCESS: JUG matches PINT exactly!")
else:
    print("✗ MISMATCH")
    print(f"  Parameter updates match: {dpars_match}")
    print(f"  Errors match: {errors_match}")
print("="*80)
