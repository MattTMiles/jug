"""Simple debug test for fitter convergence."""

import numpy as np
import pint.models as pm
import pint.toa as pt
import pint.fitter as pf
from pint.residuals import Residuals

# Load data
model = pm.get_model('data/pulsars/J1909-3744_tdb.par')
toas = pt.get_TOAs('data/pulsars/J1909-3744.tim', planets=True)

# Get errors
errors_us = toas.get_errors().to('us').value

# Store reference values
ref_F0 = model.F0.value
ref_F1 = model.F1.value

# Perturb
model.F0.value = ref_F0 * (1 + 1e-10)
model.F1.value = ref_F1 * (1 + 1e-9)

print(f"Reference F0: {ref_F0:.15e}")
print(f"Perturbed F0: {model.F0.value:.15e}")
print(f"Reference F1: {ref_F1:.15e}")
print(f"Perturbed F1: {model.F1.value:.15e}")

# Compute initial residuals
res0 = Residuals(toas, model)
r0_sec = res0.time_resids.to('s').value  # Use seconds!
errors_sec = errors_us * 1e-6  # Convert to seconds
weights = 1.0 / errors_sec**2
chi2_0 = np.sum(weights * r0_sec**2)

print(f"\nInitial chi2: {chi2_0:.6f}")
print(f"Initial RMS: {np.std(r0_sec) * 1e6:.3f} us")

# Test: what happens if we nudge F0 in the right direction?
delta_F0 = ref_F0 - model.F0.value
print(f"\nDelta F0 needed: {delta_F0:.6e}")

# Try 10% of the needed correction
model.F0.value += 0.1 * delta_F0
res1 = Residuals(toas, model)
r1_sec = res1.time_resids.to('s').value
chi2_1 = np.sum(weights * r1_sec**2)

print(f"\nAfter 10% correction:")
print(f"  Chi2: {chi2_1:.6f} (change: {chi2_1 - chi2_0:.6f})")
print(f"  RMS: {np.std(r1_sec) * 1e6:.3f} us")

# Check if chi2 improved
if chi2_1 < chi2_0:
    print("  ✓ Chi2 improved!")
else:
    print("  ✗ Chi2 got worse!")

# Now test numerical derivative
h_abs = ref_F0 * 1e-8  # Absolute step
model.F0.value = ref_F0 * (1 + 1e-10)  # Reset
res_base = Residuals(toas, model)
r_base = res_base.time_resids.to('s').value  # Seconds!

model.F0.value = ref_F0 * (1 + 1e-10) + h_abs
res_pert = Residuals(toas, model)
r_pert = res_pert.time_resids.to('s').value  # Seconds!

deriv_numerical = (r_pert - r_base) / h_abs  # Absolute derivative

print(f"\nNumerical derivative (∂r/∂F0):")
print(f"  Mean: {np.mean(deriv_numerical):.6e}")
print(f"  Std: {np.std(deriv_numerical):.6e}")
print(f"  Min: {np.min(deriv_numerical):.6e}")
print(f"  Max: {np.max(deriv_numerical):.6e}")

# Predict what delta F0 should be using Gauss-Newton
# delta = -(A^T W A)^-1 A^T W r
A = deriv_numerical[:, np.newaxis].astype(np.float64)  # Design matrix (n x 1)
ATA = (A.T @ np.diag(weights.astype(np.float64)) @ A).astype(np.float64)
ATr = (A.T @ np.diag(weights.astype(np.float64)) @ r_base.astype(np.float64)).astype(np.float64)

predicted_delta_F0 = -np.linalg.solve(ATA, ATr)[0]

print(f"\nGauss-Newton predicted delta F0: {predicted_delta_F0:.6e}")
print(f"True delta F0 needed: {delta_F0:.6e}")
print(f"Ratio: {predicted_delta_F0 / delta_F0:.3f}")

# Apply predicted update and check
model.F0.value = ref_F0 * (1 + 1e-10) + predicted_delta_F0
res_pred = Residuals(toas, model)
r_pred = res_pred.time_resids.to('s').value
chi2_pred = np.sum(weights * r_pred**2)

print(f"\nAfter predicted update:")
print(f"  New F0: {model.F0.value:.15e}")
print(f"  Chi2: {chi2_pred:.6f}")
print(f"  RMS: {np.std(r_pred) * 1e6:.3f} us")
print(f"  Improvement: {chi2_0 - chi2_pred:.6f}")

if chi2_pred < chi2_0:
    print("  ✓ Gauss-Newton step improved fit!")
else:
    print("  ✗ Gauss-Newton step made it worse!")
