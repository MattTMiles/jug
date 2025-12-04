"""
Debug: Compare PINT's design matrix with numerical derivatives.
"""

import numpy as np
from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WLSFitter

# Load the data
model = get_model('data/pulsars/J1909-3744_tdb.par')
toas = get_TOAs('data/pulsars/J1909-3744.tim', planets=True)

# Get PINT's design matrix
M_pint, params, units = model.designmatrix(toas=toas, incfrozen=False, incoffset=True)

print("="*80)
print("PINT's Design Matrix")
print("="*80)
print(f"Shape: {M_pint.shape}")
print(f"Parameters: {params}")
print(f"\nFirst few rows of design matrix:")
for i in range(min(5, len(params))):
    print(f"  {params[i]:10s}: mean={np.mean(np.abs(M_pint[:, i])):.6e}, max={np.max(np.abs(M_pint[:, i])):.6e}")

# Now compute numerical derivatives
print("\n" + "="*80)
print("Numerical Derivatives")
print("="*80)

from pint.residuals import Residuals

def get_residuals(model, toas):
    resids = Residuals(toas=toas, model=model)
    return resids.time_resids.to('s').value

# Get baseline residuals
residuals_base = get_residuals(model, toas)

# Compute numerical derivatives for first few parameters (skip Offset)
eps = 1e-12
n_toas = len(toas)

for i, param_name in enumerate(params[1:6]):
    param = getattr(model, param_name)
    original_value = param.value
    
    # Perturb parameter
    if abs(original_value) > 1.0:
        step = eps * abs(original_value)
    else:
        step = eps
    
    param.value = original_value + step
    residuals_plus = get_residuals(model, toas)
    param.value = original_value  # Reset
    
    # Compute derivative
    d_resid_d_param = (residuals_plus - residuals_base) / step
    
    print(f"\n{param_name:10s}:")
    print(f"  Original value: {original_value}")
    print(f"  Step size: {step:.6e}")
    print(f"  PINT d(model)/d(param): mean={np.mean(np.abs(M_pint[:, i+1])):.6e}, max={np.max(np.abs(M_pint[:, i+1])):.6e}")
    print(f"  Numerical d(resid)/d(param): mean={np.mean(np.abs(d_resid_d_param)):.6e}, max={np.max(np.abs(d_resid_d_param)):.6e}")
    print(f"  Ratio: {np.mean(np.abs(d_resid_d_param)) / np.mean(np.abs(M_pint[:, i+1])):.3f}")
    
    # They should be opposite in sign
    sign_check = np.sum(np.sign(d_resid_d_param) == -np.sign(M_pint[:, i+1])) / n_toas
    print(f"  Fraction with opposite sign: {sign_check:.3f}")
