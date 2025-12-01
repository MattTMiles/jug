"""
Debug the SVD solver by comparing PINT's fit_wls_svd with our implementation.
"""

import numpy as np
import scipy.linalg
from pint.fitter import fit_wls_svd
from pint.utils import normalize_designmatrix
import jax.numpy as jnp

# Import our implementation
from jug.fitting.wls_fitter import wls_solve_svd

# Create simple test data
np.random.seed(42)
n_toas = 100
n_params = 3

# Random design matrix and residuals
M = np.random.randn(n_toas, n_params)
residuals = np.random.randn(n_toas) * 0.01
sigma = np.ones(n_toas) * 0.01

# Test PINT's implementation
params = ['P1', 'P2', 'P3']
threshold = 1e-14 * max(M.shape)

print("="*80)
print("Testing SVD Solver: PINT vs JUG")
print("="*80)

# PINT's fit_wls_svd
dpars_pint, Sigma_pint, Adiag_pint, (U_pint, S_pint, VT_pint) = fit_wls_svd(
    residuals, sigma, M, params, threshold
)

print("\nPINT Results:")
print(f"  dpars: {dpars_pint}")
print(f"  Sigma diagonal: {np.diag(Sigma_pint)}")
print(f"  Adiag: {Adiag_pint}")

# Our implementation (with negate_dpars=False to match PINT)
M_jax = jnp.array(M)
residuals_jax = jnp.array(residuals)
sigma_jax = jnp.array(sigma)

dpars_jug, Sigma_jug, Adiag_jug = wls_solve_svd(
    residuals_jax, sigma_jax, M_jax, threshold, negate_dpars=False
)

print("\nJUG Results:")
print(f"  dpars: {np.array(dpars_jug)}")
print(f"  Sigma diagonal: {np.diag(Sigma_jug)}")
print(f"  Adiag: {np.array(Adiag_jug)}")

# Compare
print("\n" + "="*80)
print("Comparison:")
print("="*80)
print(f"\ndpars diff: {np.linalg.norm(dpars_pint - np.array(dpars_jug)):.6e}")
print(f"Sigma diff: {np.linalg.norm(Sigma_pint - np.array(Sigma_jug)):.6e}")
print(f"Adiag diff: {np.linalg.norm(Adiag_pint - np.array(Adiag_jug)):.6e}")

dpars_match = np.allclose(dpars_pint, np.array(dpars_jug), rtol=1e-10)
Sigma_match = np.allclose(Sigma_pint, np.array(Sigma_jug), rtol=1e-10)
Adiag_match = np.allclose(Adiag_pint, np.array(Adiag_jug), rtol=1e-10)

print(f"\ndpars match: {dpars_match}")
print(f"Sigma match: {Sigma_match}")
print(f"Adiag match: {Adiag_match}")

if dpars_match and Sigma_match and Adiag_match:
    print("\n✓ SUCCESS: JUG SVD solver matches PINT exactly!")
else:
    print("\n✗ MISMATCH: JUG SVD solver differs from PINT")
    
    # Debug: manually compute step by step
    print("\n" + "="*80)
    print("Step-by-step debugging:")
    print("="*80)
    
    # Step 1: Weight by sigma
    r1_pint = residuals / sigma
    M1_pint = M / sigma[:, None]
    
    r1_jug = np.array(residuals_jax / sigma_jax)
    M1_jug = np.array(M_jax / sigma_jax[:, None])
    
    print(f"\nStep 1 - Weighting:")
    print(f"  r1 diff: {np.linalg.norm(r1_pint - r1_jug):.6e}")
    print(f"  M1 diff: {np.linalg.norm(M1_pint - M1_jug):.6e}")
    
    # Step 2: Normalize design matrix
    M2_pint, Adiag_pint_step = normalize_designmatrix(M1_pint, params)
    
    from jug.fitting.wls_fitter import normalize_designmatrix as normalize_jug
    M2_jug, Adiag_jug_step = normalize_jug(jnp.array(M1_pint))
    M2_jug = np.array(M2_jug)
    Adiag_jug_step = np.array(Adiag_jug_step)
    
    print(f"\nStep 2 - Normalization:")
    print(f"  M2 diff: {np.linalg.norm(M2_pint - M2_jug):.6e}")
    print(f"  Adiag diff: {np.linalg.norm(Adiag_pint_step - Adiag_jug_step):.6e}")
    print(f"  PINT Adiag: {Adiag_pint_step}")
    print(f"  JUG Adiag:  {Adiag_jug_step}")
    
    # Step 3: SVD
    U_check, S_check, VT_check = scipy.linalg.svd(M2_pint, full_matrices=False)
    print(f"\nStep 3 - SVD:")
    print(f"  PINT S: {S_pint}")
    print(f"  Check S: {S_check}")
    print(f"  S diff: {np.linalg.norm(S_pint - S_check):.6e}")
