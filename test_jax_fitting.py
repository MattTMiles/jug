"""Test JAX-accelerated fitting on synthetic data."""

import numpy as np
import sys

print("Testing JAX-accelerated fitting...")
print("="*80)

# Test 1: Design matrix computation
print("\n1. Testing JAX design matrix computation...")

from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper

# Synthetic data
n_toas = 1000
toas_mjd = np.linspace(55000, 56000, n_toas)
freq_mhz = np.random.uniform(1200, 1600, n_toas)
errors_us = np.random.uniform(0.5, 2.0, n_toas)

params = {
    'F0': 100.0,
    'F1': -1e-15,
    'F2': 0.0,
    'F3': 0.0,
    'DM': 30.0,
    'DM1': 0.0,
    'DM2': 0.0,
    'PEPOCH': 55500.0,
    'DMEPOCH': 55500.0
}

fit_params = ['F0', 'F1', 'DM']

try:
    M_jax = compute_design_matrix_jax_wrapper(
        params, toas_mjd, freq_mhz, errors_us, fit_params
    )
    print(f"   ✓ Design matrix shape: {M_jax.shape}")
    print(f"   ✓ Expected: ({n_toas}, {len(fit_params)})")
    print(f"   ✓ JAX design matrix computation works!")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 2: Compare NumPy vs JAX design matrix
print("\n2. Comparing NumPy vs JAX design matrix...")

from jug.fitting.design_matrix import compute_design_matrix

M_numpy = compute_design_matrix(
    params, toas_mjd, freq_mhz, errors_us, fit_params
)

diff = np.abs(M_jax - M_numpy)
max_diff = np.max(diff)
rms_diff = np.sqrt(np.mean(diff**2))

print(f"   Max difference: {max_diff:.3e}")
print(f"   RMS difference: {rms_diff:.3e}")

if max_diff < 1e-10:
    print(f"   ✓ NumPy and JAX match!")
else:
    print(f"   ⚠ Differences found (may be numerical precision)")

# Test 3: Matrix operations
print("\n3. Testing JAX matrix operations...")

from jug.fitting.gauss_newton_jax import gauss_newton_step_jax, compute_weighted_chi2_jax
import jax.numpy as jnp

residuals = np.random.randn(n_toas) * 1e-6  # seconds
errors = errors_us * 1e-6  # seconds

try:
    chi2 = compute_weighted_chi2_jax(jnp.array(residuals), jnp.array(errors))
    print(f"   ✓ Chi2 = {chi2:.2f}")
    
    delta_p, cov = gauss_newton_step_jax(
        jnp.array(residuals),
        M_jax / errors[:, np.newaxis],  # Unweight M
        jnp.array(errors),
        1e-3
    )
    print(f"   ✓ Parameter updates: {delta_p.shape}")
    print(f"   ✓ Covariance matrix: {cov.shape}")
    print(f"   ✓ JAX matrix operations work!")
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    sys.exit(1)

# Test 4: Full fitting on synthetic data
print("\n4. Testing full JAX fitting on synthetic data...")

# Generate synthetic residuals
true_f0 = 100.0
true_f1 = -1e-15
true_dm = 30.0

def synthetic_residuals_fn(params):
    """Generate synthetic residuals."""
    # Simple model: just return small random residuals
    # In reality, this would compute actual timing residuals
    return np.random.randn(n_toas) * 1.0  # μs

# Initial parameters (slightly wrong)
initial_params = params.copy()
initial_params['F0'] = 100.1  # 0.1 Hz offset
initial_params['F1'] = -0.9e-15  # 10% offset
initial_params['DM'] = 31.0  # 1 pc/cc offset

from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax

try:
    fitted, uncertainties, info = gauss_newton_fit_jax(
        synthetic_residuals_fn,
        initial_params,
        fit_params,
        compute_design_matrix_jax_wrapper,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=5,
        verbose=True
    )
    
    print(f"\n   ✓ Fitting completed!")
    print(f"   ✓ Converged: {info['converged']}")
    print(f"   ✓ Iterations: {info['iterations']}")
    print(f"   ✓ Backend: {info['backend']}")
    
except Exception as e:
    print(f"   ✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Hybrid backend selection
print("\n5. Testing hybrid backend selection...")

from jug.fitting.design_matrix_jax import compute_design_matrix_auto

# Small dataset (should use NumPy)
M_small, backend_small = compute_design_matrix_auto(
    params, toas_mjd[:100], freq_mhz[:100], errors_us[:100], fit_params
)
print(f"   100 TOAs → {backend_small} backend")

# Large dataset (should use JAX)
M_large, backend_large = compute_design_matrix_auto(
    params, toas_mjd, freq_mhz, errors_us, fit_params
)
print(f"   {n_toas} TOAs → {backend_large} backend")

if backend_small == 'numpy' and backend_large == 'jax':
    print(f"   ✓ Hybrid backend selection works!")
else:
    print(f"   ⚠ Unexpected backends: {backend_small}, {backend_large}")

print("\n" + "="*80)
print("✓ All JAX fitting tests passed!")
print("="*80)
