"""Synthetic JUMP Parameter Integration Test.

This test verifies end-to-end JUMP fitting using synthetic data.
We create TOAs from two "observatories" with a known offset and verify
that the fitter recovers the correct JUMP value.

Uses JAX for all computations.
"""

import jax
jax.config.update("jax_enable_x64", True)  # Enable float64

import jax.numpy as jnp
import numpy as np
from typing import Dict

from jug.fitting.derivatives_jump import (
    compute_jump_derivatives,
    create_jump_mask_from_mjd_range,
)
from jug.fitting.derivatives_spin import compute_spin_derivatives
from jug.fitting.wls_fitter import wls_solve_svd
from jug.model.parameter_spec import is_jump_param


def create_synthetic_jump_data(
    n_toas_per_backend: int = 50,
    f0: float = 300.0,
    jump_value_us: float = 5.0,  # True JUMP in microseconds
    noise_rms_us: float = 1.0,
) -> Dict:
    """Create synthetic TOAs with a known JUMP offset.
    
    Creates TOAs from two "backends":
    - Backend A: MJD 58000-58100 (no offset)
    - Backend B: MJD 58100-58200 (offset by JUMP)
    
    Parameters
    ----------
    n_toas_per_backend : int
        Number of TOAs per backend
    f0 : float
        Spin frequency in Hz
    jump_value_us : float
        True JUMP offset in microseconds
    noise_rms_us : float
        RMS of Gaussian noise to add
        
    Returns
    -------
    data : dict
        Dictionary with TOAs, errors, residuals, params, and masks
    """
    np.random.seed(42)  # Reproducibility
    
    # Create TOA times for two backends
    mjd_a = np.linspace(58000, 58100, n_toas_per_backend)
    mjd_b = np.linspace(58100.1, 58200, n_toas_per_backend)
    toas_mjd = jnp.concatenate([jnp.array(mjd_a), jnp.array(mjd_b)])
    
    n_toas = len(toas_mjd)
    
    # Errors (uniform for simplicity)
    errors_us = jnp.ones(n_toas) * noise_rms_us
    
    # Create backend masks
    mask_a = jnp.concatenate([
        jnp.ones(n_toas_per_backend, dtype=bool),
        jnp.zeros(n_toas_per_backend, dtype=bool)
    ])
    mask_b = ~mask_a
    
    # True residuals: backend B has an offset
    # We inject the JUMP so that fitting should recover -jump_value_us
    true_residuals = jnp.zeros(n_toas)
    true_residuals = true_residuals.at[n_toas_per_backend:].set(jump_value_us)
    
    # Add noise
    noise = jnp.array(np.random.normal(0, noise_rms_us, n_toas))
    observed_residuals_us = true_residuals + noise
    
    # Parameters
    params = {
        'F0': f0,
        'PEPOCH': 58100.0,
        'JUMP1': 0.0,  # Initial guess (unknown)
    }
    
    jump_masks = {
        'JUMP1': mask_b,  # JUMP applies to backend B
    }
    
    return {
        'toas_mjd': toas_mjd,
        'errors_us': errors_us,
        'residuals_us': observed_residuals_us,
        'params': params,
        'jump_masks': jump_masks,
        'true_jump_us': jump_value_us,
        'mask_a': mask_a,
        'mask_b': mask_b,
    }


def fit_jump_only(data: Dict) -> Dict:
    """Fit JUMP parameter using WLS.
    
    Parameters
    ----------
    data : dict
        Output from create_synthetic_jump_data()
        
    Returns
    -------
    result : dict
        Fitting result with fitted JUMP value
    """
    toas_mjd = data['toas_mjd']
    errors_us = data['errors_us']
    residuals_us = data['residuals_us']
    params = data['params']
    jump_masks = data['jump_masks']
    
    # Compute JUMP derivatives
    derivs = compute_jump_derivatives(
        params=params,
        toas_mjd=toas_mjd,
        fit_params=['JUMP1'],
        jump_masks=jump_masks,
    )
    
    # Build design matrix (single column for JUMP1)
    M = derivs['JUMP1'].reshape(-1, 1)
    
    # Convert to seconds for WLS solver
    residuals_sec = np.array(residuals_us) * 1e-6
    errors_sec = np.array(errors_us) * 1e-6
    M_np = np.array(M)
    
    # Solve WLS
    delta_params, cov, _ = wls_solve_svd(residuals_sec, errors_sec, M_np)
    
    # Convert back to microseconds
    fitted_jump_us = delta_params[0] * 1e6
    uncertainty_us = np.sqrt(cov[0, 0]) * 1e6
    
    return {
        'fitted_jump_us': fitted_jump_us,
        'uncertainty_us': uncertainty_us,
        'true_jump_us': data['true_jump_us'],
        'residual_before_us': float(jnp.std(residuals_us)),
    }


def run_synthetic_jump_test():
    """Run the synthetic JUMP fitting test."""
    print("=" * 70)
    print("Synthetic JUMP Parameter Integration Test")
    print("=" * 70)
    
    # Create synthetic data with known JUMP
    true_jump_us = 5.0
    data = create_synthetic_jump_data(
        n_toas_per_backend=50,
        jump_value_us=true_jump_us,
        noise_rms_us=1.0,
    )
    
    print(f"\nTest setup:")
    print(f"  TOAs: {len(data['toas_mjd'])} total (50 per backend)")
    print(f"  True JUMP: {true_jump_us:.3f} μs")
    print(f"  Noise RMS: 1.0 μs")
    
    # Fit JUMP
    result = fit_jump_only(data)
    
    print(f"\nFitting results:")
    print(f"  Fitted JUMP: {result['fitted_jump_us']:.3f} ± {result['uncertainty_us']:.3f} μs")
    print(f"  True JUMP:   {result['true_jump_us']:.3f} μs")
    print(f"  Error:       {abs(result['fitted_jump_us'] - result['true_jump_us']):.3f} μs")
    
    # Check if fit recovered the correct value within 3-sigma
    error = abs(result['fitted_jump_us'] - result['true_jump_us'])
    success = error < 3 * result['uncertainty_us']
    
    if success:
        print(f"\n✓ JUMP recovered within 3σ uncertainty!")
    else:
        print(f"\n✗ JUMP recovery failed (error > 3σ)")
    
    # Additional verification: residuals should improve
    mask_b = data['mask_b']
    residuals_before = data['residuals_us']
    residuals_after = residuals_before - result['fitted_jump_us'] * jnp.array(mask_b, dtype=jnp.float64)
    
    rms_before = float(jnp.std(residuals_before))
    rms_after = float(jnp.std(residuals_after))
    
    print(f"\nResidual improvement:")
    print(f"  RMS before JUMP correction: {rms_before:.3f} μs")
    print(f"  RMS after JUMP correction:  {rms_after:.3f} μs")
    print(f"  Improvement: {(rms_before - rms_after) / rms_before * 100:.1f}%")
    
    assert success, f"JUMP fitting failed: error {error:.3f} > 3σ ({3*result['uncertainty_us']:.3f})"
    assert rms_after < rms_before, "RMS should improve after JUMP correction"
    
    print("\n" + "=" * 70)
    print("✓ All synthetic JUMP tests passed!")
    print("=" * 70)
    
    return result


if __name__ == '__main__':
    run_synthetic_jump_test()
