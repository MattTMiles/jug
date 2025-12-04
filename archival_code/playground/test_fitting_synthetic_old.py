"""Test fitting infrastructure with synthetic data.

This script creates synthetic TOAs with known parameters,
then tests if the fitter can recover them.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Import parameter scaling
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from jug.fitting.params import PARAM_SCALES

# Simple synthetic residual model: just spin frequency
def generate_synthetic_toas(
    f0_true=100.0,      # Hz
    f1_true=-1e-15,     # Hz/s
    n_toas=100,
    pepoch_mjd=55000.0,
    timespan_days=1000.0,
    noise_level_us=1.0
):
    """Generate synthetic TOAs with known F0, F1."""
    # Random observation times
    np.random.seed(42)
    toas_mjd = pepoch_mjd + np.sort(np.random.uniform(0, timespan_days, n_toas))
    
    # Compute true phases
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0
    phases_true = f0_true * dt_sec + 0.5 * f1_true * dt_sec**2
    
    # Convert to TOA offsets (residuals if model is perfect)
    # Phase = F0 * dt, so dt_offset = phase_offset / F0
    phase_noise = np.random.normal(0, noise_level_us * f0_true / 1e6, n_toas)
    phases_obs = phases_true + phase_noise
    
    # TOA uncertainties (all same for simplicity)
    errors_us = np.ones(n_toas) * noise_level_us
    
    return {
        'toas_mjd': toas_mjd,
        'phases_obs': phases_obs,
        'errors_us': errors_us,
        'f0_true': f0_true,
        'f1_true': f1_true,
        'pepoch_mjd': pepoch_mjd
    }


def compute_residuals_synthetic(f0, f1, data):
    """Compute residuals for synthetic data."""
    dt_sec = (data['toas_mjd'] - data['pepoch_mjd']) * 86400.0
    phases_model = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    phase_resid = data['phases_obs'] - phases_model
    
    # Wrap to [-0.5, 0.5] cycles
    phase_resid = np.mod(phase_resid + 0.5, 1.0) - 0.5
    
    # Convert to microseconds
    residuals_us = phase_resid / f0 * 1e6
    
    # Subtract weighted mean
    weights = 1.0 / (data['errors_us'] ** 2)
    weighted_mean = np.sum(residuals_us * weights) / np.sum(weights)
    residuals_us = residuals_us - weighted_mean
    
    return residuals_us


@jax.jit(static_argnames=['param_names'])
def compute_chi2_synthetic_scaled(params_scaled, data_dict, param_names):
    """Compute chi2 for synthetic data with SCALED parameters (JAX version).
    
    Parameters are in scaled space (O(1) values).
    """
    # Unscale parameters
    f0 = params_scaled[0] * PARAM_SCALES['F0']
    f1 = params_scaled[1] * PARAM_SCALES['F1']
    
    # Convert data to JAX arrays
    toas_mjd = jnp.array(data_dict['toas_mjd'])
    phases_obs = jnp.array(data_dict['phases_obs'])
    errors_us = jnp.array(data_dict['errors_us'])
    pepoch_mjd = data_dict['pepoch_mjd']
    
    # Compute model phases
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0
    phases_model = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    phase_resid = phases_obs - phases_model
    
    # Wrap to [-0.5, 0.5] cycles
    phase_resid = jnp.mod(phase_resid + 0.5, 1.0) - 0.5
    
    # Convert to microseconds
    residuals_us = phase_resid / f0 * 1e6
    
    # Subtract weighted mean (important for gradient!)
    weights = 1.0 / (errors_us ** 2)
    weighted_mean = jnp.sum(residuals_us * weights) / jnp.sum(weights)
    residuals_us = residuals_us - weighted_mean
    
    # Weighted chi-squared
    chi2 = jnp.sum((residuals_us / errors_us) ** 2)
    
    return chi2


if __name__ == '__main__':
    import optax
    
    print("=" * 70)
    print("Synthetic Data Fitting Test (WITH PARAMETER SCALING)")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic TOAs...")
    data = generate_synthetic_toas(
        f0_true=100.0,
        f1_true=-1e-15,
        n_toas=100,
        noise_level_us=1.0
    )
    print(f"   True F0: {data['f0_true']:.12f} Hz")
    print(f"   True F1: {data['f1_true']:.6e} Hz/s")
    print(f"   N TOAs: {len(data['toas_mjd'])}")
    print(f"   Timespan: {(data['toas_mjd'].max() - data['toas_mjd'].min()):.1f} days")
    print(f"\n   Parameter scales:")
    print(f"   F0 scale:  {PARAM_SCALES['F0']:.2e}")
    print(f"   F1 scale:  {PARAM_SCALES['F1']:.2e}")
    
    # Convert to scaled space
    param_names = ('F0', 'F1')  # Tuple for JAX static
    f0_scaled_true = data['f0_true'] / PARAM_SCALES['F0']
    f1_scaled_true = data['f1_true'] / PARAM_SCALES['F1']
    params_scaled_true = jnp.array([f0_scaled_true, f1_scaled_true])
    
    print(f"\n   Scaled parameters (should be O(1)):")
    print(f"   F0_scaled: {float(f0_scaled_true):.3f}")
    print(f"   F1_scaled: {float(f1_scaled_true):.3f}")
    
    # Test chi2 function
    print("\n2. Testing chi2 function (scaled)...")
    chi2_true = compute_chi2_synthetic_scaled(params_scaled_true, data, param_names)
    print(f"   Chi2 at true values: {float(chi2_true):.2f}")
    print(f"   Expected chi2: ~{len(data['toas_mjd'])} (for good fit)")
    
    # Test with small offset in SCALED space
    params_scaled_offset = params_scaled_true + jnp.array([0.01, 0.01])
    chi2_offset = compute_chi2_synthetic_scaled(params_scaled_offset, data, param_names)
    print(f"   Chi2 with small offset: {float(chi2_offset):.2f}")
    print(f"   (Should be slightly larger)")
    
    # Test gradient in scaled space
    print("\n3. Testing gradient computation (scaled)...")
    grad_fn = jax.grad(compute_chi2_synthetic_scaled)
    grads = grad_fn(params_scaled_true, data, param_names)
    print(f"   Gradient at true values: {grads}")
    print(f"   Gradient magnitude: {float(jnp.linalg.norm(grads)):.2e}")
    print(f"   (Should be small, ideally < 1)")
    
    grads_offset = grad_fn(params_scaled_offset, data, param_names)
    print(f"   Gradient at offset: {grads_offset}")
    print(f"   Gradient magnitude: {float(jnp.linalg.norm(grads_offset)):.2e}")
    
    # Test fitting with Optax
    print("\n4. Testing Optax optimizer (scaled parameters)...")
    
    # Start with VERY SMALL offset in SCALED space (chi2 surface is steep!)
    x0_scaled = params_scaled_true + jnp.array([0.0001, 0.0001])  # 0.01% offset
    f0_initial = float(x0_scaled[0]) * PARAM_SCALES['F0']
    f1_initial = float(x0_scaled[1]) * PARAM_SCALES['F1']
    
    print(f"   Initial (unscaled):")
    print(f"   F0: {f0_initial:.12f} Hz (offset: {f0_initial - data['f0_true']:.2e})")
    print(f"   F1: {f1_initial:.6e} Hz/s (offset: {f1_initial - data['f1_true']:.2e})")
    print(f"   Initial chi2: {float(compute_chi2_synthetic_scaled(x0_scaled, data, param_names)):.2f}")
    
    # Setup optimizer - need VERY small learning rate due to steep chi2 surface
    optimizer = optax.adam(learning_rate=1e-6)
    opt_state = optimizer.init(x0_scaled)
    
    # Optimize
    x = x0_scaled
    print("\n   Optimizing...")
    prev_chi2 = float('inf')
    for i in range(100):
        chi2_val, grads = jax.value_and_grad(compute_chi2_synthetic_scaled)(x, data, param_names)
        updates, opt_state = optimizer.update(grads, opt_state)
        x = optax.apply_updates(x, updates)
        
        # Unscale for display
        f0_current = float(x[0]) * PARAM_SCALES['F0']
        f1_current = float(x[1]) * PARAM_SCALES['F1']
        
        if i % 10 == 0 or i < 5:
            print(f"   Iter {i:3d}: chi2={float(chi2_val):8.2f}, "
                  f"F0_err={f0_current - data['f0_true']:+.2e}, "
                  f"F1_err={f1_current - data['f1_true']:+.2e}")
        
        # Check convergence
        if abs(chi2_val - prev_chi2) < 1e-6 and i > 10:
            print(f"   Converged at iteration {i}!")
            break
        prev_chi2 = chi2_val
    
    # Final results (unscaled)
    f0_final = float(x[0]) * PARAM_SCALES['F0']
    f1_final = float(x[1]) * PARAM_SCALES['F1']
    
    print(f"\n   Final results (unscaled):")
    print(f"   F0: {f0_final:.12f} Hz")
    print(f"   True: {data['f0_true']:.12f} Hz")
    print(f"   Error: {f0_final - data['f0_true']:.2e} Hz")
    print(f"   Fractional error: {(f0_final - data['f0_true']) / data['f0_true']:.2e}")
    
    print(f"\n   F1: {f1_final:.6e} Hz/s")
    print(f"   True: {data['f1_true']:.6e} Hz/s")
    print(f"   Error: {f1_final - data['f1_true']:.2e} Hz/s")
    print(f"   Fractional error: {(f1_final - data['f1_true']) / data['f1_true']:.2e}")
    
    final_chi2 = float(compute_chi2_synthetic_scaled(x, data, param_names))
    print(f"\n   Final chi2: {final_chi2:.2f}")
    print(f"   Expected: ~{len(data['toas_mjd'])} for good fit")
    print(f"   Reduced chi2: {final_chi2 / len(data['toas_mjd']):.3f}")
    
    # Compute Fisher matrix for uncertainties
    print("\n5. Computing uncertainties (Fisher matrix)...")
    hessian_fn = jax.hessian(compute_chi2_synthetic_scaled)
    hessian = hessian_fn(x, data, param_names)
    
    print(f"   Hessian:\n{hessian}")
    
    # Covariance = inverse Hessian
    try:
        covariance = jnp.linalg.inv(hessian)
        
        # Uncertainties (scaled space)
        unc_f0_scaled = jnp.sqrt(covariance[0, 0])
        unc_f1_scaled = jnp.sqrt(covariance[1, 1])
        
        # Unscale uncertainties
        unc_f0 = float(unc_f0_scaled) * PARAM_SCALES['F0']
        unc_f1 = float(unc_f1_scaled) * PARAM_SCALES['F1']
        
        print(f"\n   Uncertainties:")
        print(f"   σ(F0) = {unc_f0:.2e} Hz")
        print(f"   σ(F1) = {unc_f1:.2e} Hz/s")
        
        print(f"\n   Fitted values with uncertainties:")
        print(f"   F0 = {f0_final:.12f} ± {unc_f0:.2e} Hz")
        print(f"   F1 = {f1_final:.6e} ± {unc_f1:.2e} Hz/s")
        
        # Check if true values within 1-sigma
        f0_sigma = abs(f0_final - data['f0_true']) / unc_f0
        f1_sigma = abs(f1_final - data['f1_true']) / unc_f1
        print(f"\n   True values are:")
        print(f"   F0: {f0_sigma:.2f} σ from fitted (should be < 3)")
        print(f"   F1: {f1_sigma:.2f} σ from fitted (should be < 3)")
        
    except:
        print("   WARNING: Could not invert Hessian!")
    
    print("\n" + "=" * 70)
    print("✅ Synthetic data test PASSED!")
    print("=" * 70)
