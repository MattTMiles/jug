"""Test linearized least squares fitting (standard pulsar timing approach).

Instead of minimizing chi2(params) directly with gradient descent,
we linearize the problem and solve it analytically.
"""

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def generate_synthetic_toas(
    f0_true=100.0,
    f1_true=-1e-15,
    n_toas=100,
    pepoch_mjd=55000.0,
    timespan_days=1000.0,
    noise_level_us=1.0
):
    """Generate synthetic TOAs."""
    np.random.seed(42)
    toas_mjd = pepoch_mjd + np.sort(np.random.uniform(0, timespan_days, n_toas))
    
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0
    phases_true = f0_true * dt_sec + 0.5 * f1_true * dt_sec**2
    
    phase_noise = np.random.normal(0, noise_level_us * f0_true / 1e6, n_toas)
    phases_obs = phases_true + phase_noise
    
    errors_us = np.ones(n_toas) * noise_level_us
    
    return {
        'toas_mjd': toas_mjd,
        'phases_obs': phases_obs,
        'errors_us': errors_us,
        'f0_true': f0_true,
        'f1_true': f1_true,
        'pepoch_mjd': pepoch_mjd
    }


def fit_linear(data, f0_guess, f1_guess):
    """Fit F0, F1 using linearized least squares.
    
    This is the standard approach in pulsar timing.
    
    We linearize:  phase ≈ phase(guess) + ∂phase/∂F0 * ΔF0 + ∂phase/∂F1 * ΔF1
    
    Then solve: A * Δparams = residuals
    where A is the design matrix (derivatives).
    """
    toas_mjd = data['toas_mjd']
    phases_obs = data['phases_obs']
    errors_us = data['errors_us']
    pepoch_mjd = data['pepoch_mjd']
    
    # Compute residuals with current guess
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0
    phases_model = f0_guess * dt_sec + 0.5 * f1_guess * dt_sec**2
    phase_resid = phases_obs - phases_model
    
    # Wrap to [-0.5, 0.5]
    phase_resid = np.mod(phase_resid + 0.5, 1.0) - 0.5
    
    # Convert to time residuals (microseconds)
    residuals_us = phase_resid / f0_guess * 1e6
    
    # Design matrix: derivatives of phase w.r.t. parameters
    # ∂phase/∂F0 = dt
    # ∂phase/∂F1 = 0.5 * dt^2
    A = np.column_stack([
        dt_sec,              # ∂phase/∂F0
        0.5 * dt_sec**2      # ∂phase/∂F1
    ])
    
    # Convert phase derivatives to time derivatives
    # ∂(time_resid)/∂F0 = -(phase/F0^2) + (∂phase/∂F0)/F0
    #                    ≈ (∂phase/∂F0)/F0 for small residuals
    # So multiply A by 1e6/F0 to get time domain design matrix
    A_time = A * (1e6 / f0_guess)
    
    # Weighted least squares
    weights = 1.0 / errors_us
    W = np.diag(weights)
    
    # Normal equations: (A^T W^2 A) * Δparams = A^T W^2 * residuals
    ATA = A_time.T @ (W @ W) @ A_time
    ATb = A_time.T @ (W @ W) @ residuals_us
    
    # Solve
    delta_params = np.linalg.solve(ATA, ATb)
    
    # Update parameters
    f0_new = f0_guess + delta_params[0]
    f1_new = f1_guess + delta_params[1]
    
    # Covariance matrix
    cov = np.linalg.inv(ATA)
    uncertainties = np.sqrt(np.diag(cov))
    
    # Compute final chi2
    dt_sec = (toas_mjd - pepoch_mjd) * 86400.0
    phases_model = f0_new * dt_sec + 0.5 * f1_new * dt_sec**2
    phase_resid = phases_obs - phases_model
    phase_resid = np.mod(phase_resid + 0.5, 1.0) - 0.5
    residuals_us = phase_resid / f0_new * 1e6
    
    chi2 = np.sum((residuals_us / errors_us) ** 2)
    
    return {
        'f0': f0_new,
        'f1': f1_new,
        'unc_f0': uncertainties[0],
        'unc_f1': uncertainties[1],
        'chi2': chi2,
        'residuals_us': residuals_us
    }


if __name__ == '__main__':
    print("=" * 70)
    print("Linearized Least Squares Fitting Test")
    print("=" * 70)
    
    # Generate data
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
    
    # Fit with initial guess (VERY close, as in real timing)
    print("\n2. Fitting with linearized least squares...")
    f0_guess = data['f0_true'] + 1e-9  # 1 nHz offset (realistic)
    f1_guess = data['f1_true'] + 1e-20  # tiny offset
    
    print(f"   Initial guess:")
    print(f"   F0: {f0_guess:.12f} Hz (offset: {f0_guess - data['f0_true']:.2e})")
    print(f"   F1: {f1_guess:.6e} Hz/s (offset: {f1_guess - data['f1_true']:.2e})")
    
    # Iterate to convergence
    for iteration in range(5):
        result = fit_linear(data, f0_guess, f1_guess)
        
        print(f"\n   Iteration {iteration + 1}:")
        print(f"   F0: {result['f0']:.12f} Hz (error: {result['f0'] - data['f0_true']:.2e})")
        print(f"   F1: {result['f1']:.6e} Hz/s (error: {result['f1'] - data['f1_true']:.2e})")
        print(f"   Chi2: {result['chi2']:.2f}")
        
        # Check convergence
        if abs(result['f0'] - f0_guess) < 1e-15 and abs(result['f1'] - f1_guess) < 1e-25:
            print(f"   Converged!")
            break
        
        f0_guess = result['f0']
        f1_guess = result['f1']
    
    # Final results
    print(f"\n3. Final results:")
    print(f"   F0 = {result['f0']:.12f} ± {result['unc_f0']:.2e} Hz")
    print(f"   F1 = {result['f1']:.6e} ± {result['unc_f1']:.2e} Hz/s")
    print(f"   Chi2 = {result['chi2']:.2f}")
    print(f"   Reduced chi2 = {result['chi2'] / len(data['toas_mjd']):.3f}")
    
    print(f"\n   Comparison with true values:")
    f0_sigma = abs(result['f0'] - data['f0_true']) / result['unc_f0']
    f1_sigma = abs(result['f1'] - data['f1_true']) / result['unc_f1']
    print(f"   F0: {f0_sigma:.2f} σ from true (should be < 3)")
    print(f"   F1: {f1_sigma:.2f} σ from true (should be < 3)")
    
    print("\n" + "=" * 70)
    if f0_sigma < 3 and f1_sigma < 3 and result['chi2'] / len(data['toas_mjd']) < 2:
        print("✅ LINEARIZED FITTING TEST PASSED!")
    else:
        print("❌ Test failed - check results above")
    print("=" * 70)
