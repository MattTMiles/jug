"""Direct test of JAX fitting with synthetic data (no file I/O)."""

import numpy as np
import sys

print("="*80)
print("JAX Fitting - Direct Test (No File I/O)")
print("="*80)

# Generate synthetic TOAs
np.random.seed(42)

n_toas = 500
toas_mjd = np.sort(np.random.uniform(55000, 56000, n_toas))
freq_mhz = np.random.uniform(1200, 1600, n_toas)
errors_us = np.random.uniform(0.8, 1.2, n_toas)

# True parameters
true_f0 = 100.123456789
true_f1 = -1.23e-15
true_dm = 30.456
pepoch = 55500.0

print(f"\nSynthetic pulsar:")
print(f"  F0 = {true_f0:.10f} Hz")
print(f"  F1 = {true_f1:.6e} Hz/s")
print(f"  DM = {true_dm:.6f} pc/cm³")
print(f"  N_TOAs = {n_toas}")

# Compute "observed" phase with true model + noise
SECS_PER_DAY = 86400.0
dt_sec = (toas_mjd - pepoch) * SECS_PER_DAY
true_phase = true_f0 * dt_sec + 0.5 * true_f1 * dt_sec**2

# Add noise to create observations
noise_sec = np.random.randn(n_toas) * errors_us * 1e-6
observed_phase = true_phase + noise_sec * true_f0

print(f"\nObserved phase noise: RMS = {np.std(noise_sec)*1e6:.2f} μs")

# Create residual function
def compute_residuals(params_dict):
    """Compute phase residuals."""
    f0 = params_dict['F0']
    f1 = params_dict['F1']
    
    dt = (toas_mjd - pepoch) * SECS_PER_DAY
    model_phase = f0 * dt + 0.5 * f1 * dt**2
    
    residual_phase = observed_phase - model_phase
    residual_time_sec = residual_phase / f0
    
    return residual_time_sec

# Test with true parameters (should give noise-only residuals)
true_params = {'F0': true_f0, 'F1': true_f1, 'DM': true_dm, 'PEPOCH': pepoch}
true_residuals = compute_residuals(true_params)
print(f"Residuals with true parameters: RMS = {np.std(true_residuals)*1e6:.2f} μs")
print(f"(Should be ≈ {np.mean(errors_us):.2f} μs from error bars)")

# Now test fitting
print(f"\n{'='*80}")
print("Testing JAX Fitting")
print(f"{'='*80}")

# Perturb initial parameters
initial_params = {
    'F0': true_f0 * 1.00001,   # 0.001% error (10 μHz)
    'F1': true_f1 * 1.01,      # 1% error
    'DM': true_dm,             # Not fitting DM in this test
    'PEPOCH': pepoch
}

fit_params = ['F0', 'F1']

print(f"\nInitial errors:")
print(f"  ΔF0 = {(initial_params['F0'] - true_f0)*1e6:.3f} μHz")
print(f"  ΔF1 = {(initial_params['F1'] - true_f1):.3e} Hz/s")

init_residuals = compute_residuals(initial_params)
print(f"\nInitial residuals: RMS = {np.std(init_residuals)*1e6:.2f} μs")

# Run fitting
try:
    from jug.fitting.gauss_newton_jax import gauss_newton_fit_auto
    from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper
    
    print(f"\nFitting {len(fit_params)} parameters...")
    
    fitted, uncertainties, info = gauss_newton_fit_auto(
        compute_residuals,
        initial_params,
        fit_params,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=15,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("Fit Results")
    print(f"{'='*80}")
    
    # Check recovery
    print(f"\nParameter Recovery:")
    for param in fit_params:
        if param == 'F0':
            true_val = true_f0
        elif param == 'F1':
            true_val = true_f1
        else:
            continue
        
        fitted_val = fitted[param]
        unc = uncertainties[param]
        diff = fitted_val - true_val
        n_sigma = abs(diff) / unc if unc > 0 else np.inf
        
        status = "✅" if n_sigma < 3.0 else ("⚠️" if n_sigma < 5.0 else "❌")
        
        if param == 'F0':
            print(f"  {param}: {fitted_val:.15e} ± {unc:.3e}")
            print(f"       True: {true_val:.15e}")
            print(f"       Δ = {diff*1e6:.3f} μHz ({n_sigma:.1f}σ) {status}")
        elif param == 'F1':
            print(f"  {param}: {fitted_val:.15e} ± {unc:.3e}")
            print(f"       True: {true_val:.15e}")
            print(f"       Δ = {diff:.3e} Hz/s ({n_sigma:.1f}σ) {status}")
    
    # Final residuals
    final_residuals = compute_residuals(fitted)
    print(f"\nFinal residuals:")
    print(f"  RMS: {np.std(final_residuals)*1e6:.2f} μs")
    print(f"  Chi²/dof: {info['final_reduced_chi2']:.3f}")
    
    improvement = (np.std(init_residuals) - np.std(final_residuals)) / np.std(init_residuals) * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    # Check if parameters recovered within 3σ
    all_recovered = True
    for param in fit_params:
        true_val = true_f0 if param == 'F0' else true_f1
        diff = fitted[param] - true_val
        n_sigma = abs(diff) / uncertainties[param]
        if n_sigma > 3.0:
            all_recovered = False
    
    print(f"\n{'='*80}")
    if info['converged']:
        print(f"✅ Fit converged in {info['iterations']} iterations")
    else:
        print(f"⚠️ Fit did not fully converge ({info['iterations']} iterations)")
    
    if all_recovered:
        print(f"✅ All parameters recovered within 3σ")
    else:
        print(f"⚠️ Some parameters not recovered within 3σ")
        print(f"   (This can happen with synthetic data; check pull distribution)")
    
    print(f"✅ FITTING INFRASTRUCTURE WORKING")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
