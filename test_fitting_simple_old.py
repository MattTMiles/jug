"""Test JAX fitting with proper timing residuals computation."""

import numpy as np
import sys

print("="*80)
print("Testing JAX Fitting with Analytical Timing Model")
print("="*80)

# Constants
SECS_PER_DAY = 86400.0
K_DM_SEC = 4.148808e3

np.random.seed(42)

# Observation parameters
n_toas = 500
mjd_start = 55000.0
mjd_end = 56000.0
toas_mjd = np.sort(np.random.uniform(mjd_start, mjd_end, n_toas))
freq_mhz = np.random.uniform(1200, 1600, n_toas)
errors_us = np.random.uniform(0.8, 1.2, n_toas)

# True timing model
true_params = {
    'F0': 100.0,
    'F1': -1.0e-15,
    'DM': 30.0,
    'PEPOCH': 55500.0,
    'DMEPOCH': 55500.0
}

print(f"\nGenerating synthetic TOAs with:")
print(f"  F0 = {true_params['F0']:.6f} Hz")
print(f"  F1 = {true_params['F1']:.6e} Hz/s")
print(f"  DM = {true_params['DM']:.2f} pc/cm³")
print(f"  N_TOAs = {n_toas}")
print(f"  MJD range: {mjd_start:.1f} - {mjd_end:.1f}")

# Generate "observed" TOAs with true model + noise
def compute_phase_residuals(params, toas, freqs):
    """Compute timing residuals from model."""
    dt_sec = (toas - params['PEPOCH']) * SECS_PER_DAY
    
    # Phase from spin model (using Horner's method)
    f0 = params['F0']
    f1 = params.get('F1', 0.0)
    phase = f0 * dt_sec + 0.5 * f1 * dt_sec**2
    
    # Convert phase to time (fractional phase / frequency)
    phase_frac = phase - np.round(phase)
    time_from_spin_sec = phase_frac / f0
    
    # DM delay (makes pulse arrive later, positive delay)
    dm = params['DM']
    dm_delay_sec = K_DM_SEC * dm / (freqs**2)
    
    # Total residual: spin phase residual + DM delay
    # (Both contribute to observed arrival time)
    residual_sec = time_from_spin_sec + dm_delay_sec
    
    # Convert to microseconds
    return residual_sec * 1e6

# Generate synthetic observed residuals (true model + noise)
true_residuals = compute_phase_residuals(true_params, toas_mjd, freq_mhz)
measurement_noise = np.random.randn(n_toas) * errors_us
observed_residuals = true_residuals + measurement_noise

print(f"\nObserved residuals:")
print(f"  RMS: {np.std(observed_residuals):.2f} μs")
print(f"  Mean: {np.mean(observed_residuals):.2f} μs")

# Define residual function for fitting
def residuals_fn(params):
    """Compute model residuals relative to observations."""
    model_residuals = compute_phase_residuals(params, toas_mjd, freq_mhz)
    # Return difference: observed - model
    # When model matches truth, this should just be noise
    return observed_residuals - model_residuals

# Check that true parameters give noise-like residuals
check_res = residuals_fn(true_params)
print(f"\nResiduals with true parameters:")
print(f"  RMS: {np.std(check_res):.2f} μs (should ≈ error bars)")
print(f"  Mean: {np.mean(check_res):.2f} μs (should ≈ 0)")

# Test fitting
print(f"\n{'='*80}")
print(f"Fitting with JAX Backend")
print(f"{'='*80}")

from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax
from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper

# Start with perturbed parameters
initial_params = true_params.copy()
initial_params['F0'] = 100.001      # 1 mHz error
initial_params['F1'] = -1.1e-15     # 10% error  
initial_params['DM'] = 31.0         # 1 pc/cm³ error

fit_params = ['F0', 'F1', 'DM']

print(f"\nInitial parameter errors:")
print(f"  ΔF0 = {(initial_params['F0']-true_params['F0'])*1e3:.3f} mHz")
print(f"  ΔF1 = {(initial_params['F1']-true_params['F1']):.2e} Hz/s")
print(f"  ΔDM = {(initial_params['DM']-true_params['DM']):.3f} pc/cm³")

# Initial residuals
init_res = residuals_fn(initial_params)
print(f"\nInitial residuals: RMS = {np.std(init_res):.2f} μs")

try:
    fitted, uncertainties, info = gauss_newton_fit_jax(
        residuals_fn,
        initial_params,
        fit_params,
        compute_design_matrix_jax_wrapper,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=15,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"Fit Results")
    print(f"{'='*80}")
    
    # Check recovery
    print(f"\nParameter Recovery (Fitted - True) / Uncertainty:")
    recovery_ok = True
    for param in fit_params:
        true_val = true_params[param]
        fitted_val = fitted[param]
        unc = uncertainties[param]
        
        diff = fitted_val - true_val
        n_sigma = abs(diff) / unc if unc > 0 else np.inf
        
        status = "✅" if n_sigma < 3.0 else "❌"
        
        if param == 'F0':
            print(f"  {param}: Δ = {diff*1e6:.3f} μHz, σ = {unc*1e6:.3f} μHz, pull = {n_sigma:.1f}σ {status}")
        elif param == 'F1':
            print(f"  {param}: Δ = {diff:.3e} Hz/s, σ = {unc:.3e} Hz/s, pull = {n_sigma:.1f}σ {status}")
        else:
            print(f"  {param}: Δ = {diff:.5f} pc/cm³, σ = {unc:.5f} pc/cm³, pull = {n_sigma:.1f}σ {status}")
        
        if n_sigma > 3.0:
            recovery_ok = False
    
    # Final residuals
    final_res = residuals_fn(fitted)
    print(f"\nFinal residuals:")
    print(f"  RMS: {np.std(final_res):.2f} μs")
    print(f"  Chi²/dof: {info['final_reduced_chi2']:.3f}")
    
    if recovery_ok:
        print(f"\n✅ SUCCESS: All parameters recovered within 3σ!")
    else:
        print(f"\n⚠️  Some parameters not recovered within 3σ")
        print(f"    (This may indicate fitting issues or unrealistic uncertainties)")
    
    if info['converged']:
        print(f"✅ Fit converged in {info['iterations']} iterations")
    else:
        print(f"⚠️  Fit did not converge in {info['iterations']} iterations")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n{'='*80}")
print(f"Test Complete")
print(f"{'='*80}")
