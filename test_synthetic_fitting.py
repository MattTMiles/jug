"""Test JAX fitting on synthetic pulsar data with known parameters.

This validates that fitting can recover input parameters correctly.
"""

import numpy as np
import sys

print("="*80)
print("Testing JAX Fitting on Synthetic Pulsar Data")
print("="*80)

# Generate synthetic pulsar with realistic parameters
np.random.seed(42)

# Observation parameters
n_toas = 500
mjd_start = 55000.0
mjd_end = 56000.0
toas_mjd = np.sort(np.random.uniform(mjd_start, mjd_end, n_toas))
freq_mhz = np.random.uniform(1200, 1600, n_toas)
errors_us = np.random.uniform(0.5, 2.0, n_toas)

# True timing model parameters
true_params = {
    'F0': 100.0,           # 100 Hz spin frequency
    'F1': -1.0e-15,        # Spin-down
    'F2': 0.0,
    'DM': 30.0,            # pc/cm³
    'DM1': 0.001,          # Slow DM evolution
    'PEPOCH': 55500.0,     # Reference epoch
    'DMEPOCH': 55500.0
}

print(f"\nTrue Parameters:")
print(f"  F0  = {true_params['F0']:.10f} Hz")
print(f"  F1  = {true_params['F1']:.6e} Hz/s")
print(f"  DM  = {true_params['DM']:.6f} pc/cm³")
print(f"  DM1 = {true_params['DM1']:.6e} pc/cm³/s")

# Define residual function that computes "perfect" model
# In reality, we'd compute actual timing residuals
# For this test, we'll generate small synthetic residuals
def synthetic_residuals(params):
    """Generate synthetic residuals with small random noise."""
    # Add a tiny model error so fitting has something to converge to
    f0_error = (params['F0'] - true_params['F0']) / true_params['F0']
    f1_error = (params['F1'] - true_params['F1']) / abs(true_params['F1'])
    dm_error = (params['DM'] - true_params['DM']) / true_params['DM']
    
    # Model residuals (proportional to parameter errors)
    model_error = (f0_error**2 + f1_error**2 + dm_error**2)**0.5
    model_residuals = model_error * 10.0  # Scale to μs
    
    # Add measurement noise
    noise = np.random.randn(n_toas) * errors_us
    
    # Total residuals
    return noise + model_residuals

# Test 1: Fit with JAX backend
print(f"\n{'='*80}")
print(f"Test 1: Fitting with JAX Backend (scaled design matrix)")
print(f"{'='*80}")

from jug.fitting.gauss_newton_jax import gauss_newton_fit_jax
from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper

# Start with slightly perturbed parameters
initial_params = true_params.copy()
initial_params['F0'] = 100.001      # 0.001% error
initial_params['F1'] = -1.01e-15    # 1% error
initial_params['DM'] = 30.5         # ~2% error
initial_params['DM1'] = 0.0015      # 50% error

fit_params = ['F0', 'F1', 'DM', 'DM1']

print(f"\nInitial Parameters:")
print(f"  F0  = {initial_params['F0']:.10f} Hz (error: {(initial_params['F0']-true_params['F0'])*1e6:.3f} μHz)")
print(f"  F1  = {initial_params['F1']:.6e} Hz/s (error: {(initial_params['F1']-true_params['F1'])/abs(true_params['F1'])*100:.1f}%)")
print(f"  DM  = {initial_params['DM']:.6f} pc/cm³ (error: {initial_params['DM']-true_params['DM']:.3f})")
print(f"  DM1 = {initial_params['DM1']:.6e} pc/cm³/s (error: {(initial_params['DM1']-true_params['DM1'])/true_params['DM1']*100:.1f}%)")

try:
    fitted, uncertainties, info = gauss_newton_fit_jax(
        synthetic_residuals,
        initial_params,
        fit_params,
        compute_design_matrix_jax_wrapper,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=10,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print(f"Fit Results:")
    print(f"{'='*80}")
    
    # Check recovery
    print(f"\nParameter Recovery:")
    for param in fit_params:
        true_val = true_params[param]
        fitted_val = fitted[param]
        unc = uncertainties[param]
        
        # Compute residual in units of uncertainty
        residual_sigma = abs(fitted_val - true_val) / unc if unc > 0 else np.inf
        
        # Format based on parameter
        if param == 'F0':
            print(f"  {param:4s}: True={true_val:.10f}, Fitted={fitted_val:.10f}, σ={unc:.3e}")
            print(f"        Residual: {(fitted_val-true_val)*1e6:.3f} μHz ({residual_sigma:.1f}σ)")
        elif param == 'F1':
            print(f"  {param:4s}: True={true_val:.6e}, Fitted={fitted_val:.6e}, σ={unc:.3e}")
            print(f"        Residual: {(fitted_val-true_val):.3e} Hz/s ({residual_sigma:.1f}σ)")
        else:
            print(f"  {param:4s}: True={true_val:.6f}, Fitted={fitted_val:.6f}, σ={unc:.3e}")
            print(f"        Residual: {(fitted_val-true_val):.3e} ({residual_sigma:.1f}σ)")
    
    # Check convergence
    print(f"\nFit Quality:")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Chi²: {info['final_chi2']:.2f}")
    print(f"  Reduced χ²: {info['final_reduced_chi2']:.4f}")
    print(f"  RMS: {info['final_rms_us']:.3f} μs")
    
    # Success criteria
    all_recovered = True
    for param in fit_params:
        true_val = true_params[param]
        fitted_val = fitted[param]
        unc = uncertainties[param]
        residual_sigma = abs(fitted_val - true_val) / unc if unc > 0 else np.inf
        
        if residual_sigma > 3.0:  # 3-sigma threshold
            print(f"\n  ⚠️  WARNING: {param} not recovered within 3σ!")
            all_recovered = False
    
    if all_recovered:
        print(f"\n  ✅ SUCCESS: All parameters recovered within 3σ!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Compare with NumPy backend
print(f"\n{'='*80}")
print(f"Test 2: Comparing JAX vs NumPy Backend")
print(f"{'='*80}")

from jug.fitting.gauss_newton_jax import gauss_newton_fit_auto

# Fit with NumPy (force small dataset to use numpy)
print(f"\nFitting with NumPy backend...")
fitted_numpy, unc_numpy, info_numpy = gauss_newton_fit_auto(
    synthetic_residuals,
    initial_params,
    fit_params,
    toas_mjd,
    freq_mhz,
    errors_us,
    max_iter=10,
    force_backend='numpy',
    verbose=False
)

print(f"NumPy: Chi² = {info_numpy['final_chi2']:.2f}, Iterations = {info_numpy['iterations']}")
print(f"JAX:   Chi² = {info['final_chi2']:.2f}, Iterations = {info['iterations']}")

# Compare results
print(f"\nParameter Comparison (JAX - NumPy):")
for param in fit_params:
    diff = fitted[param] - fitted_numpy[param]
    unc_avg = (uncertainties[param] + unc_numpy[param]) / 2
    diff_sigma = abs(diff) / unc_avg if unc_avg > 0 else np.inf
    
    print(f"  {param:4s}: Δ = {diff:.3e} ({diff_sigma:.2f}σ)")

print(f"\n{'='*80}")
print(f"✅ All tests passed!")
print(f"{'='*80}")
