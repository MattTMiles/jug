"""Simple JAX fitting test on J1909-3744."""

import numpy as np
from pathlib import Path
import sys

print("="*80)
print("JAX Fitting Test - J1909-3744")
print("="*80)

# Load data
par_file = "data/pulsars/J1909-3744_tdb.par"
tim_file = "data/pulsars/J1909-3744.tim"

if not Path(par_file).exists() or not Path(tim_file).exists():
    print(f"❌ Data files not found")
    sys.exit(1)

try:
    from jug.residuals.simple_calculator import compute_residuals_simple
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    
    # Get initial residuals
    print(f"\nComputing initial residuals...")
    result = compute_residuals_simple(par_file, tim_file)
    
    print(f"\nInitial solution (from .par file):")
    print(f"  N_TOAs: {result['n_toas']}")
    print(f"  RMS: {result['rms_us']:.3f} μs")
    print(f"  Mean: {result['mean_us']:.3f} μs")
    
    # Parse for fitting
    params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    
    toas_mjd = np.array([t.mjd_int + t.mjd_frac for t in toas])
    freq_mhz = np.array([t.freq_mhz for t in toas])
    errors_us = np.array([t.error_us for t in toas])
    
    # Store true values
    true_f0 = float(params['F0'])
    true_f1 = float(params['F1'])
    
    print(f"\nTrue parameters:")
    print(f"  F0 = {true_f0:.15e} Hz")
    print(f"  F1 = {true_f1:.15e} Hz/s")
    
    # Perturb parameters
    perturbed_params = dict(params)
    perturbed_params['F0'] = true_f0 * (1.0 + 1e-8)  # 10 ppb ~ 3.3 μHz
    perturbed_params['F1'] = true_f1 * (1.0 + 1e-4)  # 0.01%
    
    print(f"\nPerturbed parameters:")
    print(f"  ΔF0 = {(perturbed_params['F0'] - true_f0) * 1e6:.3f} μHz")
    print(f"  ΔF1 = {(perturbed_params['F1'] - true_f1):.3e} Hz/s")
    
    # Define residual function for fitting
    def compute_residuals_for_fit(params_dict):
        """Compute residuals for given parameters."""
        import tempfile
        import os
        
        # Write temporary par file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
            temp_par = f.name
            for key, val in params_dict.items():
                if isinstance(val, (int, float, np.number)):
                    f.write(f"{key} {val:.18e}\n")
                else:
                    f.write(f"{key} {val}\n")
        
        try:
            result = compute_residuals_simple(temp_par, tim_file)
            return result['residuals_us'] / 1e6  # Convert to seconds
        finally:
            os.unlink(temp_par)
    
    # Test perturbed residuals
    perturbed_res = compute_residuals_for_fit(perturbed_params) * 1e6
    print(f"\nPerturbed RMS: {np.sqrt(np.mean(perturbed_res**2)):.3f} μs")
    
    # Now fit
    print(f"\n{'='*80}")
    print("Starting JAX Fit")
    print(f"{'='*80}")
    
    from jug.fitting.gauss_newton_jax import gauss_newton_fit_auto
    
    fit_params = ['F0', 'F1']
    
    fitted, uncertainties, info = gauss_newton_fit_auto(
        compute_residuals_for_fit,
        perturbed_params,
        fit_params,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=10,
        verbose=True
    )
    
    # Results
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    
    print(f"\nF0:")
    print(f"  True:    {true_f0:.15e} Hz")
    print(f"  Fitted:  {fitted['F0']:.15e} ± {uncertainties['F0']:.3e} Hz")
    print(f"  Δ = {(fitted['F0'] - true_f0) * 1e9:.3f} nHz ({abs(fitted['F0'] - true_f0) / uncertainties['F0']:.1f}σ)")
    
    print(f"\nF1:")
    print(f"  True:    {true_f1:.15e} Hz/s")
    print(f"  Fitted:  {fitted['F1']:.15e} ± {uncertainties['F1']:.3e} Hz/s")
    print(f"  Δ = {(fitted['F1'] - true_f1):.3e} Hz/s ({abs(fitted['F1'] - true_f1) / uncertainties['F1']:.1f}σ)")
    
    final_res = compute_residuals_for_fit(fitted) * 1e6
    final_rms = np.sqrt(np.mean(final_res**2))
    
    print(f"\nRMS:")
    print(f"  Initial:   {result['rms_us']:.3f} μs")
    print(f"  Perturbed: {np.sqrt(np.mean(perturbed_res**2)):.3f} μs")
    print(f"  Fitted:    {final_rms:.3f} μs")
    
    print(f"\nFit status:")
    print(f"  Converged: {'✅ Yes' if info['converged'] else '❌ No'}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Chi²/dof: {info['final_reduced_chi2']:.3f}")
    
    # Check success
    f0_ok = abs(fitted['F0'] - true_f0) / uncertainties['F0'] < 3.0
    f1_ok = abs(fitted['F1'] - true_f1) / uncertainties['F1'] < 3.0
    
    print(f"\n{'='*80}")
    if f0_ok and f1_ok and info['converged']:
        print("✅ TEST PASSED - Fitting infrastructure working!")
    else:
        print("⚠️  TEST INCOMPLETE")
        if not f0_ok:
            print("   F0 not recovered within 3σ")
        if not f1_ok:
            print("   F1 not recovered within 3σ")
        if not info['converged']:
            print("   Fit did not converge")
    print(f"{'='*80}")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
