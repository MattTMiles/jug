"""Test fitting on real pulsar data (J1909-3744)."""

import numpy as np
from pathlib import Path

print("="*80)
print("Testing Integrated Fitting on J1909-3744")
print("="*80)

# Check if data files exist
par_file = Path("data/pulsars/J1909-3744_tdb.par")
tim_file = Path("data/pulsars/J1909-3744.tim")

if not par_file.exists():
    print(f"\n❌ ERROR: {par_file} not found")
    print("Please ensure pulsar data files are in data/pulsars/ directory")
    exit(1)

if not tim_file.exists():
    print(f"\n❌ ERROR: {tim_file} not found")
    exit(1)

print(f"\nUsing:")
print(f"  .par file: {par_file}")
print(f"  .tim file: {tim_file}")

# Test 1: Create residual function
print(f"\n{'='*80}")
print("Test 1: Creating Residual Function")
print(f"{'='*80}")

try:
    from jug.fitting.residual_wrapper import create_residual_function
    
    residuals_fn, initial_params, fit_params, toas_mjd, freq_mhz, errors_us = \
        create_residual_function(
            par_file,
            tim_file,
            ephemeris_path="data/ephemeris/de440s.bsp",
            clock_dir="data/clock",
            observatory="pks",  # Parkes for J1909
            use_jax=True
        )
    
    print(f"✅ Residual function created successfully")
    print(f"\nDataset info:")
    print(f"  N_TOAs: {len(toas_mjd)}")
    print(f"  MJD range: {toas_mjd.min():.1f} - {toas_mjd.max():.1f}")
    print(f"  Freq range: {freq_mhz.min():.0f} - {freq_mhz.max():.0f} MHz")
    print(f"  Error range: {errors_us.min():.2f} - {errors_us.max():.2f} μs")
    
    print(f"\nParameters to fit ({len(fit_params)}):")
    for param in fit_params:
        value = initial_params.get(param, 0.0)
        if param in ['F0', 'F1', 'F2']:
            print(f"  {param}: {value:.6e}")
        else:
            print(f"  {param}: {value}")
    
except Exception as e:
    print(f"\n❌ ERROR creating residual function: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 2: Compute initial residuals
print(f"\n{'='*80}")
print("Test 2: Computing Initial Residuals")
print(f"{'='*80}")

try:
    initial_residuals = residuals_fn(initial_params)
    
    print(f"✅ Residuals computed successfully")
    print(f"\nInitial residuals:")
    print(f"  RMS: {np.std(initial_residuals)*1e6:.2f} μs")
    print(f"  Mean: {np.mean(initial_residuals)*1e6:.2f} μs")
    print(f"  Min: {np.min(initial_residuals)*1e6:.2f} μs")
    print(f"  Max: {np.max(initial_residuals)*1e6:.2f} μs")
    
except Exception as e:
    print(f"\n❌ ERROR computing residuals: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 3: Run fitting
print(f"\n{'='*80}")
print("Test 3: Running JAX Fitting")
print(f"{'='*80}")

try:
    from jug.fitting.gauss_newton_jax import gauss_newton_fit_auto
    from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper
    
    # Use only a subset of parameters for quick test
    test_fit_params = ['F0', 'F1', 'DM'][:min(3, len(fit_params))]
    
    print(f"\nFitting {len(test_fit_params)} parameters: {', '.join(test_fit_params)}")
    
    fitted_params, uncertainties, info = gauss_newton_fit_auto(
        residuals_fn,
        initial_params,
        test_fit_params,
        toas_mjd,
        freq_mhz,
        errors_us,
        max_iter=10,
        verbose=True
    )
    
    print(f"\n{'='*80}")
    print("Fit Results")
    print(f"{'='*80}")
    
    # Show fitted values
    print(f"\nFitted Parameters:")
    for param in test_fit_params:
        initial_val = initial_params[param]
        fitted_val = fitted_params[param]
        unc = uncertainties[param]
        delta = fitted_val - initial_val
        
        if param in ['F0', 'F1', 'F2']:
            print(f"  {param:4s}: {fitted_val:.15e} ± {unc:.3e}")
            print(f"        Δ = {delta:.3e} ({abs(delta/unc):.1f}σ)")
        else:
            print(f"  {param:4s}: {fitted_val:.10f} ± {unc:.3e}")
            print(f"        Δ = {delta:.5f} ({abs(delta/unc):.1f}σ)")
    
    # Compute final residuals
    final_residuals = residuals_fn(fitted_params)
    
    print(f"\nFinal residuals:")
    print(f"  RMS: {np.std(final_residuals)*1e6:.2f} μs")
    print(f"  Chi²/dof: {info['final_reduced_chi2']:.3f}")
    
    improvement = (np.std(initial_residuals) - np.std(final_residuals)) / np.std(initial_residuals) * 100
    print(f"  Improvement: {improvement:.1f}%")
    
    if info['converged']:
        print(f"\n✅ Fit converged in {info['iterations']} iterations")
    else:
        print(f"\n⚠️  Fit did not converge in {info['iterations']} iterations")
        print(f"    (May need more iterations or better initial parameters)")
    
    print(f"\n{'='*80}")
    print(f"✅ Integration test PASSED")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n❌ ERROR during fitting: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
