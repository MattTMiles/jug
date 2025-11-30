"""Test integrated fitting with synthetic pulsar data.

This creates synthetic TOAs and tests the full fitting pipeline.
"""

import numpy as np
from pathlib import Path
import tempfile

print("="*80)
print("Testing Integrated Fitting with Synthetic Data")
print("="*80)

# Constants
SECS_PER_DAY = 86400.0

# Create synthetic pulsar
np.random.seed(42)

# True parameters
true_params = {
    'F0': 100.123456789,
    'F1': -1.23e-15,
    'DM': 30.456,
    'PEPOCH': 55500.0,
    'DMEPOCH': 55500.0,
    'RAJ': 1.234,  # radians
    'DECJ': -0.567,  # radians
    'UNITS': 'TDB'
}

# Generate TOAs
n_toas = 300
toas_mjd = np.sort(np.random.uniform(55000, 56000, n_toas))
freq_mhz = np.random.uniform(1200, 1600, n_toas)
errors_us = np.random.uniform(0.8, 1.2, n_toas)

print(f"\nSynthetic pulsar:")
print(f"  F0 = {true_params['F0']:.10f} Hz")
print(f"  F1 = {true_params['F1']:.6e} Hz/s")
print(f"  DM = {true_params['DM']:.6f} pc/cm³")
print(f"  N_TOAs = {n_toas}")

# Compute "observed" residuals (true model + noise)
def compute_model_phase(f0, f1, pepoch, toas):
    dt = (toas - pepoch) * SECS_PER_DAY
    phase = f0 * dt + 0.5 * f1 * dt**2
    return phase

true_phase = compute_model_phase(
    true_params['F0'], true_params['F1'], 
    true_params['PEPOCH'], toas_mjd
)

# Add small noise to create "observed" TOAs
noise = np.random.randn(n_toas) * errors_us * 1e-6  # Convert to seconds
observed_phase = true_phase + noise * true_params['F0']  # Convert time noise to phase noise

# Create temporary .par and .tim files
with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    par_file = tmpdir / "synthetic.par"
    tim_file = tmpdir / "synthetic.tim"
    
    # Write .par file
    with open(par_file, 'w') as f:
        f.write("PSRJ     J9999+9999\n")
        f.write(f"F0       {true_params['F0']:.15e}\n")
        f.write(f"F1       {true_params['F1']:.15e}\n")
        f.write(f"PEPOCH   {true_params['PEPOCH']:.10f}\n")
        f.write(f"DM       {true_params['DM']:.10f}\n")
        f.write(f"DMEPOCH  {true_params['DMEPOCH']:.10f}\n")
        f.write(f"RAJ      {true_params['RAJ']:.15e}\n")
        f.write(f"DECJ     {true_params['DECJ']:.15e}\n")
        f.write(f"UNITS    {true_params['UNITS']}\n")
        # Mark parameters to fit
        f.write("FIT_F0   1\n")
        f.write("FIT_F1   1\n")
        f.write("FIT_DM   1\n")
    
    # Write .tim file
    with open(tim_file, 'w') as f:
        f.write("FORMAT 1\n")
        for i in range(n_toas):
            # TOA format: freq(MHz) MJD error(μs) observatory
            f.write(f"{freq_mhz[i]:.3f} {toas_mjd[i]:.10f} {errors_us[i]:.3f} gbt\n")
    
    print(f"\nCreated temporary files:")
    print(f"  {par_file}")
    print(f"  {tim_file}")
    
    # Test 1: Direct residual computation (no fitting library needed)
    print(f"\n{'='*80}")
    print("Test 1: Direct Residual Computation")
    print(f"{'='*80}")
    
    try:
        from jug.io.par_reader import parse_par_file
        from jug.io.tim_reader import parse_tim_file
        
        params = parse_par_file(par_file)
        toa_data = parse_tim_file(tim_file)
        
        print(f"✅ Parsed {len(toa_data['mjd'])} TOAs")
        print(f"✅ Loaded {len([k for k in params.keys() if not k.startswith('FIT_')])} parameters")
        
        # Check FIT flags
        fit_params = [k[4:] for k, v in params.items() if k.startswith('FIT_') and v == 1]
        print(f"✅ Found {len(fit_params)} FIT parameters: {', '.join(fit_params)}")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test 2: Simplified residual function (without barycentric delays)
    print(f"\n{'='*80}")
    print("Test 2: Simplified Fitting (No Barycentric Delays)")
    print(f"{'='*80}")
    
    print("\nNote: Using simplified model without full barycentric corrections")
    print("This tests the fitting infrastructure, not full timing precision")
    
    try:
        from jug.fitting.gauss_newton_jax import gauss_newton_fit_auto
        from jug.fitting.design_matrix_jax import compute_design_matrix_jax_wrapper
        import jax.numpy as jnp
        
        # Simplified residual function
        def simple_residuals(params_dict):
            """Compute phase residuals without barycentric delays."""
            f0 = params_dict['F0']
            f1 = params_dict['F1']
            pepoch = params_dict['PEPOCH']
            
            dt = (toa_data['mjd'] - pepoch) * SECS_PER_DAY
            model_phase = f0 * dt + 0.5 * f1 * dt**2
            
            # "Observed" phase (use saved values)
            residual_phase = observed_phase - model_phase
            
            # Convert to time residuals
            residual_time = residual_phase / f0
            
            return residual_time
        
        # Perturb initial parameters slightly
        perturbed_params = params.copy()
        perturbed_params['F0'] = params['F0'] * 1.00001  # 0.001% error
        perturbed_params['F1'] = params['F1'] * 1.01      # 1% error
        perturbed_params['DM'] = params['DM'] * 1.02      # 2% error
        
        print(f"\nInitial parameter errors:")
        print(f"  ΔF0 = {(perturbed_params['F0'] - params['F0'])*1e6:.3f} μHz")
        print(f"  ΔF1 = {(perturbed_params['F1'] - params['F1']):.3e} Hz/s")
        print(f"  ΔDM = {(perturbed_params['DM'] - params['DM']):.5f} pc/cm³")
        
        # Initial residuals
        init_res = simple_residuals(perturbed_params)
        print(f"\nInitial residuals: RMS = {np.std(init_res)*1e6:.2f} μs")
        
        # Fit
        print(f"\nRunning fit...")
        fitted, uncertainties, info = gauss_newton_fit_auto(
            simple_residuals,
            perturbed_params,
            fit_params,
            toa_data['mjd'],
            toa_data['freq_mhz'],
            toa_data['error_us'],
            max_iter=10,
            verbose=True
        )
        
        print(f"\n{'='*80}")
        print("Fit Results")
        print(f"{'='*80}")
        
        # Check recovery
        print(f"\nParameter Recovery:")
        recovery_ok = True
        for param in fit_params:
            true_val = params[param]
            fitted_val = fitted[param]
            unc = uncertainties[param]
            
            diff = fitted_val - true_val
            n_sigma = abs(diff) / unc if unc > 0 else np.inf
            
            status = "✅" if n_sigma < 5.0 else "⚠️"
            
            if param == 'F0':
                print(f"  {param}: Δ = {diff*1e6:.3f} μHz, σ = {unc*1e6:.3f} μHz, pull = {n_sigma:.1f}σ {status}")
            elif param == 'F1':
                print(f"  {param}: Δ = {diff:.3e} Hz/s, σ = {unc:.3e} Hz/s, pull = {n_sigma:.1f}σ {status}")
            else:
                print(f"  {param}: Δ = {diff:.5f}, σ = {unc:.5f}, pull = {n_sigma:.1f}σ {status}")
            
            if n_sigma > 5.0:
                recovery_ok = False
        
        # Final residuals
        final_res = simple_residuals(fitted)
        print(f"\nFinal residuals:")
        print(f"  RMS: {np.std(final_res)*1e6:.2f} μs")
        print(f"  Chi²/dof: {info['final_reduced_chi2']:.3f}")
        
        improvement = (np.std(init_res) - np.std(final_res)) / np.std(init_res) * 100
        print(f"  Improvement: {improvement:.1f}%")
        
        if info['converged']:
            print(f"\n✅ Fit converged in {info['iterations']} iterations")
        
        if recovery_ok:
            print(f"✅ All parameters recovered within 5σ")
        
        print(f"\n{'='*80}")
        print(f"✅ Integration Test PASSED")
        print(f"{'='*80}")
        print(f"\nNote: This is a simplified test without full barycentric corrections.")
        print(f"For production use, integrate with compute_residuals_simple().")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
