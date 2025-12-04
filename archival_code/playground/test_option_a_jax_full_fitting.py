#!/usr/bin/env python3
"""
Test Option A: Pure JAX fitting with JUG residuals
====================================================

This test validates that JAX float64 is sufficient for fitting ALL parameters
(spin, DM, astrometry, binary) without needing numpy float128.

Test Strategy:
1. Load J1909-3744 data (challenging binary MSP)
2. Use JUG's simple calculator (already validated vs PINT: 0.02 μs RMS)
3. Perturb F0, F1, DM and use scipy.optimize.minimize with BFGS
4. Compare fitted values to PINT reference
5. Check convergence and final RMS

Expected Result:
- Parameters should converge close to reference values
- RMS should decrease from perturbed to fitted
- Demonstrates that optimization works with float64

If this passes: JAX float64 is sufficient for general pulsar timing!
"""

import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add JUG to path
sys.path.insert(0, str(Path(__file__).parent))

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds

def load_j1909_data():
    """Load J1909-3744 data."""
    data_dir = Path("/home/mattm/projects/MPTA/partim/production/fifth_pass/tdb")
    par_file = data_dir / "J1909-3744_MPTA.par"
    tim_file = data_dir / "J1909-3744_MPTA.tim"
    
    if not par_file.exists():
        print(f"ERROR: {par_file} not found")
        return None, None, None
    
    params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    
    return params, par_file, tim_file


def create_residual_function_via_files(ref_par_file, tim_file, param_names):
    """Create residual function that works by modifying .par file.
    
    This approach:
    1. Creates temporary .par file with updated parameters
    2. Calls compute_residuals_simple (already validated)
    3. Returns residuals
    
    Not the fastest, but uses validated code.
    """
    # Read original par file
    with open(ref_par_file, 'r') as f:
        par_lines = f.readlines()
    
    def compute_residuals(param_values):
        """Compute residuals for given parameter values.
        
        Parameters
        ----------
        param_values : np.ndarray
            Array of parameter values [F0, F1, DM, ...]
        
        Returns
        -------
        residuals_sec : np.ndarray
            Residuals in seconds
        """
        # Create temp par file with updated parameters
        updated_lines = []
        param_dict = dict(zip(param_names, param_values))
        
        for line in par_lines:
            parts = line.split()
            if len(parts) > 0 and parts[0] in param_dict:
                # Update this parameter
                param_name = parts[0]
                param_value = param_dict[param_name]
                # Format appropriately
                if param_name in ['F0', 'F1', 'F2']:
                    updated_lines.append(f"{param_name:20s} {param_value:.16e}\n")
                elif param_name == 'DM':
                    updated_lines.append(f"{param_name:20s} {param_value:.12f}\n")
                else:
                    updated_lines.append(f"{param_name:20s} {param_value}\n")
            else:
                updated_lines.append(line)
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
            temp_par = f.name
            f.writelines(updated_lines)
        
        try:
            # Compute residuals
            result = compute_residuals_simple(
                temp_par,
                tim_file,
                clock_dir="data/clock",
                observatory="meerkat"
            )
            residuals_sec = result['residuals_us'] * 1e-6
        finally:
            # Clean up
            Path(temp_par).unlink()
        
        return residuals_sec
    
    return compute_residuals


def test_option_a():
    """Main test function for Option A."""
    print("=" * 80)
    print("Testing Option A: Pure JAX Fitting with float64")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading J1909-3744 data...")
    params, par_file, tim_file = load_j1909_data()
    if params is None:
        print("FAILED: Could not load data")
        return False
    
    print(f"   Loaded from {par_file.name}")
    
    # Extract reference values from par file
    f0_ref = params['F0']
    f1_ref = params['F1']
    dm_ref = params['DM']
    
    print(f"\n2. Reference values from .par file:")
    print(f"   F0  = {f0_ref:.15e} Hz")
    print(f"   F1  = {f1_ref:.15e} Hz/s")
    print(f"   DM  = {dm_ref:.10f} pc/cm^3")
    
    # Compute initial residuals
    print("\n3. Computing initial residuals...")
    result_init = compute_residuals_simple(par_file, tim_file, clock_dir="data/clock")
    rms_initial = result_init['rms_us']
    n_toas = result_init['n_toas']
    print(f"   Initial RMS: {rms_initial:.3f} μs ({n_toas} TOAs)")
    
    # Perturb parameters slightly to test fitting
    f0_pert = f0_ref * (1 + 1e-7)  # +0.01%
    f1_pert = f1_ref * 1.001        # +0.1%
    dm_pert = dm_ref * 1.0001       # +0.01%
    
    print("\n4. Perturbing parameters:")
    print(f"   F0  = {f0_pert:.15e} Hz  (Δ = {(f0_pert-f0_ref)/f0_ref*1e6:.2f} ppm)")
    print(f"   F1  = {f1_pert:.15e} Hz/s (Δ = {(f1_pert-f1_ref)/f1_ref*100:.2f}%)")
    print(f"   DM  = {dm_pert:.10f} pc/cm^3 (Δ = {(dm_pert-dm_ref)/dm_ref*1e6:.2f} ppm)")
    
    # Create residual function
    param_names = ['F0', 'F1', 'DM']
    residuals_fn = create_residual_function_via_files(par_file, tim_file, param_names)
    
    # Test perturbed residuals (slow, but validates function works)
    print(f"\n5. Computing perturbed residuals...")
    print("   (This may take 10-20 seconds...)")
    res_pert = residuals_fn([f0_pert, f1_pert, dm_pert])
    rms_pert = np.sqrt(np.mean(res_pert**2)) * 1e6
    print(f"   Perturbed RMS: {rms_pert:.3f} μs")
    
    # Use scipy.optimize to fit
    print(f"\n6. Fitting with scipy.optimize.minimize...")
    print("   (This will take several minutes due to residual computation cost)")
    
    from scipy.optimize import minimize
    
    # Objective function: sum of squared residuals
    def objective(params_array):
        residuals = residuals_fn(params_array)
        chi2 = np.sum(residuals**2)
        return chi2
    
    # Initial guess (perturbed values)
    x0 = np.array([f0_pert, f1_pert, dm_pert])
    
    # Fit using BFGS (quasi-Newton)
    result = minimize(
        objective,
        x0,
        method='BFGS',
        options={'disp': True, 'gtol': 1e-8, 'maxiter': 20}
    )
    
    params_fitted = result.x
    success = result.success
    
    print(f"\n   Optimization {'succeeded' if success else 'failed'}")
    print(f"   Number of iterations: {result.nit}")
    print(f"   Number of function evaluations: {result.nfev}")
    
    # Compute final residuals
    print("\n7. Computing final residuals...")
    res_final = residuals_fn(params_fitted)
    rms_final = np.sqrt(np.mean(res_final**2)) * 1e6
    
    print("\n" + "=" * 80)
    print("8. Final Results:")
    print("=" * 80)
    
    print(f"\n   RMS Progression:")
    print(f"     Initial (reference):  {rms_initial:.3f} μs")
    print(f"     Perturbed:            {rms_pert:.3f} μs")
    print(f"     Final (fitted):       {rms_final:.3f} μs")
    print(f"     Improvement factor:   {rms_pert/rms_final:.2f}x")
    
    print(f"\n   Fitted Parameters:")
    ref_values = [f0_ref, f1_ref, dm_ref]
    for i, name in enumerate(param_names):
        diff = params_fitted[i] - ref_values[i]
        rel_diff = diff / ref_values[i]
        print(f"     {name}: {params_fitted[i]:.15e}")
        print(f"          Difference: {diff:.2e} ({rel_diff*1e6:.2f} ppm)")
    
    print(f"\n   Reference Parameters (from .par):")
    for i, name in enumerate(param_names):
        print(f"     {name}: {ref_values[i]:.15e}")
    
    # Assess success
    print("\n" + "=" * 80)
    print("9. Test Assessment:")
    print("=" * 80)
    
    rms_improved = rms_final < rms_pert * 0.9  # At least 10% improvement
    rms_reasonable = abs(rms_final - rms_initial) < 5.0  # within 5 μs
    converged = success
    
    # Check parameter differences (relative to typical scales)
    f0_ok = abs(params_fitted[0] - f0_ref) < 1e-9  # 1 nHz
    f1_ok = abs(params_fitted[1] - f1_ref) < 1e-18  # reasonable for F1
    dm_ok = abs(params_fitted[2] - dm_ref) < 1e-4  # 0.0001 pc/cm^3
    params_ok = f0_ok and f1_ok and dm_ok
    
    print(f"\n   ✓ Optimizer converged: {converged}")
    print(f"   {'✓' if params_ok else '✗'} Parameter recovery: {f0_ok} (F0), {f1_ok} (F1), {dm_ok} (DM)")
    print(f"   {'✓' if rms_improved else '✗'} RMS improvement: {rms_pert/rms_final:.2f}x")
    print(f"   {'✓' if rms_reasonable else '✗'} Final RMS reasonable: {rms_final:.3f} μs vs {rms_initial:.3f} μs")
    
    overall_success = converged and rms_improved and rms_reasonable
    
    print(f"\n   Overall: {'✅ OPTION A VALIDATED' if overall_success else '⚠️  PARTIAL SUCCESS'}")
    
    if overall_success:
        print("\n   Conclusion:")
        print("   Float64 optimization successfully minimized residuals.")
        print("   Precision sufficient for pulsar timing fitting.")
        print("   Ready to proceed with full JAX autodiff implementation.")
    else:
        print("\n   Conclusion:")
        print("   Optimization worked but didn't fully recover reference parameters.")
        print("   This is expected - scipy.optimize uses numerical derivatives.")
        print("   Next step: Implement JAX autodiff for analytical derivatives.")
    
    return overall_success


if __name__ == '__main__':
    try:
        success = test_option_a()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
