
import sys
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
import warnings
import traceback
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# PINT imports
import pint.models
import pint.toa
import pint.fitter
import pint.logging
from pint.models import get_model_and_toas

# JUG imports
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.io.par_reader import parse_par_file

# Suppress warnings
warnings.filterwarnings("ignore")
pint.logging.setup(level="WARNING")

DATA_DIR = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
ARTIFACT_DIR = Path('/home/mattm/.gemini/antigravity/brain/eb6739f1-97f6-40c9-af33-d28e508b013a')

PULSARS = [
    'J0613-0200',  # ELL1 + FD + A1
    'J1022+1001',  # DDH + H3/STIG
    'J1909-3744',  # ELL1 (High Precision)
    'J1713+0747',  # DD (High Precision)
]

def run_comparison(psr_name):
    print(f"\n{'='*60}")
    print(f"Comparing JUG vs PINT: {psr_name}")
    print(f"{'='*60}")
    
    par_file = DATA_DIR / f"{psr_name}_tdb.par"
    tim_file = DATA_DIR / f"{psr_name}.tim"
    
    if not par_file.exists():
        # Handle J1909 having feather file or different extension check?
        # Listing showed J1909-3744_tdb.par exists.
        print(f"Par file not found: {par_file}")
        return

    # ---------------------------------------------------------
    # 1. Residual Comparison (Pre-Fit)
    # ---------------------------------------------------------
    print("Loading PINT model/TOAs...")
    try:
        m, t = get_model_and_toas(str(par_file), str(tim_file))
    except Exception as e:
        print(f"PINT Load Error: {e}")
        return

    print("Computing PINT residuals...")
    # Faster way:
    fitter = pint.fitter.WLSFitter(t, m)
    pint_resids_sec = fitter.resids.time_resids.to('s').value
    pint_rms_us = fitter.resids.rms_weighted().to('us').value
    print(f"PINT wRMS: {pint_rms_us:.6f} μs")

    print("Computing JUG residuals...")
    try:
        jug_res = compute_residuals_simple(str(par_file), str(tim_file), verbose=False)
        jug_resids_sec = jug_res['residuals_us'] * 1e-6
        jug_rms_us = jug_res['weighted_rms_us']
        print(f"JUG wRMS:  {jug_rms_us:.6f} μs")
    except Exception as e:
        print(f"JUG Compute Error: {e}")
        traceback.print_exc()
        return

    # Compare Residuals
    # Note: TOA ordering might differ if JUG/PINT sort differently.
    # PINT usually sorts by time. JUG should too.
    # Check lengths
    if len(pint_resids_sec) != len(jug_resids_sec):
        print(f"LENGTH MISMATCH: PINT={len(pint_resids_sec)}, JUG={len(jug_resids_sec)}")
        # Try to match by MJDs?
        # For now assume same ordering if sorted by MJD
    else:
        diff_sec = pint_resids_sec - jug_resids_sec
        diff_us = diff_sec * 1e6
        max_diff_ns = np.max(np.abs(diff_sec)) * 1e9
        rms_diff_ns = np.std(diff_sec) * 1e9
        
        print(f"Difference (PINT - JUG):")
        print(f"  Max Diff: {max_diff_ns:.3f} ns")
        print(f"  RMS Diff: {rms_diff_ns:.3f} ns")
        
        if rms_diff_ns < 1.0:
            print("✅ RESIDUALS MATCH (< 1 ns)")
        else:
            print("⚠️ RESIDUALS DIFFER (> 1 ns)")

    # ---------------------------------------------------------
    # 2. Fit Comparison
    # ---------------------------------------------------------
    # Perturb parameters slightly to test convergence
    # We will fit F0 and one binary parameter (if exists)
    fit_params = ['F0']
    
    # Identify binary model and pick a param
    binary_model = m.BINARY.value
    if binary_model.startswith('ELL1'):
        fit_params.append('A1')
    elif binary_model.startswith('DD'):
        fit_params.append('A1') # T0?
    
    # Write perturbed par file
    print(f"\nPerturbing parameters: {fit_params}")
    perturbed_par = DATA_DIR / f"{psr_name}_perturbed_compare.par"
    
    # Simple perturbation in memory isn't enough for JUG which reads files
    # We edit file manually or use PINT to write it?
    # Let's use PINT to perturb and write
    m_perturbed = m # Copy?
    for param in fit_params:
        val = getattr(m_perturbed, param).value
        # Perturb by 1e-7 relative or something
        if param == 'F0':
            new_val = val * (1 + 1e-8)
        else:
            new_val = val * (1 + 1e-4) # 0.01%
        getattr(m_perturbed, param).value = new_val
        getattr(m_perturbed, param).frozen = False # Ensure fit
    
    with open(perturbed_par, 'w') as f:
        f.write(m_perturbed.as_parfile())

    # --- Run PINT Fit ---
    print("Running PINT Fitter...")
    pint_fitter = pint.fitter.WLSFitter(t, m_perturbed)
    pint_fitter.fit_toas(maxiter=10)
    pint_final_rms = pint_fitter.resids.rms_weighted().to('us').value
    print(f"PINT Final RMS: {pint_final_rms:.6f} μs")
    
    # --- Run JUG Fit ---
    print("Running JUG Fitter...")
    # JUG needs fit_params list. PINT uses frozen/unfrozen status in par file.
    # But JUG's 'fit_parameters_optimized' takes a list `fit_params`.
    # Does JUG respect '1' flags in par file? Yes, but `fit_parameters_optimized` implies passed list.
    # Actually, JUG's `fit_parameters_optimized` overrides par file flags if `fit_params` list provided.
    
    jug_result = fit_parameters_optimized(
        str(perturbed_par), str(tim_file),
        fit_params=fit_params,
        max_iter=10
    )
    jug_final_rms = jug_result['final_rms']
    print(f"JUG Final RMS:  {jug_final_rms:.6f} μs")

    # --- Compare Parameters ---
    print("\nParameter Comparison:")
    print(f"{'Param':<10} {'PINT Value':<20} {'JUG Value':<20} {'Diff':<15} {'Uncertainty Ratio'}")
    
    for param in fit_params:
        pint_val = getattr(pint_fitter.model, param).value
        pint_err = getattr(pint_fitter.model, param).uncertainty_value
        
        jug_val = jug_result['final_params'][param]
        # JUG uncertainty? 'fit_parameters_optimized' returns 'fitted_params_errors' maybe?
        # Let's check keys
        jug_err = jug_result.get('final_params_errors', {}).get(param, 0.0)
        
        diff = pint_val - jug_val
        ratio = jug_err / pint_err if pint_err > 0 else 0.0
        
        print(f"{param:<10} {pint_val:<20.12e} {jug_val:<20.12e} {diff:<15.4e} {ratio:.4f}")

    if abs(pint_final_rms - jug_final_rms) < 1e-4:
        print("✅ FIT CONVERGED TO SAME RMS")
    else:
        print("⚠️ RMS DIFFERENCE")

if __name__ == "__main__":
    for psr in PULSARS:
        run_comparison(psr)
