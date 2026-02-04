#!/usr/bin/env python3
"""Test FD and H3/STIG parameter fitting on real MPTA pulsar data.

This script:
1. Loads a pulsar with FD and/or H3/STIG parameters
2. Creates a perturbed model
3. Runs the fitter
4. Shows pre-fit and post-fit comparison
"""

import sys
import os
import copy

# Add JUG to path
sys.path.insert(0, '/home/mattm/soft/JUG')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# JUG imports
from jug.io.par_reader import parse_par_file
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.fitting.optimized_fitter import fit_parameters_optimized


def test_fd_h3_stig_fitting():
    """Test fitting FD and H3/STIG parameters on J0613-0200."""
    
    # Data paths - J0613-0200 has ELL1H with H3+STIG
    data_dir = Path('/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb')
    par_file = data_dir / 'J0613-0200_tdb.par'
    tim_file = data_dir / 'J0613-0200.tim'
    
    print("=" * 70)
    print("Testing FD and H3/STIG Parameter Fitting")
    print("Pulsar: J0613-0200 (ELL1H binary model)")
    print("=" * 70)
    
    # Parse original model
    original_params = parse_par_file(par_file)
    print("\n--- Original Model Parameters ---")
    params_to_test = ['H3', 'STIG', 'FD1', 'FD2', 'A1', 'PB']
    for p in params_to_test:
        if p in original_params:
            print(f"  {p}: {original_params[p]}")
    
    # Compute original residuals
    print("\n--- Computing Original Residuals ---")
    orig_result = compute_residuals_simple(
        par_file=str(par_file),
        tim_file=str(tim_file),
        verbose=False
    )
    orig_res_us = orig_result['residuals_us']
    orig_rms = np.std(orig_res_us)
    orig_mjd = orig_result['tdb_mjd']
    print(f"Original residual RMS: {orig_rms:.3f} µs")
    
    # Create perturbed model
    print("\n--- Creating Perturbed Model ---")
    perturbed_params = copy.deepcopy(original_params)
    
    # Perturb H3 and STIG
    perturbations = {}
    
    if 'H3' in perturbed_params:
        orig_h3 = perturbed_params['H3']
        perturbed_params['H3'] = orig_h3 * 0.9  # -10%
        perturbations['H3'] = (orig_h3, perturbed_params['H3'])
        
    if 'STIG' in perturbed_params:
        orig_stig = perturbed_params['STIG']
        perturbed_params['STIG'] = orig_stig * 1.05  # +5%
        perturbations['STIG'] = (orig_stig, perturbed_params['STIG'])
    
    for p, (orig, pert) in perturbations.items():
        print(f"  {p}: {orig:.6e} → {pert:.6e}")
    
    # Write perturbed par file
    perturbed_par = data_dir / 'J0613-0200_tdb_perturbed.par'
    with open(par_file, 'r') as f:
        lines = f.readlines()
    
    with open(perturbed_par, 'w') as f:
        for line in lines:
            written = False
            for param in perturbations:
                if line.strip().startswith(param + ' '):
                    parts = line.split()
                    parts[1] = f'{perturbed_params[param]:.15e}'
                    f.write('  '.join(parts) + '\n')
                    written = True
                    break
            if not written:
                f.write(line)
    
    # Compute perturbed residuals
    print("\n--- Computing Pre-Fit (Perturbed) Residuals ---")
    prefit_result = compute_residuals_simple(
        par_file=str(perturbed_par),
        tim_file=str(tim_file),
        verbose=False
    )
    prefit_res_us = prefit_result['residuals_us']
    prefit_rms = np.std(prefit_res_us)
    print(f"Pre-fit residual RMS: {prefit_rms:.3f} µs")
    
    # Run fitter
    print("\n--- Running Fitter ---")
    fit_params = list(perturbations.keys())
    print(f"Fitting parameters: {fit_params}")
    
    try:
        fit_result = fit_parameters_optimized(
            par_file=str(perturbed_par),
            tim_file=str(tim_file),
            fit_params=fit_params,
            max_iter=20,
            verbose=True
        )
        
        # Get post-fit results directly from fitter
        postfit_res_us = fit_result['residuals_us']
        postfit_rms = fit_result['final_rms']
        final_params = fit_result['final_params']
        uncertainties = fit_result['uncertainties']
        
        print(f"\n--- Post-Fit Results ---")
        print(f"Post-fit residual RMS (from fitter): {postfit_rms:.3f} µs")
        
        # Compare parameters
        print("\n--- Parameter Comparison ---")
        print(f"{'Parameter':>10} | {'Original':>15} | {'Pre-Fit':>15} | {'Post-Fit':>15} | {'Uncertainty':>12}")
        print("-" * 80)
        
        for p in fit_params:
            orig_val = original_params.get(p, 0.0)
            prefit_val = perturbations.get(p, (0, 0))[1]
            postfit_val = final_params.get(p, prefit_val)
            unc = uncertainties.get(p, 0.0)
            
            print(f"{p:>10} | {orig_val:>15.6e} | {prefit_val:>15.6e} | {postfit_val:>15.6e} | {unc:>12.3e}")
        
        # Summary
        print("\n--- Summary ---")
        print(f"Original RMS:    {orig_rms:.3f} µs (correct model)")
        print(f"Pre-fit RMS:     {prefit_rms:.3f} µs (perturbed model)")
        print(f"Post-fit RMS:    {postfit_rms:.3f} µs (after fitting H3/STIG)")
        print(f"RMS improvement: {(prefit_rms - postfit_rms):.3f} µs")
        
        # Create plots
        print("\n--- Generating Plots ---")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Use mjd from fitter result 
        mjd = fit_result['tdb_mjd']
        
        # Original residuals
        axes[0].scatter(orig_mjd, orig_res_us, s=1, alpha=0.5)
        axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[0].set_xlabel('MJD')
        axes[0].set_ylabel('Residual (µs)')
        axes[0].set_title(f'Original Model\nRMS = {orig_rms:.3f} µs')
        
        # Pre-fit residuals (perturbed)
        axes[1].scatter(mjd, prefit_res_us, s=1, alpha=0.5, color='orange')
        axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('MJD')
        axes[1].set_ylabel('Residual (µs)')
        axes[1].set_title(f'Pre-Fit (Perturbed H3/STIG)\nRMS = {prefit_rms:.3f} µs')
        
        # Post-fit residuals
        axes[2].scatter(mjd, postfit_res_us, s=1, alpha=0.5, color='green')
        axes[2].axhline(0, color='k', linestyle='--', alpha=0.3)
        axes[2].set_xlabel('MJD')
        axes[2].set_ylabel('Residual (µs)')
        axes[2].set_title(f'Post-Fit\nRMS = {postfit_rms:.3f} µs')
        
        plt.suptitle('J0613-0200 H3/STIG Fitting Test', fontsize=14)
        plt.tight_layout()
        
        plot_path = '/home/mattm/.gemini/antigravity/brain/eb6739f1-97f6-40c9-af33-d28e508b013a/h3_stig_fit_test.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.close()
        
        return {
            'success': True,
            'original_rms': orig_rms,
            'prefit_rms': prefit_rms,
            'postfit_rms': postfit_rms,
            'plot_path': plot_path
        }
        
    except Exception as e:
        print(f"\n!!! Error during fitting: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    result = test_fd_h3_stig_fitting()
    
    if result['success']:
        print("\n" + "=" * 70)
        print("✓ TEST COMPLETED SUCCESSFULLY")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ TEST FAILED")
        print(f"Error: {result.get('error', 'Unknown')}")
        print("=" * 70)
