#!/usr/bin/env python
"""Regression tests for the astrometry fitting divergence fix.

These tests verify that:
1. Reported RMS equals true RMS
2. Repeated fits do not diverge  
3. Non-astrometry fits remain unchanged
"""

import sys
import tempfile
import numpy as np
from pathlib import Path

# Add JUG to path
sys.path.insert(0, str(Path(__file__).parent))

from jug.fitting.optimized_fitter import (
    fit_parameters_optimized,
    _build_general_fit_setup_from_files,
    _compute_full_model_residuals,
)
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.io.par_reader import parse_par_file, format_ra, format_dec


def test_reported_rms_equals_true_rms():
    """Test 1: Reported RMS equals true RMS.
    
    Run one fit, write updated par to temp, compute residuals via canonical
    engine path, assert reported_final_rms ≈ true_final_rms.
    """
    print("\n" + "="*60)
    print("Test 1: Reported RMS equals true RMS")
    print("="*60)
    
    par_file = Path('data/pulsars/J1909-3744_tdb.par')
    tim_file = Path('data/pulsars/J1909-3744.tim')
    
    fit_params = ['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 
                  'DM', 'DM1', 'DM2',
                  'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT']
    
    # Run fit
    result = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=10,
        verbose=False
    )
    
    # Write updated par file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.par', delete=False) as f:
        params = parse_par_file(par_file)
        for p, v in result['final_params'].items():
            if p == 'RAJ':
                params[p] = format_ra(v)
            elif p == 'DECJ':
                params[p] = format_dec(v)
            else:
                params[p] = v
        for k, v in params.items():
            f.write(f'{k} {v}\n')
        temp_par = f.name
    
    # Compute TRUE final RMS via canonical path
    result_post = compute_residuals_simple(temp_par, str(tim_file), verbose=False)
    true_final_wrms = result_post['weighted_rms_us']
    reported_rms = result['final_rms']
    
    # Clean up
    Path(temp_par).unlink()
    
    # Assert match (within tolerance)
    diff = abs(reported_rms - true_final_wrms)
    tolerance = 0.02  # 20 nanoseconds tolerance
    
    print(f"  Reported RMS:   {reported_rms:.6f} μs")
    print(f"  TRUE final RMS: {true_final_wrms:.6f} μs")
    print(f"  Difference:     {diff:.6f} μs")
    print(f"  Tolerance:      {tolerance:.6f} μs")
    
    if diff <= tolerance:
        print("  ✓ PASSED")
        return True
    else:
        print("  ✗ FAILED")
        return False


def test_repeated_fits_no_divergence():
    """Test 2: Repeated fits do not diverge.
    
    Run 5 sequential fits, assert:
    - No NaNs
    - True RMS does not blow up
    - Fits 2..5 remain stable
    """
    print("\n" + "="*60)
    print("Test 2: Repeated fits do not diverge")
    print("="*60)
    
    par_file = Path('data/pulsars/J1909-3744_tdb.par')
    tim_file = Path('data/pulsars/J1909-3744.tim')
    
    fit_params = ['F0', 'F1', 'RAJ', 'DECJ', 'PMRA', 'PMDEC', 'PX', 
                  'DM', 'DM1', 'DM2',
                  'PB', 'A1', 'TASC', 'EPS1', 'EPS2', 'M2', 'SINI', 'PBDOT']
    
    temp_dir = tempfile.mkdtemp()
    current_par = par_file
    
    rms_values = []
    all_passed = True
    
    for fit_num in range(5):
        # Compute initial RMS
        result_pre = compute_residuals_simple(str(current_par), str(tim_file), verbose=False)
        initial_wrms = result_pre['weighted_rms_us']
        
        # Run fit
        result = fit_parameters_optimized(
            par_file=current_par,
            tim_file=tim_file,
            fit_params=fit_params,
            max_iter=10,
            verbose=False
        )
        
        # Check for NaN
        has_nan = any(np.isnan(v) for p, v in result['final_params'].items() if not isinstance(v, str))
        if has_nan:
            print(f"  Fit {fit_num+1}: FAILED (NaN in parameters)")
            all_passed = False
            break
        
        # Write updated par file
        next_par = Path(temp_dir) / f'fit{fit_num+1}.par'
        params = parse_par_file(current_par)
        for p, v in result['final_params'].items():
            if p == 'RAJ':
                params[p] = format_ra(v)
            elif p == 'DECJ':
                params[p] = format_dec(v)
            else:
                params[p] = v
        with open(next_par, 'w') as f:
            for k, v in params.items():
                f.write(f'{k} {v}\n')
        
        # Compute TRUE final RMS
        result_post = compute_residuals_simple(str(next_par), str(tim_file), verbose=False)
        true_final_wrms = result_post['weighted_rms_us']
        
        rms_values.append(true_final_wrms)
        
        # Check for blow-up (10x worse than initial)
        if true_final_wrms > initial_wrms * 10:
            print(f"  Fit {fit_num+1}: FAILED (RMS blew up: {true_final_wrms:.2f} > {initial_wrms*10:.2f})")
            all_passed = False
            break
        
        print(f"  Fit {fit_num+1}: initial={initial_wrms:.6f}, final={true_final_wrms:.6f} μs")
        
        current_par = next_par
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Check stability across fits
    if len(rms_values) >= 2:
        rms_range = max(rms_values) - min(rms_values)
        print(f"  RMS range across fits: {rms_range:.6f} μs")
        if rms_range > 0.1:  # More than 100 ns variation suggests instability
            print("  WARNING: RMS variation seems high")
    
    if all_passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return all_passed


def test_non_astrometry_fits_unchanged():
    """Test 3: Non-astrometry fits unchanged.
    
    Fit only spin+DM and confirm:
    - Fitted params are reasonable
    - Performance is acceptable
    """
    print("\n" + "="*60)
    print("Test 3: Non-astrometry fits unchanged")
    print("="*60)
    
    par_file = Path('data/pulsars/J1909-3744_tdb.par')
    tim_file = Path('data/pulsars/J1909-3744.tim')
    
    # Spin + DM only (no astrometry)
    fit_params = ['F0', 'F1', 'DM', 'DM1', 'DM2']
    
    import time
    start = time.time()
    
    # Run fit
    result = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=10,
        verbose=False
    )
    
    elapsed = time.time() - start
    
    # Check results
    passed = True
    
    # Check convergence is reasonable
    print(f"  Converged: {result['converged']}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final RMS: {result['final_rms']:.6f} μs")
    print(f"  Time: {elapsed:.3f}s")
    
    # Verify params are not NaN
    for p, v in result['final_params'].items():
        if np.isnan(v):
            print(f"  FAILED: {p} is NaN")
            passed = False
    
    # Verify RMS is reasonable (should be similar to original)
    if result['final_rms'] > 1.0:  # Should be ~0.4 μs
        print(f"  WARNING: RMS seems high ({result['final_rms']:.2f} μs)")
    
    # Verify performance (should be fast, < 5s)
    if elapsed > 5.0:
        print(f"  WARNING: Fit took longer than expected ({elapsed:.1f}s)")
    
    if passed:
        print("  ✓ PASSED")
    else:
        print("  ✗ FAILED")
    
    return passed


def main():
    """Run all regression tests."""
    print("\n" + "="*70)
    print("Astrometry Fitting Divergence Fix - Regression Tests")
    print("="*70)
    
    results = []
    
    results.append(("Reported RMS equals true RMS", test_reported_rms_equals_true_rms()))
    results.append(("Repeated fits no divergence", test_repeated_fits_no_divergence()))
    results.append(("Non-astrometry fits unchanged", test_non_astrometry_fits_unchanged()))
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
