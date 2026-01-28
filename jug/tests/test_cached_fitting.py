"""
Regression test for cached fitting - ensures bit-for-bit identical results.

This test verifies that the cached fitting path produces EXACTLY the same
results as the file-based fitting path (no tolerance, bit-for-bit equality).
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.engine.session import TimingSession


def test_cached_vs_file_fitting_bitwise_identical():
    """
    Test that cached fitting produces bit-for-bit identical results to file fitting.
    
    This is a STRICT test - uses np.array_equal, not np.allclose.
    Any difference (even 1 ULP) will fail the test.
    """
    # Use deterministic test data
    par_file = Path("data/pulsars/J1909-3744_tdb.par")
    tim_file = Path("data/pulsars/J1909-3744.tim")
    
    if not par_file.exists() or not tim_file.exists():
        print(f"SKIP: Test data not found ({par_file}, {tim_file})")
        return
    
    fit_params = ['F0', 'F1', 'DM']
    
    # OLD PATH: File-based fitting
    print("\n" + "="*80)
    print("OLD PATH: File-based fitting")
    print("="*80)
    result_old = fit_parameters_optimized(
        par_file=par_file,
        tim_file=tim_file,
        fit_params=fit_params,
        max_iter=25,
        convergence_threshold=1e-14,
        clock_dir=None,
        verbose=True,
        device='cpu'  # Force CPU for determinism
    )
    
    # NEW PATH: Cached fitting via TimingSession
    print("\n" + "="*80)
    print("NEW PATH: Cached fitting via TimingSession")
    print("="*80)
    session = TimingSession(par_file, tim_file, verbose=True)
    
    # Populate cache with subtract_tzr=False (needed for fitting)
    print("\nPopulating cache...")
    _ = session.compute_residuals(subtract_tzr=False)
    
    # Run fit (should use cached path)
    print("\nRunning fit with cached arrays...")
    result_new = session.fit_parameters(
        fit_params=fit_params,
        max_iter=25,
        convergence_threshold=1e-14,
        verbose=True
    )
    
    # STRICT BIT-FOR-BIT COMPARISON
    print("\n" + "="*80)
    print("BIT-FOR-BIT COMPARISON")
    print("="*80)
    
    # Check final parameters (exact equality)
    for param in fit_params:
        val_old = result_old['final_params'][param]
        val_new = result_new['final_params'][param]
        
        if val_old == val_new:
            print(f"✓ {param}: IDENTICAL ({val_old:.20e})")
        else:
            diff = abs(val_new - val_old)
            rel_diff = diff / abs(val_old) if val_old != 0 else diff
            print(f"✗ {param}: DIFFERS!")
            print(f"  Old: {val_old:.20e}")
            print(f"  New: {val_new:.20e}")
            print(f"  Diff: {diff:.6e} (rel: {rel_diff:.6e})")
            raise AssertionError(f"Parameter {param} differs between old and new path")
    
    # Check uncertainties
    for param in fit_params:
        unc_old = result_old['uncertainties'][param]
        unc_new = result_new['uncertainties'][param]
        
        if unc_old == unc_new:
            print(f"✓ {param} uncertainty: IDENTICAL ({unc_old:.6e})")
        else:
            print(f"✗ {param} uncertainty: DIFFERS!")
            print(f"  Old: {unc_old:.20e}")
            print(f"  New: {unc_new:.20e}")
            raise AssertionError(f"Uncertainty for {param} differs")
    
    # Check RMS
    if result_old['final_rms'] == result_new['final_rms']:
        print(f"✓ Final RMS: IDENTICAL ({result_old['final_rms']:.10f} μs)")
    else:
        print(f"✗ Final RMS: DIFFERS!")
        print(f"  Old: {result_old['final_rms']:.10f} μs")
        print(f"  New: {result_new['final_rms']:.10f} μs")
        raise AssertionError("Final RMS differs")
    
    # Check iterations
    if result_old['iterations'] == result_new['iterations']:
        print(f"✓ Iterations: IDENTICAL ({result_old['iterations']})")
    else:
        print(f"✗ Iterations: DIFFERS!")
        print(f"  Old: {result_old['iterations']}")
        print(f"  New: {result_new['iterations']}")
        raise AssertionError("Iteration count differs")
    
    # Check residuals arrays (bit-for-bit)
    if np.array_equal(result_old['residuals_us'], result_new['residuals_us']):
        print(f"✓ Residuals array: IDENTICAL (all {len(result_old['residuals_us'])} values)")
    else:
        max_diff = np.max(np.abs(result_old['residuals_us'] - result_new['residuals_us']))
        print(f"✗ Residuals array: DIFFERS! (max diff: {max_diff:.6e} μs)")
        raise AssertionError("Residuals array differs")
    
    # Check covariance matrix
    if np.array_equal(result_old['covariance'], result_new['covariance']):
        print(f"✓ Covariance matrix: IDENTICAL")
    else:
        max_diff = np.max(np.abs(result_old['covariance'] - result_new['covariance']))
        print(f"✗ Covariance matrix: DIFFERS! (max diff: {max_diff:.6e})")
        raise AssertionError("Covariance matrix differs")
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED: Cached path is bit-for-bit identical!")
    print("="*80)


if __name__ == '__main__':
    test_cached_vs_file_fitting_bitwise_identical()
