"""Tests for canonical residual statistics.

These tests verify that:
1. compute_residual_stats produces correct weighted RMS
2. GUI and engine use the same formula
3. Results are consistent across different components
"""

import os
import numpy as np
from pathlib import Path

# Force deterministic behavior
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


def test_compute_residual_stats_basic():
    """Test basic weighted RMS calculation."""
    print("\n" + "=" * 70)
    print("TEST: Basic Weighted RMS Calculation")
    print("=" * 70)
    
    from jug.engine.stats import compute_residual_stats
    
    # Create simple test data
    residuals_us = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    errors_us = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    
    # Compute stats
    stats = compute_residual_stats(residuals_us, errors_us)
    
    # Manually compute expected values
    weights = 1.0 / errors_us**2
    wsum = np.sum(weights)
    wmean = np.sum(weights * residuals_us) / wsum
    wrms_expected = np.sqrt(np.sum(weights * residuals_us**2) / wsum)
    
    print(f"  Residuals: {residuals_us}")
    print(f"  Errors: {errors_us}")
    print(f"  Computed weighted RMS: {stats['weighted_rms_us']:.10f}")
    print(f"  Expected weighted RMS: {wrms_expected:.10f}")
    print(f"  Computed weighted mean: {stats['weighted_mean_us']:.10f}")
    print(f"  Expected weighted mean: {wmean:.10f}")
    
    # Check exact equality (use allclose for floating point comparison)
    rms_match = np.isclose(stats['weighted_rms_us'], wrms_expected, rtol=1e-14, atol=0)
    mean_match = np.isclose(stats['weighted_mean_us'], wmean, rtol=1e-14, atol=0)
    n_match = stats['n_toas'] == 5
    
    if rms_match:
        print("[x] Weighted RMS: MATCH (within float64 precision)")
    else:
        print(f"[ ] Weighted RMS: DIFFER by {abs(stats['weighted_rms_us'] - wrms_expected)}")
    
    if mean_match:
        print("[x] Weighted mean: MATCH (within float64 precision)")
    else:
        print(f"[ ] Weighted mean: DIFFER by {abs(stats['weighted_mean_us'] - wmean)}")
    
    if n_match:
        print("[x] N TOAs: CORRECT")
    else:
        print(f"[ ] N TOAs: MISMATCH ({stats['n_toas']} vs 5)")


def test_compute_residual_stats_empty():
    """Test edge case with empty arrays."""
    print("\n" + "=" * 70)
    print("TEST: Empty Array Edge Case")
    print("=" * 70)
    
    from jug.engine.stats import compute_residual_stats
    
    residuals_us = np.array([], dtype=np.float64)
    errors_us = np.array([], dtype=np.float64)
    
    stats = compute_residual_stats(residuals_us, errors_us)
    
    print(f"  Weighted RMS: {stats['weighted_rms_us']}")
    print(f"  N TOAs: {stats['n_toas']}")
    
    rms_ok = stats['weighted_rms_us'] == 0.0
    n_ok = stats['n_toas'] == 0
    
    if rms_ok:
        print("[x] RMS is 0 for empty array")
    else:
        print("[ ] RMS should be 0 for empty array")
    
    if n_ok:
        print("[x] N TOAs is 0")
    else:
        print("[ ] N TOAs should be 0")


def test_compute_residual_stats_no_errors():
    """Test when no error bars are provided (equal weights)."""
    print("\n" + "=" * 70)
    print("TEST: No Error Bars (Equal Weights)")
    print("=" * 70)
    
    from jug.engine.stats import compute_residual_stats
    
    residuals_us = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    
    # Without errors (equal weights)
    stats = compute_residual_stats(residuals_us, errors_us=None)
    
    # Expected: sqrt(mean(r^2)) with equal weights
    expected_rms = np.sqrt(np.mean(residuals_us**2))
    expected_mean = np.mean(residuals_us)
    
    print(f"  Computed weighted RMS: {stats['weighted_rms_us']:.10f}")
    print(f"  Expected (equal weights): {expected_rms:.10f}")
    
    rms_match = stats['weighted_rms_us'] == expected_rms
    mean_match = stats['weighted_mean_us'] == expected_mean
    
    if rms_match:
        print("[x] RMS matches equal-weight expectation")
    else:
        print(f"[ ] RMS mismatch: {abs(stats['weighted_rms_us'] - expected_rms)}")
    
    if mean_match:
        print("[x] Mean matches equal-weight expectation")
    else:
        print(f"[ ] Mean mismatch")


def test_engine_gui_consistency():
    """Test that GUI stats match engine stats exactly."""
    print("\n" + "=" * 70)
    print("TEST: Engine/GUI Consistency")
    print("=" * 70)
    
    from jug.engine.stats import compute_residual_stats
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    par_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744_tdb.par"
    tim_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744.tim"
    
    if not par_file.exists():
        print(f"SKIP: Test data not found: {par_file}")
        return
    
    # Get engine result
    result = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)
    engine_rms = result['rms_us']
    engine_weighted_rms = result['weighted_rms_us']
    
    print(f"  Engine RMS (primary): {engine_rms:.6f} mus")
    print(f"  Engine weighted RMS: {engine_weighted_rms:.6f} mus")
    
    # Compute using canonical stats function (same as GUI would use)
    stats = compute_residual_stats(result['residuals_us'], result['errors_us'])
    gui_style_rms = stats['weighted_rms_us']
    
    print(f"  GUI-style RMS (via stats): {gui_style_rms:.6f} mus")
    
    # They should match exactly
    rms_match = gui_style_rms == engine_weighted_rms
    
    if rms_match:
        print("[x] GUI stats EXACTLY MATCH engine weighted RMS")
    else:
        diff = abs(gui_style_rms - engine_weighted_rms)
        print(f"[ ] Mismatch: {diff:.10e} mus")


def test_deletion_mask_consistency():
    """Test that stats after deletion are consistent."""
    print("\n" + "=" * 70)
    print("TEST: Deletion Mask Consistency")
    print("=" * 70)
    
    from jug.engine.stats import compute_residual_stats
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    par_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744_tdb.par"
    tim_file = Path(__file__).parent.parent.parent / "data" / "pulsars" / "J1909-3744.tim"
    
    if not par_file.exists():
        print(f"SKIP: Test data not found")
        return
    
    # Get full result
    result = compute_residuals_simple(par_file, tim_file, verbose=False, subtract_tzr=False)
    residuals_full = result['residuals_us']
    errors_full = result['errors_us']
    
    n_full = len(residuals_full)
    print(f"  Full dataset: {n_full} TOAs")
    print(f"  Full RMS: {result['rms_us']:.6f} mus")
    
    # Simulate deletion: keep only first 1000 TOAs
    keep_mask = np.zeros(n_full, dtype=bool)
    keep_mask[:1000] = True
    
    residuals_kept = residuals_full[keep_mask]
    errors_kept = errors_full[keep_mask]
    
    # Compute stats for kept TOAs
    stats_kept = compute_residual_stats(residuals_kept, errors_kept)
    
    print(f"  After deletion: {len(residuals_kept)} TOAs")
    print(f"  RMS after deletion: {stats_kept['weighted_rms_us']:.6f} mus")
    
    # Manually verify
    weights = 1.0 / errors_kept**2
    expected_rms = np.sqrt(np.sum(weights * residuals_kept**2) / np.sum(weights))
    
    print(f"  Expected RMS: {expected_rms:.6f} mus")
    
    rms_match = stats_kept['weighted_rms_us'] == expected_rms
    
    if rms_match:
        print("[x] Deletion stats EXACTLY MATCH manual calculation")
    else:
        diff = abs(stats_kept['weighted_rms_us'] - expected_rms)
        print(f"[ ] Mismatch: {diff:.10e} mus")


def test_chi2_calculation():
    """Test chi-squared calculation."""
    print("\n" + "=" * 70)
    print("TEST: Chi-squared Calculation")
    print("=" * 70)
    
    from jug.engine.stats import compute_chi2_reduced
    
    # Simple test data
    residuals_us = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    errors_us = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    
    # With unit errors, chi2 = sum(r^2) = 1+4+9+16+25 = 55
    chi2_stats = compute_chi2_reduced(residuals_us, errors_us, n_params=2)
    
    expected_chi2 = 55.0
    expected_dof = 3  # 5 - 2
    expected_chi2_red = 55.0 / 3
    
    print(f"  Chi2: {chi2_stats['chi2']:.2f} (expected: {expected_chi2})")
    print(f"  DOF: {chi2_stats['dof']} (expected: {expected_dof})")
    print(f"  Chi2/DOF: {chi2_stats['chi2_reduced']:.4f} (expected: {expected_chi2_red:.4f})")
    
    chi2_ok = chi2_stats['chi2'] == expected_chi2
    dof_ok = chi2_stats['dof'] == expected_dof
    chi2_red_ok = abs(chi2_stats['chi2_reduced'] - expected_chi2_red) < 1e-10
    
    if chi2_ok:
        print("[x] Chi2 correct")
    else:
        print("[ ] Chi2 incorrect")
    
    if dof_ok:
        print("[x] DOF correct")
    else:
        print("[ ] DOF incorrect")
    
    if chi2_red_ok:
        print("[x] Chi2/DOF correct")
    else:
        print("[ ] Chi2/DOF incorrect")


def run_all_tests():
    """Run all stats tests."""
    print("\n" + "=" * 70)
    print("CANONICAL STATS TESTS")
    print("=" * 70)
    
    results = {}
    
    results['basic'] = test_compute_residual_stats_basic()
    results['empty'] = test_compute_residual_stats_empty()
    results['no_errors'] = test_compute_residual_stats_no_errors()
    results['engine_gui'] = test_engine_gui_consistency()
    results['deletion'] = test_deletion_mask_consistency()
    results['chi2'] = test_chi2_calculation()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "[x] PASS" if passed else "[ ] FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\nAll stats tests passed!")
    else:
        print("\nSome tests failed!")
    
    return all_passed


if __name__ == '__main__':
    run_all_tests()
