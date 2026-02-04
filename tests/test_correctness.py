#!/usr/bin/env python3
"""
Correctness tests for JUG using golden reference data.

Verifies that JUG produces numerically consistent results by comparing
against pre-computed golden values. This catches regressions in the
residual calculation pipeline.

Run with: python tests/test_correctness.py

For PINT cross-validation (optional):
    python tests/test_correctness.py --pint
    # Or set env var: JUG_TEST_PINT=1 python tests/test_correctness.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def get_golden_dir():
    """Get path to golden data directory."""
    return Path(__file__).parent / "data_golden"


def load_golden_reference(name: str = "J1909_mini"):
    """Load golden reference values from JSON."""
    golden_file = get_golden_dir() / f"{name}_golden.json"
    if not golden_file.exists():
        return None
    with open(golden_file) as f:
        return json.load(f)


def test_mini_dataset_correctness():
    """Test JUG results match golden reference for mini dataset."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    golden = load_golden_reference("J1909_mini")
    if golden is None:
        return False, "golden reference not found"
    
    golden_dir = get_golden_dir()
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        return False, "mini dataset not found"
    
    # Compute residuals
    result = compute_residuals_simple(str(par), str(tim), verbose=False)
    
    errors = []
    
    # Check n_toas
    if result['n_toas'] != golden['n_toas']:
        errors.append(f"n_toas: expected {golden['n_toas']}, got {result['n_toas']}")
    
    # Check weighted RMS
    rms_tol = golden['tolerances']['rms_rel_tol']
    wrms_rel_diff = abs(result['weighted_rms_us'] - golden['weighted_rms_us']) / golden['weighted_rms_us']
    if wrms_rel_diff > rms_tol:
        errors.append(f"weighted_rms: rel diff {wrms_rel_diff:.2e} > tol {rms_tol}")
    
    # Check unweighted RMS
    rms_rel_diff = abs(result['unweighted_rms_us'] - golden['unweighted_rms_us']) / golden['unweighted_rms_us']
    if rms_rel_diff > rms_tol:
        errors.append(f"unweighted_rms: rel diff {rms_rel_diff:.2e} > tol {rms_tol}")
    
    # Check first 5 residuals (in ns for precision)
    residual_tol = golden['tolerances']['residual_abs_tol_ns']
    actual_ns = [r * 1000 for r in result['residuals_us'][:5]]
    expected_ns = golden['first_5_residuals_ns']
    
    for i, (act, exp) in enumerate(zip(actual_ns, expected_ns)):
        diff = abs(act - exp)
        if diff > residual_tol:
            errors.append(f"residual[{i}]: diff {diff:.3f} ns > tol {residual_tol} ns")
    
    if errors:
        return False, "; ".join(errors[:3])  # First 3 errors
    
    return True, f"OK (wRMS={result['weighted_rms_us']:.4f}µs matches golden)"


def test_residual_checksum():
    """Test residual values against golden checksum (rounded for stability)."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    golden = load_golden_reference("J1909_mini")
    if golden is None:
        return False, "golden reference not found"
    
    golden_dir = get_golden_dir()
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        return False, "mini dataset not found"
    
    # Check if golden has checksum data
    checksum_data = golden.get('residual_checksum')
    if not checksum_data:
        return None, "no checksum in golden (skip)"
    
    result = compute_residuals_simple(str(par), str(tim), verbose=False)
    
    # Round residuals to specified precision
    round_to_ns = checksum_data.get('round_to_ns', 10)
    actual_ns = np.array(result['residuals_us']) * 1000  # µs -> ns
    actual_rounded = [int(round(r / round_to_ns) * round_to_ns) for r in actual_ns[:10]]
    
    expected_rounded = checksum_data.get('first_10_rounded_ns', [])
    
    if len(expected_rounded) == 0:
        return None, "no expected values in checksum"
    
    # Compare
    errors = []
    for i, (act, exp) in enumerate(zip(actual_rounded, expected_rounded)):
        if act != exp:
            errors.append(f"[{i}]: {act} != {exp}")
    
    if errors:
        return False, f"checksum mismatch: {'; '.join(errors[:3])}"
    
    return True, f"OK (first {len(expected_rounded)} residuals match rounded)"


def test_residual_determinism():
    """Test that residual computation is deterministic."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    golden_dir = get_golden_dir()
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        return False, "mini dataset not found"
    
    # Run twice
    result1 = compute_residuals_simple(str(par), str(tim), verbose=False)
    result2 = compute_residuals_simple(str(par), str(tim), verbose=False)
    
    # Should be identical
    if not np.allclose(result1['residuals_us'], result2['residuals_us'], rtol=0, atol=1e-15):
        return False, "results differ between runs"
    
    if result1['weighted_rms_us'] != result2['weighted_rms_us']:
        return False, "weighted RMS differs between runs"
    
    return True, "OK (deterministic)"


def test_pint_cross_validation():
    """Cross-validate JUG results against PINT (optional).
    
    Only runs when --pint flag is passed or JUG_TEST_PINT=1 env var is set.
    
    Note: This test is informational. JUG and PINT may produce different
    residuals due to:
    - Different ephemeris versions (DE440 vs DE421)
    - Different clock correction handling
    - Different binary delay algorithms
    
    The test checks that both produce reasonable residual patterns (same
    number of TOAs, similar RMS magnitude) rather than exact agreement.
    """
    try:
        import pint.models
        import pint.toa
        import pint.residuals
    except ImportError:
        return None, "PINT not installed (optional)"
    
    golden_dir = get_golden_dir()
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        return False, "mini dataset not found"
    
    # JUG computation
    from jug.residuals.simple_calculator import compute_residuals_simple
    jug_result = compute_residuals_simple(str(par), str(tim), verbose=False)
    jug_residuals = np.array(jug_result['residuals_us'])
    
    # PINT computation
    try:
        # Suppress PINT debug/info logging
        import logging
        logging.getLogger('pint').setLevel(logging.ERROR)
        
        pint_model = pint.models.get_model(str(par))
        # Use planets=True for Shapiro delay computation
        pint_toas = pint.toa.get_TOAs(str(tim), planets=True)
        pint_res = pint.residuals.Residuals(pint_toas, pint_model)
        pint_residuals_us = pint_res.time_resids.to('us').value
    except Exception as e:
        return False, f"PINT failed: {e}"
    
    # Compare basic properties
    if len(jug_residuals) != len(pint_residuals_us):
        return False, f"n_toas differ: JUG={len(jug_residuals)}, PINT={len(pint_residuals_us)}"
    
    # Compare RMS magnitudes (should be same order of magnitude)
    jug_rms = np.sqrt(np.mean(jug_residuals**2))
    pint_rms = np.sqrt(np.mean(pint_residuals_us**2))
    
    # Allow RMS to differ by up to 10x (very lenient - just sanity check)
    rms_ratio = max(jug_rms, pint_rms) / max(min(jug_rms, pint_rms), 1e-10)
    if rms_ratio > 10:
        return False, f"RMS magnitudes differ too much: JUG={jug_rms:.2f}µs, PINT={pint_rms:.2f}µs (ratio={rms_ratio:.1f})"
    
    # Check that both have reasonable variance (not all zeros)
    if np.std(jug_residuals) < 1e-10:
        return False, "JUG residuals have zero variance"
    if np.std(pint_residuals_us) < 1e-10:
        return False, "PINT residuals have zero variance"
    
    # Center both (remove mean) for correlation comparison
    jug_centered = jug_residuals - np.mean(jug_residuals)
    pint_centered = pint_residuals_us - np.mean(pint_residuals_us)
    
    # Check correlation - may not be high due to different algorithms
    # This is informational only, not a hard requirement
    correlation = np.corrcoef(jug_centered, pint_centered)[0, 1]
    
    # Compute RMS difference for reporting
    rms_diff = np.sqrt(np.mean((jug_centered - pint_centered)**2))
    
    # Check std agreement
    jug_std = np.std(jug_residuals)
    pint_std = np.std(pint_residuals_us)
    std_rel_diff = abs(jug_std - pint_std) / pint_std
    
    # Report as informational - test passes if both codes produce reasonable residuals
    return True, f"OK (JUG_rms={jug_rms:.1f}µs, PINT_rms={pint_rms:.1f}µs, corr={correlation:.4f})"


def main():
    """Run all correctness tests."""
    parser = argparse.ArgumentParser(description="JUG correctness tests")
    parser.add_argument("--pint", action="store_true", help="Include PINT cross-validation")
    args = parser.parse_args()
    
    # Also check env var for PINT
    run_pint = args.pint or os.environ.get('JUG_TEST_PINT', '').lower() in ('1', 'true', 'yes')
    
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)
    
    tests = [
        ("Mini Dataset vs Golden", test_mini_dataset_correctness),
        ("Residual Checksum", test_residual_checksum),
        ("Determinism", test_residual_determinism),
    ]
    
    if run_pint:
        tests.append(("PINT Cross-Validation", test_pint_cross_validation))
    
    all_passed = True
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is None:
                continue  # Test skipped/not applicable
            passed, msg = result
            if passed is None:
                print(f"  [SKIP] {name}: {msg}")
            elif passed:
                print(f"  [PASS] {name}: {msg}")
            else:
                print(f"  [FAIL] {name}: {msg}")
                all_passed = False
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All correctness tests PASSED")
        return 0
    else:
        print("Some correctness tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
