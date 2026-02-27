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
import glob
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


def _compare_jug_pint(par_path, tim_path, tol_pct=1.0):
    """Compare JUG vs PINT WRMS for a single pulsar.
    
    Uses EFAC/EQUAD-scaled errors for both (weighted_rms_scaled_us for JUG,
    rms_weighted() for PINT) and passes the correct ephemeris from the par
    file to PINT's get_TOAs().
    
    Returns (passed, message) tuple.
    """
    import pint.models
    import pint.toa
    import pint.residuals
    import logging
    logging.getLogger('pint').setLevel(logging.ERROR)
    
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    par_path, tim_path = str(par_path), str(tim_path)
    
    # JUG
    jug_result = compute_residuals_simple(par_path, tim_path, verbose=False)
    jug_wrms = jug_result['weighted_rms_scaled_us']
    n_jug = jug_result['n_toas']
    
    # PINT — use ephemeris from par file to avoid DE421 default mismatch
    pint_model = pint.models.get_model(par_path)
    ephem = pint_model.EPHEM.value if hasattr(pint_model, 'EPHEM') else None
    pint_toas = pint.toa.get_TOAs(tim_path, planets=True, ephem=ephem)
    pint_res = pint.residuals.Residuals(pint_toas, pint_model)
    pint_wrms = pint_res.rms_weighted().to('us').value
    n_pint = pint_toas.ntoas
    
    if n_jug != n_pint:
        return False, f"n_toas differ: JUG={n_jug}, PINT={n_pint}"
    
    delta = abs(jug_wrms - pint_wrms)
    pct = delta / pint_wrms * 100 if pint_wrms > 0 else 0
    
    if pct > tol_pct:
        return False, (f"WRMS mismatch: JUG={jug_wrms:.4f}µs, "
                       f"PINT={pint_wrms:.4f}µs ({pct:.2f}%)")
    
    return True, f"OK (JUG={jug_wrms:.4f}µs, PINT={pint_wrms:.4f}µs, Δ={pct:.3f}%)"


def test_pint_cross_validation():
    """Cross-validate JUG against PINT on golden mini dataset.
    
    Only runs when --pint flag is passed or JUG_TEST_PINT=1 env var is set.
    
    Compares EFAC/EQUAD-scaled WRMS between JUG and PINT, using the
    correct ephemeris from the par file for both codes.
    """
    try:
        import pint.models
    except ImportError:
        return None, "PINT not installed (optional)"
    
    golden_dir = get_golden_dir()
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if not par.exists() or not tim.exists():
        return False, "mini dataset not found"
    
    return _compare_jug_pint(par, tim, tol_pct=1.0)


def test_pint_parity_ng15yr():
    """Cross-validate JUG against PINT on all NANOGrav 15yr pulsars.
    
    Only runs when --pint-ng flag is passed or JUG_TEST_PINT_NG=1 env var
    is set. Requires NG 15yr data in data/pulsars/NG_data/NG_15yr_partim/.
    
    Compares EFAC/EQUAD-scaled WRMS (weighted_rms_scaled_us vs
    rms_weighted()) with the correct ephemeris from each par file.
    All 76 pulsars must match within 1%.
    """
    try:
        import pint.models
    except ImportError:
        return None, "PINT not installed (optional)"
    
    import glob
    ng_dir = repo_root / "data" / "pulsars" / "NG_data" / "NG_15yr_partim"
    par_files = sorted(glob.glob(str(ng_dir / "*.par")))
    
    if not par_files:
        return None, f"NG data not found in {ng_dir}"
    
    n_pass = 0
    n_fail = 0
    failures = []
    
    for par in par_files:
        name = os.path.basename(par).replace('.nb.par', '').split('_PINT')[0]
        tim = par.replace('.nb.par', '.nb.tim')
        if not os.path.exists(tim):
            continue
        
        try:
            passed, msg = _compare_jug_pint(par, tim, tol_pct=1.0)
            if passed:
                n_pass += 1
            else:
                n_fail += 1
                failures.append(f"{name}: {msg}")
        except Exception as e:
            n_fail += 1
            failures.append(f"{name}: {type(e).__name__}: {e}")
    
    total = n_pass + n_fail
    if n_fail > 0:
        return False, f"{n_fail}/{total} failed: {'; '.join(failures[:3])}"
    
    return True, f"OK ({n_pass}/{total} pulsars match PINT within 1%)"


def main():
    """Run all correctness tests."""
    parser = argparse.ArgumentParser(description="JUG correctness tests")
    parser.add_argument("--pint", action="store_true",
                        help="Include PINT cross-validation (golden mini dataset)")
    parser.add_argument("--pint-ng", action="store_true",
                        help="Run full NG 15yr parity test against PINT (76 pulsars)")
    args = parser.parse_args()
    
    run_pint = args.pint or os.environ.get('JUG_TEST_PINT', '').lower() in ('1', 'true', 'yes')
    run_pint_ng = args.pint_ng or os.environ.get('JUG_TEST_PINT_NG', '').lower() in ('1', 'true', 'yes')
    
    print("=" * 60)
    print("Correctness Tests")
    print("=" * 60)
    
    tests = [
        ("Mini Dataset vs Golden", test_mini_dataset_correctness),
        ("Residual Checksum", test_residual_checksum),
        ("Determinism", test_residual_determinism),
    ]
    
    if run_pint:
        tests.append(("PINT Cross-Validation (mini)", test_pint_cross_validation))
    if run_pint_ng:
        tests.append(("PINT Parity (NG 15yr, 76 pulsars)", test_pint_parity_ng15yr))
    
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
