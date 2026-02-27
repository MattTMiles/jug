#!/usr/bin/env python3
"""
Python API workflow tests for JUG.

Tests core API functionality without external data dependencies
by using the bundled mini dataset.

Run with: python tests/test_api_workflow.py
"""

import sys
from pathlib import Path

# Ensure jug module is importable
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def get_mini_paths():
    """Get paths to bundled mini dataset."""
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    
    if par.exists() and tim.exists():
        return str(par), str(tim)
    return None, None


def test_simple_calculator():
    """Test compute_residuals_simple API."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    result = compute_residuals_simple(par, tim, verbose=False)
    
    # Check expected keys
    required_keys = ['n_toas', 'residuals_us', 'weighted_rms_us', 'unweighted_rms_us']
    missing = [k for k in required_keys if k not in result]
    if missing:
        return False, f"missing keys: {missing}"
    
    # Check values are sensible
    if result['n_toas'] != 20:
        return False, f"expected 20 TOAs, got {result['n_toas']}"
    
    if len(result['residuals_us']) != 20:
        return False, f"residuals array wrong size"
    
    return True, f"OK (n_toas={result['n_toas']}, wRMS={result['weighted_rms_us']:.2f}µs)"


def test_timing_session():
    """Test TimingSession API."""
    from jug.engine.session import TimingSession
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    session = TimingSession(par, tim, verbose=False)
    result = session.compute_residuals()
    
    # Check basic result
    if result.get('n_toas', 0) != 20:
        return False, f"expected 20 TOAs, got {result.get('n_toas')}"
    
    return True, f"OK (n_toas={result['n_toas']}, RMS={result.get('rms_us', 0):.2f}µs)"


def test_par_reader():
    """Test par file reader."""
    from jug.io.par_reader import parse_par_file
    
    par, _ = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    params = parse_par_file(par)
    
    # Check we got a dict with parameters
    if not isinstance(params, dict):
        return False, f"expected dict, got {type(params)}"
    
    # Check for key parameters
    if 'PSR' not in params and 'PSRJ' not in params:
        return False, "missing PSR/PSRJ"
    
    if 'F0' not in params:
        return False, "missing F0"
    
    return True, f"OK ({len(params)} parameters)"


def test_tim_reader():
    """Test tim file reader."""
    from jug.io.tim_reader import parse_tim_file_mjds
    
    _, tim = get_mini_paths()
    if tim is None:
        return False, "mini dataset not found"
    
    toas = parse_tim_file_mjds(tim)
    
    if len(toas) != 20:
        return False, f"expected 20 TOAs, got {len(toas)}"
    
    return True, f"OK (n_toas={len(toas)})"


def main():
    """Run all API workflow tests."""
    print("=" * 60)
    print("API Workflow Tests")
    print("=" * 60)
    
    tests = [
        ("Simple Calculator", test_simple_calculator),
        ("Timing Session", test_timing_session),
        ("Par Reader", test_par_reader),
        ("Tim Reader", test_tim_reader),
    ]
    
    all_passed = True
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All API workflow tests PASSED")
        return 0
    else:
        print("Some API workflow tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
