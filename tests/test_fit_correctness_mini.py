#!/usr/bin/env python3
"""
Fit correctness tests for JUG using bundled mini data.

Verifies that fitting is "correct" in the sense that:
- It converges or at least reduces RMS
- It produces finite parameters
- It is deterministic within tolerance

Run with: pytest tests/test_fit_correctness_mini.py -v
         python tests/test_fit_correctness_mini.py   (standalone)

Category: correctness (quick, uses bundled mini data)

NOTE: These tests use J1909_mini (20 TOAs). The fitter can be unstable on this
degenerate dataset when fitting many parameters simultaneously, but fitting only
F0/F1 is well-conditioned and reliable.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Pytest-compatible tests (use assert / pytest.skip)
# ---------------------------------------------------------------------------

def test_fit_reduces_rms():
    """Test that fitting F0/F1 reduces or maintains RMS."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("mini dataset not found")

    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test

    prefit = session.compute_residuals(force_recompute=True)
    prefit_rms = prefit.get('weighted_rms_us') or prefit.get('rms_us')

    assert prefit_rms is not None and prefit_rms > 0, f"invalid prefit RMS: {prefit_rms}"

    session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)

    postfit = session.compute_residuals(force_recompute=True)
    postfit_rms = postfit.get('weighted_rms_us') or postfit.get('rms_us')

    assert postfit_rms is not None and postfit_rms > 0, f"invalid postfit RMS: {postfit_rms}"
    # Fit should not make things significantly worse (allow 5% tolerance)
    assert postfit_rms <= prefit_rms * 1.05, (
        f"RMS increased: {prefit_rms:.4f} -> {postfit_rms:.4f} µs"
    )


def test_fit_produces_finite_params():
    """Test that fitted parameters are finite (not NaN/inf)."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("mini dataset not found")

    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}
    session.compute_residuals(force_recompute=True)

    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)

    assert fit_result.get('iterations', 0) > 0, "fit did not iterate"

    param_values = fit_result.get('final_params', {}) or fit_result.get('params', {})

    for name, value in param_values.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            assert np.isfinite(value), f"non-finite param {name}={value}"
        elif hasattr(value, '__iter__'):
            arr = np.asarray(value)
            assert np.all(np.isfinite(arr)), f"param {name} contains non-finite values"


def test_fit_reasonable_param_changes():
    """Test that fitted parameters don't change by absurd amounts."""
    from jug.engine.session import TimingSession
    from jug.io.par_reader import parse_par_file

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("mini dataset not found")

    original = parse_par_file(par)
    f0_val = original.get('F0')

    if isinstance(f0_val, dict):
        f0_orig = float(f0_val.get('value', 0))
    elif f0_val is not None:
        f0_orig = float(f0_val)
    else:
        f0_orig = 0

    if f0_orig == 0:
        pytest.skip("could not parse original F0")

    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}
    session.compute_residuals(force_recompute=True)
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    param_values = fit_result.get('final_params', {}) or fit_result.get('params', {})
    f0_fitted = param_values.get('F0')

    if f0_fitted is None:
        pytest.skip("F0 not in fit result (param extraction skipped)")

    rel_change = abs(f0_fitted - f0_orig) / f0_orig
    # Allow up to 0.1% change (very loose for mini data)
    assert rel_change <= 0.001, (
        f"F0 changed too much: {f0_orig:.10f} -> {f0_fitted:.10f} ({rel_change*100:.4f}%)"
    )


def test_fit_determinism():
    """Test that fitting is deterministic (same result twice)."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("mini dataset not found")

    rms_values = []
    iter_values = []
    for _ in range(2):
        session = TimingSession(par, tim, verbose=False)
        session.params['_fit_flags'] = {}
        session.compute_residuals(force_recompute=True)
        fit = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
        postfit = session.compute_residuals(force_recompute=True)
        rms = postfit.get('weighted_rms_us') or postfit.get('rms_us')
        assert rms is not None, "could not get RMS value"
        rms_values.append(rms)
        iter_values.append(fit.get('iterations', 0))

    assert abs(rms_values[0] - rms_values[1]) <= 1e-10, (
        f"RMS differs between runs: {rms_values[0]:.10f} vs {rms_values[1]:.10f}"
    )
    assert iter_values[0] == iter_values[1], (
        f"iteration count differs: {iter_values[0]} vs {iter_values[1]}"
    )


def test_fit_iterations_positive():
    """Test that fit reports positive iterations (did work)."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("mini dataset not found")

    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}
    session.compute_residuals(force_recompute=True)
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)

    iterations = fit_result.get('iterations', 0)
    assert iterations > 0, f"fit reported {iterations} iterations"


# ---------------------------------------------------------------------------
# Standalone runner (preserves original behaviour for python tests/... usage)
# ---------------------------------------------------------------------------

def _run_test(fn):
    """Run a single test function, return (passed, message)."""
    try:
        fn()
        return True, "OK"
    except pytest.skip.Exception as e:
        return None, str(e)
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    """Run all fit correctness tests standalone."""
    print("=" * 60)
    print("Fit Correctness Tests (Mini Data)")
    print("=" * 60)

    tests = [
        ("Fit Reduces/Maintains RMS", test_fit_reduces_rms),
        ("Fit Produces Finite Params", test_fit_produces_finite_params),
        ("Fit Reasonable Param Changes", test_fit_reasonable_param_changes),
        ("Fit Determinism", test_fit_determinism),
        ("Fit Iterations Positive", test_fit_iterations_positive),
    ]

    all_passed = True
    for name, fn in tests:
        passed, msg = _run_test(fn)
        if passed is None:
            print(f"  [SKIP] {name}: {msg}")
        elif passed:
            print(f"  [PASS] {name}: {msg}")
        else:
            print(f"  [FAIL] {name}: {msg}")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("All fit correctness tests PASSED")
        return 0
    else:
        print("Some fit correctness tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
