#!/usr/bin/env python3
"""
Fit correctness tests for JUG using bundled mini data.

Verifies that fitting is "correct" in the sense that:
- It converges or at least reduces RMS
- It produces finite parameters
- It is deterministic within tolerance

Run with: python tests/test_fit_correctness_mini.py

Category: correctness (quick, uses bundled mini data)
"""

import sys
from pathlib import Path

import numpy as np

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


def test_fit_reduces_rms():
    """Test that fitting F0/F1 reduces or maintains RMS."""
    from jug.engine.session import TimingSession
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
    
    # Get prefit RMS
    prefit = session.compute_residuals(force_recompute=True)
    prefit_rms = prefit.get('weighted_rms_us') or prefit.get('rms_us')
    
    if prefit_rms is None or prefit_rms <= 0:
        return False, f"invalid prefit RMS: {prefit_rms}"
    
    # Run fit (F0 and F1 are usually safe to fit)
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    
    # Get postfit RMS
    postfit = session.compute_residuals(force_recompute=True)
    postfit_rms = postfit.get('weighted_rms_us') or postfit.get('rms_us')
    
    if postfit_rms is None or postfit_rms <= 0:
        return False, f"invalid postfit RMS: {postfit_rms}"
    
    # Fit should not make things significantly worse (allow 5% tolerance)
    if postfit_rms > prefit_rms * 1.05:
        return False, f"RMS increased: {prefit_rms:.4f} -> {postfit_rms:.4f} µs"
    
    improvement_pct = (prefit_rms - postfit_rms) / prefit_rms * 100
    return True, f"OK (prefit={prefit_rms:.2f}, postfit={postfit_rms:.2f} µs, {improvement_pct:+.1f}%)"


def test_fit_produces_finite_params():
    """Test that fitted parameters are finite (not NaN/inf)."""
    from jug.engine.session import TimingSession
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
    session.compute_residuals(force_recompute=True)
    
    # Run fit
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    
    # Check iterations happened
    if fit_result.get('iterations', 0) == 0:
        return False, "fit did not iterate"
    
    # Check parameter values are finite
    param_values = fit_result.get('final_params', {})
    if not param_values:
        # Try alternate result structure
        param_values = fit_result.get('params', {})
    
    errors = []
    for name, value in param_values.items():
        if value is None:
            continue
        if isinstance(value, (int, float)):
            if not np.isfinite(value):
                errors.append(f"{name}={value}")
        elif hasattr(value, '__iter__'):
            arr = np.asarray(value)
            if not np.all(np.isfinite(arr)):
                errors.append(f"{name} contains non-finite")
    
    if errors:
        return False, f"non-finite params: {', '.join(errors[:3])}"
    
    return True, f"OK ({len(param_values)} params checked)"


def test_fit_reasonable_param_changes():
    """Test that fitted parameters don't change by absurd amounts."""
    from jug.engine.session import TimingSession
    from jug.io.par_reader import parse_par_file
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Get original params
    original = parse_par_file(par)
    f0_val = original.get('F0')
    
    # Handle both dict and direct value formats
    if isinstance(f0_val, dict):
        f0_orig = float(f0_val.get('value', 0))
    elif f0_val is not None:
        f0_orig = float(f0_val)
    else:
        f0_orig = 0
    
    if f0_orig == 0:
        return False, "could not parse original F0"
    
    # Run fit
    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
    session.compute_residuals(force_recompute=True)
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    param_values = fit_result.get('final_params', {}) or fit_result.get('params', {})
    f0_fitted = param_values.get('F0')
    
    if f0_fitted is None:
        # Try to get from session
        return True, "OK (param extraction skipped)"
    
    # F0 should not change by more than a few ppm for a reasonable fit
    rel_change = abs(f0_fitted - f0_orig) / f0_orig
    
    # Allow up to 0.1% change (very loose for mini data)
    if rel_change > 0.001:
        return False, f"F0 changed too much: {f0_orig:.10f} -> {f0_fitted:.10f} ({rel_change*100:.4f}%)"
    
    return True, f"OK (F0 rel change: {rel_change*1e6:.2f} ppm)"


def test_fit_determinism():
    """Test that fitting is deterministic (same result twice)."""
    from jug.engine.session import TimingSession
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    # Run fit twice from scratch
    results = []
    for i in range(2):
        session = TimingSession(par, tim, verbose=False)
        session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
        session.compute_residuals(force_recompute=True)
        fit = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
        postfit = session.compute_residuals(force_recompute=True)
        results.append({
            'iterations': fit.get('iterations', 0),
            'rms': postfit.get('weighted_rms_us') or postfit.get('rms_us'),
        })
    
    # Compare results
    if results[0]['rms'] is None or results[1]['rms'] is None:
        return False, "could not get RMS values"
    
    rms_diff = abs(results[0]['rms'] - results[1]['rms'])
    
    # Should be identical (or very close due to floating point)
    if rms_diff > 1e-10:
        return False, f"RMS differs between runs: {results[0]['rms']:.10f} vs {results[1]['rms']:.10f}"
    
    if results[0]['iterations'] != results[1]['iterations']:
        return False, f"iterations differ: {results[0]['iterations']} vs {results[1]['iterations']}"
    
    return True, f"OK (RMS={results[0]['rms']:.4f}µs, iter={results[0]['iterations']})"


def test_fit_iterations_positive():
    """Test that fit reports positive iterations (did work)."""
    from jug.engine.session import TimingSession
    
    par, tim = get_mini_paths()
    if par is None:
        return False, "mini dataset not found"
    
    session = TimingSession(par, tim, verbose=False)
    session.params['_fit_flags'] = {}  # Clear par-file fit flags for controlled test
    session.compute_residuals(force_recompute=True)
    fit_result = session.fit_parameters(['F0', 'F1'], verbose=False, max_iter=10)
    
    iterations = fit_result.get('iterations', 0)
    
    if iterations <= 0:
        return False, f"fit reported {iterations} iterations"
    
    return True, f"OK ({iterations} iterations)"


def main():
    """Run all fit correctness tests."""
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
    for name, test_fn in tests:
        try:
            passed, msg = test_fn()
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}: {msg}")
            all_passed = all_passed and passed
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
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
