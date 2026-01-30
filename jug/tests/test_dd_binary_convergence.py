#!/usr/bin/env python3
"""
Test DD binary model fitting convergence.

This test verifies the fix for the J0614-3329 convergence issue where
fitting would require many clicks to converge due to a bug where
ELL1 binary model was used instead of DD for chi2 validation.

The fix ensures _compute_binary_delay routes to the correct binary model.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jug.engine.session import TimingSession


def test_dd_binary_convergence():
    """Test that DD binary pulsar fitting converges in a single fit call."""

    par_file = Path(__file__).parent.parent.parent / "data/pulsars/J0614-3329_tdb_wrong.par"
    tim_file = Path(__file__).parent.parent.parent / "data/pulsars/J0614-3329.tim"

    if not par_file.exists() or not tim_file.exists():
        print("SKIP: Test data files not found")
        return True

    # Create session
    session = TimingSession(par_file, tim_file, verbose=False)

    # Fit PB and T0 (the parameters that are wrong in the test file)
    result = session.fit_parameters(
        fit_params=['PB', 'T0'],
        max_iter=25,
        verbose=False
    )

    # Check convergence
    converged = result.get('converged', False)
    iterations = result.get('iterations', 0)
    final_rms = result.get('final_rms', float('inf'))

    print(f"\nDD Binary Convergence Test (J0614-3329)")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Final RMS: {final_rms:.6f} µs")

    # Success criteria:
    # 1. Must converge
    # 2. Must converge within 10 iterations (typically 4-5)
    # 3. Final RMS must be close to expected (~2.34 µs)
    assert converged, f"Fit did not converge after {iterations} iterations"
    assert iterations <= 10, f"Too many iterations: {iterations} (expected ≤ 10)"
    assert final_rms < 5.0, f"Final RMS too high: {final_rms:.6f} µs (expected < 5 µs)"

    print("  ✓ Test PASSED: DD binary fitting converges efficiently")
    return True


def test_dd_binary_fitting_all_params():
    """Test that DD binary fitting works with multiple parameter types."""

    par_file = Path(__file__).parent.parent.parent / "data/pulsars/J0614-3329_tdb_wrong.par"
    tim_file = Path(__file__).parent.parent.parent / "data/pulsars/J0614-3329.tim"

    if not par_file.exists() or not tim_file.exists():
        print("SKIP: Test data files not found")
        return True

    # Create session
    session = TimingSession(par_file, tim_file, verbose=False)

    # Fit multiple parameter types including binary
    result = session.fit_parameters(
        fit_params=['F0', 'F1', 'DM', 'PB', 'T0'],
        max_iter=25,
        verbose=False
    )

    converged = result.get('converged', False)
    iterations = result.get('iterations', 0)
    final_rms = result.get('final_rms', float('inf'))

    print(f"\nDD Binary Multi-Param Test (J0614-3329)")
    print(f"  Converged: {converged}")
    print(f"  Iterations: {iterations}")
    print(f"  Final RMS: {final_rms:.6f} µs")

    assert converged, f"Fit did not converge"
    assert iterations <= 10, f"Too many iterations: {iterations}"
    assert final_rms < 5.0, f"Final RMS too high: {final_rms:.6f} µs"

    print("  ✓ Test PASSED: Multi-param DD binary fitting works")
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("DD Binary Model Convergence Tests")
    print("=" * 70)

    all_passed = True

    try:
        all_passed &= test_dd_binary_convergence()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    try:
        all_passed &= test_dd_binary_fitting_all_params()
    except AssertionError as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
        sys.exit(1)
