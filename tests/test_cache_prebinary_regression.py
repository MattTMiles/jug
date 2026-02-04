#!/usr/bin/env python
"""
Regression test: Verify prebinary_delay_sec is properly cached and loaded.

This test guards against the GUI crash:
    "GeneralFitSetup.__init__() missing 1 required positional argument: 'prebinary_delay_sec'"

which occurred when _build_general_fit_setup_from_cache() didn't provide prebinary_delay_sec.

Run from repo root:
    python tests/test_cache_prebinary_regression.py

Expected output:
    All checks pass → PASS
    Any failure → FAIL with details

Environment variables for CI:
    JUG_TEST_J1713_PAR=/path/to/J1713+0747.par
    JUG_TEST_J1713_TIM=/path/to/J1713+0747.tim
"""

import sys
import numpy as np
from pathlib import Path

# Import test path utilities
try:
    from tests.test_paths import get_j1713_paths, skip_if_missing
except ImportError:
    # Running from tests/ directory
    from test_paths import get_j1713_paths, skip_if_missing

# Get paths from environment or defaults
PAR_FILE, TIM_FILE = get_j1713_paths()


def check_file_exists():
    """Check that test data files exist."""
    return skip_if_missing(PAR_FILE, TIM_FILE, "cache_prebinary")


def test_compute_residuals_returns_prebinary():
    """Test 1: compute_residuals_simple returns prebinary_delay_sec."""
    from jug.residuals.simple_calculator import compute_residuals_simple
    
    result = compute_residuals_simple(PAR_FILE, TIM_FILE, verbose=False)
    
    assert 'prebinary_delay_sec' in result, \
        "compute_residuals_simple() missing 'prebinary_delay_sec' in output"
    assert result['prebinary_delay_sec'] is not None, \
        "prebinary_delay_sec is None"
    assert isinstance(result['prebinary_delay_sec'], np.ndarray), \
        f"prebinary_delay_sec should be ndarray, got {type(result['prebinary_delay_sec'])}"
    assert len(result['prebinary_delay_sec']) == result['n_toas'], \
        f"prebinary_delay_sec length mismatch: {len(result['prebinary_delay_sec'])} vs {result['n_toas']} TOAs"
    
    print(f"  ✓ compute_residuals_simple returns prebinary_delay_sec (shape={result['prebinary_delay_sec'].shape})")
    return True


def test_session_caches_prebinary():
    """Test 2: TimingSession caches prebinary_delay_sec."""
    from jug.engine.session import TimingSession
    
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    
    # Compute residuals to populate cache
    result = session.compute_residuals(subtract_tzr=False, force_recompute=True)
    
    assert 'prebinary_delay_sec' in result, \
        "Session.compute_residuals() missing 'prebinary_delay_sec' in output"
    
    # Check the internal cache used by fit_parameters
    cached = session._cached_result_by_mode.get(False)  # subtract_tzr=False
    assert cached is not None, "Session cache is empty after compute"
    assert 'prebinary_delay_sec' in cached, \
        "Session cache missing 'prebinary_delay_sec'"
    assert cached['prebinary_delay_sec'] is not None, \
        "Cached prebinary_delay_sec is None"
    
    print(f"  ✓ TimingSession caches prebinary_delay_sec")
    return True


def test_cache_builder_provides_prebinary():
    """Test 3: _build_general_fit_setup_from_cache provides prebinary_delay_sec to GeneralFitSetup."""
    from jug.engine.session import TimingSession
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache
    import warnings
    
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    
    # Compute residuals to populate cache
    cached_result = session.compute_residuals(subtract_tzr=False, force_recompute=True)
    
    # Build session_cached_data dict (same as session.fit_parameters does)
    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    errors_us = np.array([toa.error_us for toa in session.toas_data])
    
    session_cached_data = {
        'dt_sec': cached_result['dt_sec'],
        'tdb_mjd': cached_result['tdb_mjd'],
        'freq_bary_mhz': cached_result['freq_bary_mhz'],
        'toas_mjd': toas_mjd,
        'errors_us': errors_us,
        'roemer_shapiro_sec': cached_result.get('roemer_shapiro_sec'),
        'prebinary_delay_sec': cached_result.get('prebinary_delay_sec'),
        'ssb_obs_pos_ls': cached_result.get('ssb_obs_pos_ls')
    }
    
    # Fit params that include binary (to trigger prebinary_delay_sec usage)
    fit_params = ['F0', 'F1', 'PB', 'A1']
    
    # Capture warnings - we should NOT see the fallback warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # This is the call that previously crashed with:
        # "GeneralFitSetup.__init__() missing 1 required positional argument: 'prebinary_delay_sec'"
        setup = _build_general_fit_setup_from_cache(
            session_cached_data,
            dict(session.params),
            fit_params
        )
    
    # Check no fallback warning was raised
    fallback_warnings = [x for x in w if 'prebinary_delay_sec' in str(x.message) and 'falling back' in str(x.message)]
    assert len(fallback_warnings) == 0, \
        f"Fallback warning raised - cache path not providing prebinary_delay_sec properly: {fallback_warnings}"
    
    # Check GeneralFitSetup has prebinary_delay_sec
    assert setup.prebinary_delay_sec is not None, \
        "GeneralFitSetup.prebinary_delay_sec is None"
    assert isinstance(setup.prebinary_delay_sec, np.ndarray), \
        f"setup.prebinary_delay_sec should be ndarray, got {type(setup.prebinary_delay_sec)}"
    
    n_toas = len(session.toas_data)
    assert len(setup.prebinary_delay_sec) == n_toas, \
        f"prebinary_delay_sec length mismatch: {len(setup.prebinary_delay_sec)} vs {n_toas} TOAs"
    
    print(f"  ✓ _build_general_fit_setup_from_cache provides prebinary_delay_sec (no fallback warning)")
    return True


def test_session_fit_parameters_works():
    """Test 4: Full session.fit_parameters works with binary params via cache path."""
    from jug.engine.session import TimingSession
    
    session = TimingSession(PAR_FILE, TIM_FILE, verbose=False)
    
    # Compute residuals first (populates cache)
    session.compute_residuals(subtract_tzr=False, force_recompute=True)
    
    # Fit with binary params - this triggers the cache path
    fit_params = ['F0', 'F1', 'PB', 'A1']
    
    try:
        result = session.fit_parameters(fit_params, verbose=False)
    except TypeError as e:
        if 'prebinary_delay_sec' in str(e):
            raise AssertionError(
                f"REGRESSION: GeneralFitSetup missing prebinary_delay_sec: {e}"
            )
        raise
    
    assert result is not None, "fit_parameters returned None"
    assert 'final_params' in result, "fit result missing final_params"
    assert result.get('converged', False) or result.get('iterations', 0) > 0, \
        "Fit did not run (no iterations)"
    
    print(f"  ✓ session.fit_parameters works with binary params ({result.get('iterations', '?')} iterations)")
    return True


def main():
    print("=" * 70)
    print("Regression Test: prebinary_delay_sec cache path")
    print("=" * 70)
    
    if not check_file_exists():
        print("\nSKIPPED: Test data not available")
        return 0
    
    print(f"\nUsing test data:")
    print(f"  PAR: {PAR_FILE}")
    print(f"  TIM: {TIM_FILE}")
    
    tests = [
        ("compute_residuals_simple returns prebinary", test_compute_residuals_returns_prebinary),
        ("TimingSession caches prebinary", test_session_caches_prebinary),
        ("Cache builder provides prebinary", test_cache_builder_provides_prebinary),
        ("session.fit_parameters works", test_session_fit_parameters_works),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            if test_fn():
                passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    if failed == 0:
        print(f"PASS: All {passed} tests passed")
        print("=" * 70)
        return 0
    else:
        print(f"FAIL: {failed}/{passed + failed} tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
