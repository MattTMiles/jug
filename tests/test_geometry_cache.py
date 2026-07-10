#!/usr/bin/env python3
"""
Tests for the geometry cache in compute_residuals_simple and TimingSession.

Verifies:
1. Cache is populated on first call
2. Cached call produces bit-identical residuals
3. Geometry arrays in cache are consistent with full result
4. Cache survives fit_parameters (not cleared)
5. Multiple fits in sequence all see the same geometry cache
6. Passing geometry_cache=None gives identical results to no-cache path
"""

import sys
from pathlib import Path

import numpy as np
import pytest

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))


def get_mini_paths():
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    if par.exists() and tim.exists():
        return str(par), str(tim)
    return None, None


def get_proper_paths():
    golden_dir = Path(__file__).parent / "data_golden"
    par = golden_dir / "J1909_proper.par"
    tim = golden_dir / "J1909_proper.tim"
    if par.exists() and tim.exists():
        return str(par), str(tim)
    return None, None


def _skip_if_tempo2_ephemeris_unavailable():
    try:
        from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path

        resolve_tempo2_ephemeris_path("de440")
    except FileNotFoundError:
        pytest.skip("tempo2 ephemeris files not available in this environment")


# ---------------------------------------------------------------------------
# compute_residuals_simple cache tests
# ---------------------------------------------------------------------------

def test_geometry_cache_populated_after_first_call():
    """Cache dict is empty before call, populated after."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    cache = {}
    assert len(cache) == 0

    compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)

    assert 'tdb_mjd' in cache, "tdb_mjd missing from cache"
    assert 'ssb_obs_pos_km' in cache, "ssb_obs_pos_km missing from cache"
    assert 'ssb_obs_vel_km_s' in cache, "ssb_obs_vel_km_s missing from cache"
    assert 'obs_sun_pos_km' in cache, "obs_sun_pos_km missing from cache"


def test_geometry_cache_gives_identical_residuals():
    """Second call (cache hit) produces bit-identical residuals to first."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    cache = {}
    result1 = compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)
    result2 = compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)

    np.testing.assert_array_equal(
        result1['residuals_us'], result2['residuals_us'],
        err_msg="Residuals differ between cold and warm cache calls"
    )
    np.testing.assert_array_equal(
        result1['tdb_mjd'], result2['tdb_mjd'],
        err_msg="tdb_mjd differs between cold and warm cache calls"
    )


def test_geometry_cache_matches_no_cache():
    """Cached result matches result computed without cache."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    # No-cache baseline
    result_no_cache = compute_residuals_simple(par, tim, verbose=False, geometry_cache=None)

    # Warm the cache
    cache = {}
    compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)
    # Use the warm cache
    result_cached = compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)

    np.testing.assert_array_almost_equal(
        result_no_cache['residuals_us'], result_cached['residuals_us'],
        decimal=10,
        err_msg="Cached residuals differ from no-cache residuals"
    )
    np.testing.assert_array_almost_equal(
        result_no_cache['tdb_mjd'], result_cached['tdb_mjd'],
        decimal=12,
        err_msg="tdb_mjd differs between cached and no-cache"
    )


def test_geometry_cache_arrays_consistent_with_result():
    """Arrays in geometry cache are identical to those in the result dict."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    cache = {}
    result = compute_residuals_simple(par, tim, verbose=False, geometry_cache=cache)

    np.testing.assert_array_almost_equal(
        cache['tdb_mjd'], result['tdb_mjd'],
        decimal=10,
        err_msg="cache tdb_mjd != result tdb_mjd"
    )
    if 'ssb_obs_pos_ls' in result:
        np.testing.assert_array_almost_equal(
            cache['ssb_obs_pos_km'],
            result['ssb_obs_pos_ls'] * 299792.458,  # ls -> km: ssb_obs_pos_ls * c_km_s
            decimal=4,  # float32/float64 rounding in km-scale numbers
            err_msg="cache ssb_obs_pos_km inconsistent with result ssb_obs_pos_ls",
        )


# ---------------------------------------------------------------------------
# TimingSession geometry cache tests
# ---------------------------------------------------------------------------

def test_session_geometry_cache_populated_after_compute():
    """Session._geometry_cache is populated after first compute_residuals."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    session = TimingSession(par, tim, verbose=False)
    assert len(session._geometry_cache) == 0, "Cache should start empty"

    session.compute_residuals()

    assert 'tdb_mjd' in session._geometry_cache
    assert 'ssb_obs_pos_km' in session._geometry_cache
    assert 'obs_sun_pos_km' in session._geometry_cache


def test_session_geometry_cache_survives_fit():
    """Session._geometry_cache is NOT cleared after fit_parameters."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    session = TimingSession(par, tim, verbose=False)
    session.compute_residuals()

    # Grab reference to the cache and a snapshot of tdb_mjd
    tdb_before = session._geometry_cache['tdb_mjd'].copy()

    session.fit_parameters()

    assert 'tdb_mjd' in session._geometry_cache, "Geometry cache was cleared by fit!"
    np.testing.assert_array_equal(
        session._geometry_cache['tdb_mjd'], tdb_before,
        err_msg="tdb_mjd changed after fit — cache was wrongly invalidated or recomputed"
    )


def test_session_geometry_cache_consistent_across_fits():
    """Geometry cache tdb_mjd is identical across two successive fits (J1909_proper, 100 TOAs)."""
    from jug.engine.session import TimingSession

    par, tim = get_proper_paths()
    if par is None:
        pytest.skip("J1909_proper dataset not found")

    session = TimingSession(par, tim, verbose=False)
    session.compute_residuals()

    tdb_after_first_compute = session._geometry_cache['tdb_mjd'].copy()

    session.fit_parameters()
    session.fit_parameters()

    np.testing.assert_array_equal(
        session._geometry_cache['tdb_mjd'], tdb_after_first_compute,
        err_msg="tdb_mjd changed across fits — geometry cache is being corrupted"
    )


def test_session_postfit_residuals_same_with_geometry_cache():
    """Post-fit residuals are the same whether geometry cache is used or not (J1909_proper).

    Regression guard: the geometry cache must not cause drift in residuals
    relative to a fresh session.
    """
    from jug.engine.session import TimingSession

    par, tim = get_proper_paths()
    if par is None:
        pytest.skip("J1909_proper dataset not found")

    # Session WITH geometry cache (normal path)
    s1 = TimingSession(par, tim, verbose=False)
    s1.fit_parameters()
    r1 = s1.compute_residuals(force_recompute=True)

    # Session WITHOUT geometry cache (force disable by replacing with None sentinel)
    # We simulate this by creating a fresh session and monkey-patching the cache
    # to a separate dict that won't be pre-populated (i.e. cold every time).
    s2 = TimingSession(par, tim, verbose=False)
    # Clear between every compute to simulate no geometry caching
    def fit_and_clear():
        s2._geometry_cache.clear()
        result = TimingSession.fit_parameters(s2)
        s2._geometry_cache.clear()
        return result

    s2.fit_parameters = fit_and_clear
    s2.fit_parameters()
    s2._geometry_cache.clear()
    r2 = s2.compute_residuals(force_recompute=True)

    np.testing.assert_array_almost_equal(
        r1['residuals_us'], r2['residuals_us'],
        decimal=8,
        err_msg="Post-fit residuals differ between cached and non-cached sessions"
    )


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
def test_session_residuals_at_params_forwards_compatibility(compatibility, monkeypatch):
    """residuals_at_params must pass session compatibility into cached setup builder."""
    from jug.engine import session as session_mod
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")

    captured = {}

    def _capture_build(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-setup")

    monkeypatch.setattr(session_mod, "_build_general_fit_setup_from_cache", _capture_build)

    session = TimingSession(par, tim, verbose=False, compatibility=compatibility)
    session._cached_result_by_mode[True] = {
        'dt_sec': np.zeros(20),
        'tdb_mjd': np.zeros(20),
        'freq_bary_mhz': np.ones(20),
    }

    with pytest.raises(RuntimeError, match="stop-after-setup"):
        session.residuals_at_params({'F0': float(session.params['F0'])}, subtract_tzr=True)

    assert captured.get('compatibility') == compatibility


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
def test_session_residuals_at_params_matches_override_path(compatibility):
    """Fast in-memory residuals_at_params agrees with compute_residuals override path."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")
    if compatibility == "tempo2":
        _skip_if_tempo2_ephemeris_unavailable()

    session = TimingSession(par, tim, verbose=False, compatibility=compatibility)
    session.compute_residuals(subtract_tzr=True)

    trial = {'F0': float(session.params['F0']) + 1e-12}
    slow = session.compute_residuals(params=trial, subtract_tzr=True)
    fast = session.residuals_at_params(trial, subtract_tzr=True)

    np.testing.assert_allclose(
        fast['residuals_us'],
        slow['residuals_us'],
        rtol=0.0,
        atol=6e-2,
        err_msg=(
            f"Fast residuals_at_params does not match canonical override residuals "
            f"(compatibility={compatibility})"
        ),
    )
    assert fast['used_fast_path'] is True
    assert fast['n_toas'] == slow['n_toas']


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
def test_session_residuals_at_params_avoids_temp_par_path(compatibility, monkeypatch):
    """residuals_at_params should not call _compute_residuals_with_params."""
    from jug.engine.session import TimingSession

    par, tim = get_mini_paths()
    if par is None:
        pytest.skip("Mini dataset not found")
    if compatibility == "tempo2":
        _skip_if_tempo2_ephemeris_unavailable()

    session = TimingSession(par, tim, verbose=False, compatibility=compatibility)
    session.compute_residuals(subtract_tzr=True)

    def _fail(*_args, **_kwargs):
        raise AssertionError("Unexpected slow temp-par path call")

    monkeypatch.setattr(session, "_compute_residuals_with_params", _fail)
    out = session.residuals_at_params({'F1': float(session.params.get('F1', 0.0))}, subtract_tzr=True)
    assert 'residuals_us' in out


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
