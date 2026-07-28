"""Tests for shared par-parameter normalization to numeric model state."""

from __future__ import annotations

import numpy as np
import pytest

from jug.engine.session import TimingSession
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_cache,
    _compute_designmatrix_from_setup,
)
from jug.io.par_reader import normalize_model_params, parse_dec, parse_par_file, parse_ra
from tempo2_fixtures import get_tempo2_fixture

FIXTURE_ID = "epta_j1909_t2"


@pytest.fixture(scope="module")
def tempo2_fixture():
    return get_tempo2_fixture(FIXTURE_ID)


def test_parse_par_file_leaves_hms_raj_decj_as_strings(tempo2_fixture):
    params = parse_par_file(tempo2_fixture["par_path"])
    assert isinstance(params["RAJ"], str)
    assert isinstance(params["DECJ"], str)


def test_normalize_model_params_converts_hms_raj_decj_to_radians(tempo2_fixture):
    params = parse_par_file(tempo2_fixture["par_path"])
    normalize_model_params(
        params,
        compatibility="tempo2",
        context="test",
    )
    assert isinstance(params["RAJ"], float)
    assert isinstance(params["DECJ"], float)
    np.testing.assert_allclose(
        params["RAJ"], parse_ra("19:09:47.4335779"), rtol=1e-5, atol=0.0
    )
    np.testing.assert_allclose(
        params["DECJ"], parse_dec("-37:44:14.51584"), rtol=1e-5, atol=0.0
    )


def test_equatorial_hms_params_safe_for_tempo2_raj_decj_jax():
    """Regression: tempo2 stripped JAX astrometry requires numeric RAJ/DECJ."""
    params = {
        "RAJ": "00:30:27.4283723",
        "DECJ": "+04:51:39.70713",
        "F0": 205.53,
        "PEPOCH": 55000.0,
        "UNITS": "TDB",
    }
    normalize_model_params(params, compatibility="tempo2", context="test")
    from jug.residuals.tempo2.common import _raj_decj_rad_jax

    alpha, delta = _raj_decj_rad_jax(params)
    assert isinstance(params["RAJ"], float)
    assert isinstance(params["DECJ"], float)
    np.testing.assert_allclose(
        float(alpha), params["RAJ"], rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        float(delta), params["DECJ"], rtol=0.0, atol=0.0
    )


def test_timing_session_stores_numeric_raj_decj(tempo2_fixture):
    session = TimingSession(
        tempo2_fixture["par_path"],
        tempo2_fixture["tim_path"],
        compatibility="tempo2",
        verbose=False,
    )
    assert isinstance(session.params["RAJ"], float)
    assert isinstance(session.params["DECJ"], float)


def test_cached_session_designmatrix_accepts_hms_parfile(tempo2_fixture):
    session = TimingSession(
        tempo2_fixture["par_path"],
        tempo2_fixture["tim_path"],
        compatibility="tempo2",
        verbose=False,
    )
    cached_result = session.compute_residuals(subtract_tzr=False, force_recompute=True)

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    errors_us = np.array([toa.error_us for toa in session.toas_data])
    session_cached_data = {
        "dt_sec": cached_result["dt_sec"],
        "dt_sec_ld": cached_result.get("dt_sec_ld"),
        "tdb_mjd": cached_result["tdb_mjd"],
        "freq_bary_mhz": cached_result["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": errors_us,
        "toa_flags": [toa.flags for toa in session.toas_data],
        "roemer_shapiro_sec": cached_result.get("roemer_shapiro_sec"),
        "prebinary_delay_sec": cached_result.get("prebinary_delay_sec"),
        "ssb_obs_pos_ls": cached_result.get("ssb_obs_pos_ls"),
        "sw_geometry_pc": cached_result.get("sw_geometry_pc"),
        "jump_phase": cached_result.get("jump_phase"),
        "tzr_phase": cached_result.get("tzr_phase"),
        "term_diagnostics": cached_result.get("term_diagnostics"),
    }

    setup = _build_general_fit_setup_from_cache(
        session_cached_data,
        dict(session.params),
        ["RAJ"],
        compatibility="tempo2",
    )
    matrix = _compute_designmatrix_from_setup(setup, ["RAJ"])

    assert matrix.shape == (len(toas_mjd), 1)
    assert np.all(np.isfinite(matrix))


@pytest.mark.parametrize(
    "param", ["XDOT", "A1DOT", "PBDOT", "XPBDOT", "EDOT", "EPS1DOT", "EPS2DOT"]
)
def test_tempo2_rate_params_use_1e12_dual_convention(tmp_path, param):
    """Tempo2 readParfile.C parity: a written rate with |val|>1e-7 is in units
    of 1e-12 s^-1 and is scaled by 1e-12; an already-physical small value is
    left unchanged. Regression for XDOT/PBDOT/EDOT (formerly only EPS?DOT)."""
    header = "PSR TEST\nF0 100.0\nPEPOCH 55000\n"
    par = tmp_path / "rate.par"

    par.write_text(f"{header}{param} 0.011382\n")  # "large" -> * 1e-12
    assert parse_par_file(str(par))[param] == pytest.approx(0.011382e-12, rel=1e-12)

    par.write_text(f"{header}{param} 1.1382e-14\n")  # already physical -> as-is
    assert parse_par_file(str(par))[param] == pytest.approx(1.1382e-14, rel=1e-12)
