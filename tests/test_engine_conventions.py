"""Tests for EngineConventionProfile (Phase B runtime conventions)."""

from __future__ import annotations

import pytest

from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
from jug.residuals.engine_conventions import (
    EngineConventionProfile,
    resolve_engine_profile,
    validate_engine_profile_matches_compatibility,
)
from jug.residuals.simple_calculator import _extract_binary_params


def test_tempo2_implicit_defaults_on_tdb():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert profile.implicit_tempo2_defaults is True
    assert profile.dilatefreq is True
    assert profile.planet_shapiro is True
    assert profile.correct_troposphere is True
    assert profile.phase_mean_mode == "unweighted"
    assert profile.timeeph == "IF99"


def test_pint_explicit_defaults_on_tdb():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    profile = EngineConventionProfile.from_params(params, "pint")
    assert profile.implicit_tempo2_defaults is False
    assert profile.dilatefreq is False
    assert profile.planet_shapiro is False
    assert profile.correct_troposphere is False
    assert profile.phase_mean_mode == "weighted"


def test_explicit_par_overrides_implicit():
    params = {
        "UNITS": "TDB",
        "DILATEFREQ": "N",
        "PLANET_SHAPIRO": "N",
        "CORRECT_TROPOSPHERE": "N",
    }
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert profile.dilatefreq is False
    assert profile.planet_shapiro is False
    assert profile.correct_troposphere is False
    assert profile._sources["DILATEFREQ"] == "par"
    assert profile._sources["CORRECT_TROPOSPHERE"] == "par"


def test_mismatched_engine_profile_raises():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    tempo2_profile = EngineConventionProfile.from_params(params, "tempo2")
    with pytest.raises(ValueError, match="does not match compatibility"):
        validate_engine_profile_matches_compatibility("pint", tempo2_profile)


def test_resolve_engine_profile_rejects_mixed_mode():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    tempo2_profile = EngineConventionProfile.from_params(params, "tempo2")
    with pytest.raises(ValueError, match="does not match compatibility"):
        resolve_engine_profile(params, "pint", engine_conventions=tempo2_profile)


def test_resolve_ne_sw_tempo2_implicit_default():
    params = {"UNITS": "TDB"}
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert resolve_ne_sw_cm3(params, profile) == 4.0


def test_resolve_ne_sw_explicit_zero_overrides_implicit():
    params = {"UNITS": "TDB", "NE_SW": 0.0}
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert resolve_ne_sw_cm3(params, profile) == 0.0


def test_resolve_ne_sw_pint_mode_defaults_to_zero():
    params = {"UNITS": "TDB"}
    profile = EngineConventionProfile.from_params(params, "pint")
    assert resolve_ne_sw_cm3(params, profile) == 0.0


def test_binary_t2_kin_kom_conversion_is_mode_specific():
    params = {
        "PEPOCH": 56000.0,
        "PB": 1.0,
        "BINARY": "T2",
        "KIN": 70.0,
        "KOM": 120.0,
    }

    pint_params = dict(params)
    tempo2_params = dict(params)

    _extract_binary_params(pint_params, verbose=False, compatibility="pint")
    _extract_binary_params(tempo2_params, verbose=False, compatibility="tempo2")

    assert pint_params["KIN"] != params["KIN"]
    assert pint_params["KOM"] != params["KOM"]
    assert "_t2_kin_kom_converted" in pint_params
    assert tempo2_params["KIN"] == params["KIN"]
    assert tempo2_params["KOM"] == params["KOM"]
    assert "_t2_kin_kom_converted" not in tempo2_params


def test_spin_derivatives_dispatch_by_compatibility():
    import numpy as np

    from jug.fitting.derivatives_spin import compute_spin_derivatives
    from jug.fitting.optimized_fitter import (
        _DELAY_DERIVATIVE_MODES,
        _is_delay_derivative_fd_mode,
        _normalize_fd_column_mode,
    )

    params = {"PEPOCH": 56000.0, "F0": 300.0, "F1": -1e-14}
    toas = np.array([56000.0, 56100.0, 56200.0])
    dt = (toas - params["PEPOCH"]) * 86400.0

    pint = compute_spin_derivatives(params, toas, ["F0", "F1"], compatibility="pint")
    tempo2 = compute_spin_derivatives(
        params, toas, ["F0", "F1"], compatibility="tempo2", dt_sec=dt
    )
    np.testing.assert_allclose(pint["F0"], tempo2["F0"], rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(pint["F1"], tempo2["F1"], rtol=0.0, atol=1e-15)

    assert _normalize_fd_column_mode(None, compatibility="tempo2") == "tempo2_delay"
    assert _normalize_fd_column_mode(None, compatibility="pint") == "delay_only"
    assert _DELAY_DERIVATIVE_MODES == frozenset({"tempo2_delay", "delay_only"})
    assert _is_delay_derivative_fd_mode("tempo2_delay")
    assert _is_delay_derivative_fd_mode("delay_only")
    assert not _is_delay_derivative_fd_mode("pint_phase_scaled")
