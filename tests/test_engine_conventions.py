"""Tests for EngineConventionProfile (Phase B runtime conventions)."""

from __future__ import annotations

from jug.residuals.engine_conventions import EngineConventionProfile


def test_tempo2_implicit_defaults_on_tdb():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert profile.implicit_tempo2_defaults is True
    assert profile.dilatefreq is True
    assert profile.planet_shapiro is True
    assert profile.phase_mean_mode == "unweighted"
    assert profile.timeeph == "IF99"


def test_pint_explicit_defaults_on_tdb():
    params = {"UNITS": "TDB", "EPHEM": "DE405"}
    profile = EngineConventionProfile.from_params(params, "pint")
    assert profile.implicit_tempo2_defaults is False
    assert profile.dilatefreq is False
    assert profile.planet_shapiro is False
    assert profile.phase_mean_mode == "weighted"


def test_explicit_par_overrides_implicit():
    params = {"UNITS": "TDB", "DILATEFREQ": "N", "PLANET_SHAPIRO": "N"}
    profile = EngineConventionProfile.from_params(params, "tempo2")
    assert profile.dilatefreq is False
    assert profile.planet_shapiro is False
    assert profile._sources["DILATEFREQ"] == "par"
