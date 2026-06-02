"""Tests for EngineConventionProfile (Phase B runtime conventions)."""

from __future__ import annotations

import pytest

from jug.residuals.engine_conventions import (
    EngineConventionProfile,
    resolve_engine_profile,
    validate_engine_profile_matches_compatibility,
)


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
