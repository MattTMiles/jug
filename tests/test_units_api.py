"""Tests for PINT/Vela-compatible design-matrix unit API."""

from __future__ import annotations

import numpy as np
from astropy import units as u

from jug.model.parameter_spec import get_fit_unit
from jug.utils.units import column_unit, fit_unit, validate_column_units
from jug.utils.units import fit_to_native_value, native_to_fit_value


def test_fit_unit_registry_examples():
    assert get_fit_unit("RAJ") == "hourangle"
    assert get_fit_unit("DECJ") == "deg"
    assert get_fit_unit("A1") == "ls"
    assert get_fit_unit("M2") == "solMass"
    assert get_fit_unit("DM1") == "pc cm^-3 yr^-1"
    assert get_fit_unit("F1") == "Hz/s^1"


def test_dynamic_family_fit_units():
    assert fit_unit("F3") == "Hz/s^3"
    assert fit_unit("DM4") == "pc cm^-3 yr^-4"
    assert fit_unit("FB0") == "1/s^1"
    assert fit_unit("FB2") == "1/s^3"
    assert fit_unit("JUMP42") == "s"


def test_column_unit_parseable_core_examples():
    labels = ["F0", "F1", "RAJ", "DECJ", "PMRA", "PX", "PB", "A1", "TASC", "M2"]
    units = validate_column_units(labels)
    assert len(units) == len(labels)
    for unit_str in units:
        # Round-trip parseability is the contract for API strings.
        u.Unit(unit_str)


def test_column_unit_astrometry_strings():
    assert column_unit("RAJ") == "s / hourangle"
    assert column_unit("DECJ") == "s / deg"


def test_native_fit_value_round_trip():
    raj_rad = 1.234567
    dec_rad = -0.5
    raj_fit = native_to_fit_value("RAJ", raj_rad)
    dec_fit = native_to_fit_value("DECJ", dec_rad)
    np.testing.assert_allclose(fit_to_native_value("RAJ", raj_fit), raj_rad, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(fit_to_native_value("DECJ", dec_fit), dec_rad, rtol=0.0, atol=1.0e-15)


def test_ecliptic_aliases_use_degree_fit_units_not_equatorial():
    """LAMBDA/BETA are ecliptic deg aliases of ELONG/ELAT, not RAJ/DECJ."""
    from jug.model.parameter_spec import canonicalize_param_name
    from jug.utils.units import native_derivative_to_fit_column

    assert canonicalize_param_name("LAMBDA") == "ELONG"
    assert canonicalize_param_name("BETA") == "ELAT"
    assert fit_unit("LAMBDA") == fit_unit("ELONG") == "deg"
    assert fit_unit("BETA") == fit_unit("ELAT") == "deg"
    assert fit_unit("LAMBDA") != "hourangle"

    assert native_to_fit_value("LAMBDA", 1.0) == native_to_fit_value("ELONG", 1.0) == 1.0
    assert native_to_fit_value("BETA", 1.0) == native_to_fit_value("ELAT", 1.0) == 1.0
    assert fit_to_native_value("LAMBDA", 1.0) == fit_to_native_value("ELONG", 1.0) == 1.0
    assert fit_to_native_value("BETA", 1.0) == fit_to_native_value("ELAT", 1.0) == 1.0

    assert abs(
        float(native_derivative_to_fit_column("LAMBDA", 1.0))
        - float(native_derivative_to_fit_column("ELONG", 1.0))
    ) < 1.0e-15
    assert abs(
        float(native_derivative_to_fit_column("BETA", 1.0))
        - float(native_derivative_to_fit_column("ELAT", 1.0))
    ) < 1.0e-15
    assert float(native_derivative_to_fit_column("LAMBDA", 1.0)) == 1.0
    assert float(native_derivative_to_fit_column("BETA", 1.0)) == 1.0

