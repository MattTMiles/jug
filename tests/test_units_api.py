"""Tests for PINT/Vela-compatible design-matrix unit API."""

from __future__ import annotations

from astropy import units as u

from jug.model.parameter_spec import get_fit_unit
from jug.utils.units import column_unit, fit_unit, validate_column_units


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

