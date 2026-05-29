"""Unit helpers for design-matrix API metadata.

This module intentionally focuses on the design-matrix boundary contract:
PINT/Vela-style parameter unit strings and parseable column unit strings.
"""

from __future__ import annotations

from functools import lru_cache

from astropy import constants as c
from astropy import units as u

from jug.model.parameter_spec import canonicalize_param_name, get_fit_unit


@lru_cache(maxsize=1)
def _ensure_custom_units_registered() -> None:
    """Register custom unit names used by PINT/Vela-style strings."""
    ls = u.def_unit("ls", c.c * u.s)
    dmu = u.def_unit("dmu", u.pc / u.cm**3)
    hourangle_second = u.def_unit("hourangle_second", u.hourangle / 3600.0)
    # MJD is a timescale label in timing models; represent it as day-scale.
    mjd = u.def_unit("MJD", u.day)
    u.add_enabled_units([ls, dmu, hourangle_second, mjd])


def _normalize_fit_unit(unit_str: str) -> str:
    """Normalize empty/dimensionless fit-unit labels."""
    cleaned = (unit_str or "").strip()
    return "1" if cleaned in {"", "1", "dimensionless"} else cleaned


def _parse_unit(unit_str: str) -> u.UnitBase:
    """Parse a fit-unit string with custom-unit support enabled."""
    _ensure_custom_units_registered()
    normalized = _normalize_fit_unit(unit_str)
    return u.dimensionless_unscaled if normalized == "1" else u.Unit(normalized)


def fit_unit(param_name: str) -> str:
    """Return PINT/Vela-style fit unit string for a parameter."""
    canonical = canonicalize_param_name(param_name)
    return _normalize_fit_unit(get_fit_unit(canonical))


def column_unit(param_name: str) -> str:
    """Return parseable design-matrix column unit string for a parameter."""
    unit = _parse_unit(fit_unit(param_name))
    if unit == u.dimensionless_unscaled:
        return "s"
    return str(u.s / unit)


def validate_column_units(labels: list[str]) -> list[str]:
    """Build and validate column-unit strings for a list of parameters."""
    units = [column_unit(label) for label in labels]
    _ensure_custom_units_registered()
    for unit_str in units:
        u.Unit(unit_str)
    return units

