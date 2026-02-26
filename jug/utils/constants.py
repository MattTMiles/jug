"""Physical and astronomical constants for pulsar timing.

This module contains all physical constants, planetary parameters, and
observatory coordinates used throughout JUG.

All timing-relevant constants should be defined here to avoid duplication
and ensure consistency across the codebase.
"""

import numpy as np

# === Time Constants ===

SECS_PER_DAY = 86400.0
"""Seconds per day (exact)."""

SECS_PER_YEAR = 365.25 * SECS_PER_DAY
"""Seconds per Julian year (365.25 days)."""

# === Speed of Light ===

C_KM_S = 299792.458
"""Speed of light in km/s (exact, SI definition)."""

C_M_S = 299792458.0
"""Speed of light in m/s (exact, SI definition)."""

# === Solar System Constants ===

T_SUN = 4.925490947e-6
"""Solar mass parameter GM_sun/c^3 in seconds.

Used for Shapiro delay: Delta_S = -2 * T_SUN * M2 * ln(1 - s*sin(Phi)).
This is the IAU nominal value from Prsa et al. (2016)."""

AU_KM = 149597870.7
"""Astronomical unit in km (IAU 2012 exact definition)."""

AU_M = 149597870700.0
"""Astronomical unit in meters (IAU 2012 exact definition)."""

AU_PC = AU_M / 3.0856775814913673e16  # ~4.84813681e-6
"""Astronomical unit in parsecs (AU/pc)."""

PC_M = 3.0856775814913673e16
"""Parsec in meters (IAU 2015 exact definition)."""

KPC_TO_KM = PC_M  # 1 kpc = 1000 pc * (PC_M m/pc) / (1000 m/km) = PC_M km/kpc
"""Kiloparsec in kilometers. Numerically equals PC_M since the factors of 1000 cancel."""

PC_TO_LIGHT_SEC = PC_M / C_M_S
"""Parsec in light-seconds (~1.0292e8 lt-s/pc)."""

T_PLANET = {
    'jupiter': 4.702819050227708e-09,
    'saturn':  1.408128810019423e-09,
    'uranus':  2.150589551363761e-10,
    'neptune': 2.537311999186760e-10,
    'venus':   1.205680558494223e-11,
}
"""Planetary GM/c^3 in seconds (for planetary Shapiro delays).

Values from JPL DE440 ephemeris."""

# === Dispersion Constants ===

K_DM_SEC = 1.0 / 2.41e-4
"""DM dispersion constant in s MHz^2 pc^-1 cm^3.

Cold-plasma dispersion delay: delay = K_DM_SEC * DM / freq_MHz^2.
Value from Lorimer & Kramer (2004), Handbook of Pulsar Astronomy, p86 Note 1.
This equals ~4149.378, matching PINT. Tempo2 uses 4.148808e3 (0.014% smaller),
which is closer to the CODATA 2018 value of 4148.8064 derived from e^2/(2*pi*m_e*c).
The difference is absorbed into the fitted DM value.
"""

# === Time Scale Constants ===

L_B = 1.550519768e-8
"""IAU TCB-TDB scaling factor for time scale conversion.

TDB = TCB - L_B * (JD_TCB - T0) * 86400, from IAU 2006 Resolution B3."""

MJD_TO_JD = 2400000.5
"""Offset to convert MJD to JD: JD = MJD + 2400000.5."""

# === Angular Conversion ===

DEG_TO_RAD = np.pi / 180.0
"""Degrees to radians conversion factor."""

RAD_TO_DEG = 180.0 / np.pi
"""Radians to degrees conversion factor."""

MAS_PER_RAD = 180.0 * 3600.0 * 1000.0 / np.pi
"""Milliarcseconds per radian (~206264806)."""

HOURANGLE_PER_RAD = 12.0 / np.pi
"""Hour angles per radian (~3.819)."""

DEG_PER_RAD = RAD_TO_DEG
"""Degrees per radian (alias for RAD_TO_DEG)."""

# Backwards-compatible alias (prefer T_SUN in new code)
T_SUN_SEC = T_SUN

# === Observatory Coordinates ===

OBSERVATORIES = {
    'meerkat': np.array([5109360.133, 2006852.586, -3238948.127]) / 1000,
    'parkes': np.array([-4554231.500, 2816759.100, -3454036.300]) / 1000,
    'pks': np.array([-4554231.500, 2816759.100, -3454036.300]) / 1000,
    'pk': np.array([-4554231.500, 2816759.100, -3454036.300]) / 1000,
    '7': np.array([-4554231.500, 2816759.100, -3454036.300]) / 1000,
    'gbt': np.array([882589.289, -4924872.368, 3943729.418]) / 1000,
    '1': np.array([882589.289, -4924872.368, 3943729.418]) / 1000,
    'gb': np.array([882589.289, -4924872.368, 3943729.418]) / 1000,
    'arecibo': np.array([2390487.080, -5564731.357, 1994720.633]) / 1000,
    'ao': np.array([2390487.080, -5564731.357, 1994720.633]) / 1000,
    '3': np.array([2390487.080, -5564731.357, 1994720.633]) / 1000,
    'jodrell': np.array([3822625.769, -154105.255, 5086486.256]) / 1000,
    'jb': np.array([3822625.769, -154105.255, 5086486.256]) / 1000,
    '8': np.array([3822625.769, -154105.255, 5086486.256]) / 1000,
    'effelsberg': np.array([4033947.146, 486990.898, 4900431.067]) / 1000,
    'ef': np.array([4033947.146, 486990.898, 4900431.067]) / 1000,
    'eff': np.array([4033947.146, 486990.898, 4900431.067]) / 1000,
    'g': np.array([4033947.146, 486990.898, 4900431.067]) / 1000,
    'nancay': np.array([4324165.810, 165927.110, 4670132.830]) / 1000,
    'nc': np.array([4324165.810, 165927.110, 4670132.830]) / 1000,
    'f': np.array([4324165.810, 165927.110, 4670132.830]) / 1000,
    'wsrt': np.array([3828445.659, 445223.600, 5064921.568]) / 1000,
    'we': np.array([3828445.659, 445223.600, 5064921.568]) / 1000,
    'i': np.array([3828445.659, 445223.600, 5064921.568]) / 1000,
    'vla': np.array([-1601192.000, -5041981.400, 3554871.400]) / 1000,
    'vl': np.array([-1601192.000, -5041981.400, 3554871.400]) / 1000,
    '6': np.array([-1601192.000, -5041981.400, 3554871.400]) / 1000,
}
"""Observatory coordinates in ITRF (km)

Format: {name: np.array([X, Y, Z])} where X, Y, Z are geocentric
coordinates in kilometers
"""

# === Parameter Precision Handling ===

from jug.model.parameter_spec import get_high_precision_params as _get_hp

# Derived from the ParameterSpec registry (single source of truth).
# TZRMJD is always high-precision but is not a fittable parameter in the
# registry, so we add it explicitly here.
HIGH_PRECISION_PARAMS = _get_hp() | {'TZRMJD'}
"""Parameters that require np.longdouble precision

These parameters must be parsed with maximum precision to maintain
microsecond-level accuracy in pulsar timing
"""
