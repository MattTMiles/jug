"""Physical and astronomical constants for pulsar timing.

This module contains all physical constants, planetary parameters, and
observatory coordinates used throughout JUG.
"""

import numpy as np

# === Time and Distance Constants ===

SECS_PER_DAY = 86400.0
"""Seconds per day"""

C_KM_S = 299792.458
"""Speed of light in km/s"""

C_M_S = 299792458.0
"""Speed of light in m/s"""

T_SUN_SEC = 4.925490947e-6
"""Solar GM/c^3 in seconds (for Shapiro delay)"""

AU_KM = 149597870.7
"""Astronomical unit in km"""

AU_M = 149597870700.0
"""Astronomical unit in meters"""

K_DM_SEC = 1.0 / 2.41e-4
"""DM constant: K_DM = 1 / (2.41e-4) MHz^2 pc^-1 cm^3 s

Cold-plasma dispersion delay: delay = K_DM * DM / freq^2
where DM is in pc cm^-3 and freq is in MHz
"""

L_B = 1.550519768e-8
"""IAU TCB-TDB scaling factor for time scale conversion"""

# === Planetary Parameters ===

T_PLANET = {
    'jupiter': 4.702819050227708e-09,
    'saturn':  1.408128810019423e-09,
    'uranus':  2.150589551363761e-10,
    'neptune': 2.537311999186760e-10,
    'venus':   1.205680558494223e-11,
}
"""Planetary GM/c^3 in seconds (for planetary Shapiro delays)"""

# === Observatory Coordinates ===

OBSERVATORIES = {
    'meerkat': np.array([5109360.133, 2006852.586, -3238948.127]) / 1000,
    'parkes': np.array([-4554231.533, 2816759.046, -3454036.323]) / 1000,
    'gbt': np.array([882589.289, -4924872.368, 3943729.418]) / 1000,
}
"""Observatory coordinates in ITRF (km)

Format: {name: np.array([X, Y, Z])} where X, Y, Z are geocentric
coordinates in kilometers
"""

# === Parameter Precision Handling ===

HIGH_PRECISION_PARAMS = {
    'F0', 'F1', 'F2', 'F3',
    'PEPOCH', 'TZRMJD', 'POSEPOCH', 'DMEPOCH'
}
"""Parameters that require np.longdouble precision

These parameters must be parsed with maximum precision to maintain
microsecond-level accuracy in pulsar timing
"""

# === Conversion Factors ===

MJD_TO_JD = 2400000.5
"""Offset to convert MJD to JD: JD = MJD + 2400000.5"""

DEG_TO_RAD = np.pi / 180.0
"""Degrees to radians conversion factor"""

RAD_TO_DEG = 180.0 / np.pi
"""Radians to degrees conversion factor"""
