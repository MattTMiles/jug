"""Bitwise parity: direct-ERFA troposphere elevation vs astropy reference.

The production troposphere path (_compute_apparent_elevation_deg_erfa)
replays astropy's ICRS->TETE ERFA call sequence without the frame
machinery. It must be bit-for-bit identical to the astropy frame-based
reference for all inputs; any divergence (e.g. after an astropy upgrade
changes the transform internals) must fail loudly here, at which point
either the ERFA path is updated to match or JUG_TROPO_ASTROPY=1 is used
until it is.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import astropy.units as u
from astropy.coordinates import EarthLocation

from jug.residuals.simple_calculator import (
    _compute_apparent_elevation_deg_astropy,
    _compute_apparent_elevation_deg_erfa,
)
from jug.utils.constants import OBSERVATORIES


@pytest.mark.parametrize("obs", ["arecibo", "gbt", "meerkat", "parkes"])
@pytest.mark.parametrize(
    "ra_rad,dec_rad",
    [
        (2.718204761303979, 0.17436481975522457),   # J1022+1001-like
        (5.016921493759775, -0.6580102749957517),   # southern source
        (0.013, 1.4),                               # near pole
    ],
)
def test_erfa_elevation_bitwise(obs, ra_rad, dec_rad):
    if obs not in OBSERVATORIES:
        pytest.skip(f"{obs} not in OBSERVATORIES")
    loc_km = OBSERVATORIES[obs]
    loc = EarthLocation.from_geocentric(
        loc_km[0] * u.km, loc_km[1] * u.km, loc_km[2] * u.km
    )
    rng = np.random.default_rng(12345)
    mjd = np.sort(rng.uniform(50000.0, 60500.0, 300))

    ref = _compute_apparent_elevation_deg_astropy(ra_rad, dec_rad, mjd, loc)
    fast = _compute_apparent_elevation_deg_erfa(ra_rad, dec_rad, mjd, loc)

    assert np.array_equal(ref, fast), (
        f"ERFA elevation path diverged from astropy reference: "
        f"max |diff| = {np.max(np.abs(ref - fast)):.3e} deg "
        f"({np.sum(ref != fast)}/{ref.size} elements)"
    )
