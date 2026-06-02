"""Tempo2-native JPL ephemeris access for TDB geometry (Phase B).

Uses ``jplephem`` SPK kernels (DE405 BSP from NAIF cache or TEMPO2 path) with
target/center pairs matching ``readEphemeris.C``.  Positions are returned in
light-seconds unless ``return_km=True``.

This module is used only by the tempo2 delay provider; the pint path keeps
Astropy ``solar_system_ephemeris``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from jug.utils.constants import C_KM_S

# JPL ``jpl_pleph`` numbering (tempo2 ``readEphemeris.C``)
_JPL_MERCURY = 1
_JPL_VENUS = 2
_JPL_EARTH = 3
_JPL_MARS = 4
_JPL_JUPITER = 5
_JPL_SATURN = 6
_JPL_URANUS = 7
_JPL_NEPTUNE = 8
_JPL_SUN = 11
_JPL_SSB = 12

# NAIF SPK segment pairs (center, target) for DE405-style kernels
_SSB = 0
_NAIF_SUN = 10
_NAIF_EMB = 3
_NAIF_EARTH = 399
_PLANET_BARY = {
    "mercury": 1,
    "venus": 2,
    "mars": 4,
    "jupiter": 5,
    "saturn": 6,
    "uranus": 7,
    "neptune": 8,
}


def _km_to_light_sec(km: np.ndarray) -> np.ndarray:
    return np.asarray(km, dtype=np.float64) / C_KM_S


def resolve_tempo2_ephemeris_path(ephem_name: str) -> str:
    """Resolve a par ``EPHEM`` keyword to an on-disk SPK/BSP path for jplephem."""
    name = str(ephem_name).lower().strip()
    if name.startswith("de") and len(name) >= 5 and name[2:5].isdigit():
        de = name[:5]
    else:
        de = "de405"

    from jug.residuals.simple_calculator import _resolve_ephemeris

    resolved = _resolve_ephemeris(de)
    if isinstance(resolved, str) and os.path.isfile(resolved):
        return resolved
    raise FileNotFoundError(f"Could not resolve tempo2 ephemeris path for {ephem_name!r}")


@lru_cache(maxsize=4)
def _open_spk(path: str):
    from jplephem.spk import SPK

    return SPK.open(path)


def _segment_pair(kernel, center: int, target: int):
    try:
        return kernel[center, target]
    except KeyError as exc:
        raise KeyError(
            f"Ephemeris segment ({center}, {target}) missing in {kernel.daf.locidw!r}"
        ) from exc


def _pos_vel_km(kernel, center: int, target: int, jd: float) -> tuple[np.ndarray, np.ndarray]:
    segment = _segment_pair(kernel, center, target)
    pos, vel = segment.compute_and_differentiate(jd)
    return np.asarray(pos[:3], dtype=np.float64), np.asarray(vel[:3], dtype=np.float64)


def earth_geocenter_from_ssb_km(kernel, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Earth geocenter position/velocity w.r.t. SSB (km, km/s)."""
    emb_pos, emb_vel = _pos_vel_km(kernel, _SSB, _NAIF_EMB, jd)
    earth_pos, earth_vel = _pos_vel_km(kernel, _NAIF_EMB, _NAIF_EARTH, jd)
    return emb_pos + earth_pos, emb_vel + earth_vel


def sun_from_ssb_km(kernel, jd: float) -> tuple[np.ndarray, np.ndarray]:
    return _pos_vel_km(kernel, _SSB, _NAIF_SUN, jd)


def planet_from_earth_km(kernel, planet: str, jd: float) -> tuple[np.ndarray, np.ndarray]:
    """Planet geocenter position/velocity (km, km/s), tempo2 ``jpl_pleph(N,3)``."""
    bary = _PLANET_BARY[planet]
    planet_ssb, planet_vel = _pos_vel_km(kernel, _SSB, bary, jd)
    earth_ssb, earth_vel = earth_geocenter_from_ssb_km(kernel, jd)
    return planet_ssb - earth_ssb, planet_vel - earth_vel


def mjd_to_jd(mjd: np.ndarray) -> np.ndarray:
    return np.asarray(mjd, dtype=np.float64) + 2400000.5


@dataclass
class Tempo2EphemerisState:
    """Vectorised ephemeris state at TDB epochs for one observatory site."""

    earth_ssb_ls: np.ndarray
    earth_ssb_vel_ls_s: np.ndarray
    sun_ssb_ls: np.ndarray
    obs_sun_ls: np.ndarray
    planets_obs_ls: dict[str, np.ndarray]


def compute_tempo2_ephemeris_state(
    tdb_mjd: np.ndarray,
    ssb_obs_pos_km: np.ndarray,
    *,
    ephem_path: str,
    planet_names: tuple[str, ...] = ("jupiter", "saturn", "uranus", "neptune", "venus"),
) -> Tempo2EphemerisState:
    """Compute tempo2-style SSB vectors using JPL SPK interpolation."""
    n = len(tdb_mjd)
    kernel = _open_spk(ephem_path)
    jd_arr = mjd_to_jd(tdb_mjd)

    earth_ssb_km = np.zeros((n, 3), dtype=np.float64)
    earth_vel_km_s = np.zeros((n, 3), dtype=np.float64)
    sun_ssb_km = np.zeros((n, 3), dtype=np.float64)

    for i, jd in enumerate(jd_arr):
        earth_ssb_km[i], earth_vel_km_s[i] = earth_geocenter_from_ssb_km(kernel, float(jd))
        sun_ssb_km[i], _ = sun_from_ssb_km(kernel, float(jd))

    ssb_obs_ls = _km_to_light_sec(ssb_obs_pos_km)
    earth_ssb_ls = _km_to_light_sec(earth_ssb_km)
    earth_ssb_vel_ls_s = _km_to_light_sec(earth_vel_km_s)
    sun_ssb_ls = _km_to_light_sec(sun_ssb_km)

    # Vector from observatory to Sun (km→ls), matching pint ``obs_sun_pos_km`` convention.
    observatory_earth_ls = ssb_obs_ls - earth_ssb_ls
    obs_sun_ls = sun_ssb_ls - ssb_obs_ls

    planets_obs_ls: dict[str, np.ndarray] = {}
    for planet in planet_names:
        arr = np.zeros((n, 3), dtype=np.float64)
        for i, jd in enumerate(jd_arr):
            pos_km, _ = planet_from_earth_km(kernel, planet, float(jd))
            obs_earth_km = ssb_obs_pos_km[i] - earth_ssb_km[i]
            # Planet position relative to observatory (pint convention).
            rsa_km = pos_km - obs_earth_km
            arr[i] = _km_to_light_sec(rsa_km)
        planets_obs_ls[planet] = arr

    return Tempo2EphemerisState(
        earth_ssb_ls=earth_ssb_ls,
        earth_ssb_vel_ls_s=earth_ssb_vel_ls_s,
        sun_ssb_ls=sun_ssb_ls,
        obs_sun_ls=obs_sun_ls,
        planets_obs_ls=planets_obs_ls,
    )


def close_ephemeris_cache() -> None:
    """Clear cached SPK handles (test hygiene)."""
    _open_spk.cache_clear()
