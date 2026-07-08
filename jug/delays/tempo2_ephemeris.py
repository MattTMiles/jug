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
from typing import Any

import numpy as np

from jug.utils.constants import C_KM_S, SECS_PER_DAY
from jug.utils.timescales import IFTE_K
from jug.delays.tempo2_geometry import Tempo2ObservatoryState

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


def tempo2_read_ephemeris_au_scale(*, si_units: bool = True) -> float:
    """Scale SPK km vectors to match ``readEphemeris.C`` light-second storage.

    Tempo2 ``initialise.C`` sets ``psr->units = SI_UNITS`` (1).  In that mode
    ``readEphemeris.C`` multiplies ``one_au`` by ``IFTE_K`` before converting
    AU positions/velocities to light-seconds.  JPL SPK kernels use SI km
    directly, so the equivalent correction is a uniform ``IFTE_K`` on all
    ephemeris vectors (Earth/Sun/planets), not on ``get_obsCoord`` site vectors.
    """
    return float(IFTE_K) if si_units else 1.0


def _apply_tempo2_read_ephemeris_scale(
    arrays: list[np.ndarray],
    *,
    si_units: bool = True,
) -> None:
    """In-place ``readEphemeris.C`` SI_UNITS scaling for km / (km/s) vectors."""
    scale = tempo2_read_ephemeris_au_scale(si_units=si_units)
    if scale == 1.0:
        return
    for arr in arrays:
        np.multiply(arr, scale, out=arr)


def tempo2_geometry_epochs(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    correction_tt_teph_sec: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(site_mjd, ephemeris_mjd)`` for Tempo2 geometry.

    ``site_mjd`` follows ``get_obsCoord.C`` (``sat + getCorrectionTT``).
    ``ephemeris_mjd`` follows ``readEphemeris.C`` (adds ``correctionTT_Teph``).
    """
    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    site_mjd = sat + tt / SECS_PER_DAY
    if correction_tt_teph_sec is None:
        return site_mjd, site_mjd.copy()
    teph = np.asarray(correction_tt_teph_sec, dtype=np.float64)
    ephemeris_mjd = site_mjd + teph / SECS_PER_DAY
    return site_mjd, ephemeris_mjd


def tempo2_read_ephemeris_mjd(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    *,
    correction_tt_teph_sec: np.ndarray | None = None,
) -> np.ndarray:
    """MJD for ``readEphemeris.C`` (``sat + getCorrectionTT + correctionTT_Teph``)."""
    _site, ephemeris_mjd = tempo2_geometry_epochs(
        sat_mjd, correction_tt_sec, correction_tt_teph_sec
    )
    return ephemeris_mjd


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
    """Return SPK position (km) and velocity (km/s).

    ``jplephem`` ``compute_and_differentiate`` returns velocity in km/day; tempo2
    ``readEphemeris.C`` stores km/s in ``obsn[].earth_ssb[3:6]``.
    """
    segment = _segment_pair(kernel, center, target)
    pos, vel = segment.compute_and_differentiate(jd)
    pos_km = np.asarray(pos[:3], dtype=np.float64)
    vel_km_s = np.asarray(vel[:3], dtype=np.float64) / SECS_PER_DAY
    return pos_km, vel_km_s


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

    _apply_tempo2_read_ephemeris_scale([earth_ssb_km, earth_vel_km_s, sun_ssb_km])

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


def per_toa_obs_itrf_km(
    toas: list[Any],
    default_obs_itrf_km: np.ndarray,
) -> np.ndarray:
    """Per-TOA observatory ITRF positions ``(n, 3)`` for multi-observatory data.

    Falls back to ``default_obs_itrf_km`` for codes missing from the
    observatory table (mirrors ``_tempo2_tt2tb_geometry``).
    """
    from jug.utils.constants import OBSERVATORIES

    default = np.asarray(default_obs_itrf_km, dtype=np.float64).reshape(3)
    out = np.zeros((len(toas), 3), dtype=np.float64)
    for i, toa in enumerate(toas):
        out[i] = OBSERVATORIES.get(toa.observatory.lower(), default)
    return out


@dataclass(frozen=True)
class Tempo2GeometryBootstrap:
    """Converged Tempo2 host geometry after ``readEphemeris`` / ``tt2tb`` coupling."""

    state: Tempo2ObservatoryState
    site_mjd: np.ndarray
    ephemeris_mjd: np.ndarray
    correction_tt_tb_sec: np.ndarray
    correction_tt_teph_sec: np.ndarray
    iterations: int


def bootstrap_tempo2_observatory_state(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    obs_itrf_km: np.ndarray,
    *,
    ephem_path: str,
    params: dict[str, Any],
    max_iter: int = 8,
    tol: float = 1.0e-15,
    si_units: bool = True,
    t2c_method: str = "IAU2000B",
) -> Tempo2GeometryBootstrap:
    """Fixed-point bootstrap: ``tt2tb`` Teph ↔ ``readEphemeris`` epoch ↔ SPK state.

    Raises ``RuntimeError`` if ``correction_tt_teph_sec`` does not converge within
    ``max_iter`` iterations (internal loop tolerance ``tol`` seconds).
    """
    from jug.residuals.tempo2_clock import compute_correction_tt_tb_sec

    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    obs_itrf = np.asarray(obs_itrf_km, dtype=np.float64)
    site_mjd, ephemeris_mjd = tempo2_geometry_epochs(sat, tt)
    tt_teph = np.zeros_like(sat, dtype=np.float64)
    state = compute_tempo2_observatory_state(
        ephemeris_mjd,
        obs_itrf,
        ephem_path=ephem_path,
        site_mjd=site_mjd,
        si_units=si_units,
        t2c_method=t2c_method,
        sat_mjd=sat,
        correction_tt_sec=tt,
    )
    tt_tb = np.zeros_like(sat, dtype=np.float64)
    delta = np.inf
    iterations = 0

    for n in range(max_iter):
        iterations = n + 1
        tt_tb, tt_teph_new = compute_correction_tt_tb_sec(
            site_mjd,
            observatory_earth_km=state.observatory_earth_km[:, :3],
            earth_ssb_vel_km_s=state.earth_ssb_km[:, 3:6],
            params=params,
        )
        delta = float(np.max(np.abs(tt_teph_new - tt_teph)))
        if delta < tol:
            tt_teph = tt_teph_new
            break
        tt_teph = tt_teph_new
        _, ephemeris_mjd = tempo2_geometry_epochs(sat, tt, tt_teph)
        state = compute_tempo2_observatory_state(
            ephemeris_mjd,
            obs_itrf,
            ephem_path=ephem_path,
            site_mjd=site_mjd,
            si_units=si_units,
            t2c_method=t2c_method,
            sat_mjd=sat,
            correction_tt_sec=tt,
        )
    else:
        raise RuntimeError(
            "Tempo2 geometry bootstrap did not converge: "
            f"max |ΔTeph|={delta:.3e} s after {max_iter} iterations"
        )

    # Final ``tt2tb`` at converged geometry (``formBatsAll`` order).
    tt_tb, tt_teph = compute_correction_tt_tb_sec(
        site_mjd,
        observatory_earth_km=state.observatory_earth_km[:, :3],
        earth_ssb_vel_km_s=state.earth_ssb_km[:, 3:6],
        params=params,
    )
    _, ephemeris_mjd = tempo2_geometry_epochs(sat, tt, tt_teph)
    return Tempo2GeometryBootstrap(
        state=state,
        site_mjd=site_mjd,
        ephemeris_mjd=ephemeris_mjd,
        correction_tt_tb_sec=tt_tb,
        correction_tt_teph_sec=tt_teph,
        iterations=iterations,
    )


def compute_tempo2_observatory_state(
    ephem_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    *,
    ephem_path: str,
    site_mjd: np.ndarray | None = None,
    site_time_scale: str = "tt",
    si_units: bool = True,
    t2c_method: str = "IAU2000B",
    sat_mjd: np.ndarray | None = None,
    correction_tt_sec: np.ndarray | None = None,
    planet_names: tuple[str, ...] = (
        "mercury",
        "venus",
        "mars",
        "jupiter",
        "saturn",
        "uranus",
        "neptune",
    ),
) -> Tempo2ObservatoryState:
    """Compute Tempo2-style ``obsn[]`` vectors using JPL SPK + site motion.

    ``rca`` for BCLT is ``earth_ssb[:3] + observatory_earth[:3]`` (km), matching
    ``calculate_bclt.C`` L108–110.

    Ephemeris vectors (Earth/Sun/planets) are sampled at ``ephem_mjd`` and scaled
    per ``readEphemeris.C`` SI_UNITS (``IFTE_K`` on SPK km).

    Site transform (``get_obsCoord.C``):

    * ``t2c_method="IAU2000B"`` (default): Astropy
      ``EarthLocation.get_gcrs_posvel`` at ``site_mjd`` (``sat+TT``); this
      approximates ``get_obsCoord_IAU2000B`` and is **not** a line-by-line C
      port — pass ``site_mjd`` explicitly when ``ephem_mjd`` includes Teph.
    * ``t2c_method="TEMPO"``: line-by-line port of the legacy TEMPO branch
      (LMST + IAU1976 precession + 1980 nutation), which tempo2 uses for
      tempo1-emulation (``EPHVER<5``) and ``T2CMETHOD TEMPO`` par files.
      Requires ``sat_mjd`` and ``correction_tt_sec`` for the ``tai2ut1``
      UT1 argument.

    ``obs_itrf_km`` may be a single site ``(3,)`` or per-TOA sites ``(n, 3)``
    for multi-observatory datasets.
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation
    from astropy.time import Time

    n = len(ephem_mjd)
    kernel = _open_spk(ephem_path)
    jd_arr = mjd_to_jd(ephem_mjd)

    earth_ssb = np.zeros((n, 6), dtype=np.float64)
    sun_ssb = np.zeros((n, 6), dtype=np.float64)
    planet_ssb: dict[str, np.ndarray] = {}

    for i, jd in enumerate(jd_arr):
        pos, vel = earth_geocenter_from_ssb_km(kernel, float(jd))
        earth_ssb[i, :3], earth_ssb[i, 3:] = pos, vel
        spos, svel = sun_from_ssb_km(kernel, float(jd))
        sun_ssb[i, :3], sun_ssb[i, 3:] = spos, svel

    for planet in planet_names:
        if planet not in _PLANET_BARY:
            continue
        arr = np.zeros((n, 6), dtype=np.float64)
        bary = _PLANET_BARY[planet]
        for i, jd in enumerate(jd_arr):
            pos, vel = _pos_vel_km(kernel, _SSB, bary, float(jd))
            arr[i, :3], arr[i, 3:] = pos, vel
        planet_ssb[planet] = arr

    _apply_tempo2_read_ephemeris_scale(
        [earth_ssb, sun_ssb, *planet_ssb.values()],
        si_units=si_units,
    )

    obs_itrf = np.asarray(obs_itrf_km, dtype=np.float64)
    if obs_itrf.ndim == 1:
        obs_itrf = np.broadcast_to(obs_itrf.reshape(1, 3), (n, 3))

    if str(t2c_method).upper() == "TEMPO":
        from jug.delays.tempo_t2c import (
            compute_correction_ut1_sec,
            compute_tempo_t2c_observatory_earth,
        )

        if sat_mjd is None or correction_tt_sec is None:
            raise ValueError(
                "t2c_method='TEMPO' requires sat_mjd and correction_tt_sec "
                "for the tai2ut1 UT1 argument"
            )
        correction_ut1 = compute_correction_ut1_sec(sat_mjd, correction_tt_sec)
        observatory_earth = compute_tempo_t2c_observatory_earth(
            sat_mjd,
            obs_itrf,
            correction_ut1_sec=correction_ut1,
            nutation_mjd=np.asarray(ephem_mjd, dtype=np.float64),
        )
    else:
        site_epochs = np.asarray(
            ephem_mjd if site_mjd is None else site_mjd, dtype=np.float64
        )
        times = Time(site_epochs, format="mjd", scale=site_time_scale)
        obs_loc = EarthLocation.from_geocentric(
            obs_itrf[:, 0] * u.km, obs_itrf[:, 1] * u.km, obs_itrf[:, 2] * u.km
        )
        gcrs_pos, gcrs_vel = obs_loc.get_gcrs_posvel(obstime=times)
        observatory_earth = np.zeros((n, 6), dtype=np.float64)
        observatory_earth[:, 0] = gcrs_pos.x.to(u.km).value
        observatory_earth[:, 1] = gcrs_pos.y.to(u.km).value
        observatory_earth[:, 2] = gcrs_pos.z.to(u.km).value
        observatory_earth[:, 3] = gcrs_vel.x.to(u.km / u.s).value
        observatory_earth[:, 4] = gcrs_vel.y.to(u.km / u.s).value
        observatory_earth[:, 5] = gcrs_vel.z.to(u.km / u.s).value
    site_vel = observatory_earth[:, 3:6].copy()

    return Tempo2ObservatoryState(
        earth_ssb_km=earth_ssb,
        observatory_earth_km=observatory_earth,
        sun_ssb_km=sun_ssb,
        planet_ssb_km=planet_ssb,
        site_vel_km_s=site_vel,
    )
