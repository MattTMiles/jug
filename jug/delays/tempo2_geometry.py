"""Tempo2-native Roemer/Shapiro geometry on TDB (Phase B).

Ports sign conventions and term ordering from tempo2 ``calculate_bclt.C`` and
``shapiro_delay.C``.  Internal vectors use light-seconds; exported delays are
seconds with the same sign convention as ``jug.delays.barycentric``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from jug.delays.barycentric import (
    compute_ecliptic_pulsar_direction,
    compute_pulsar_direction,
    rotate_equatorial_to_ecliptic,
)
from jug.io.par_reader import OBLIQUITY_ARCSEC, parse_dec, parse_ra
from jug.utils.constants import C_KM_S

# tempo2 ``tempo2.h``
GM_C3 = 4.925490947e-6
GMJ_C3 = 4.70255e-9
GMS_C3 = 1.47428e-8
GMU_C3 = 2.04269e-9
GMN_C3 = 2.45808e-9
GMV_C3 = 3.1764e-10
AULTSC = 499.00478364
PX_CONV = 1.74532925199432958e-2 / 3600.0e3  # mas → rad

# tempo2 hard-coded ecliptic obliquity (``ECLIPTIC_OBLIQUITY`` in tempo2.h)
TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC = 84381.4059

_PLANET_GM = {
    "jupiter": GMJ_C3,
    "saturn": GMS_C3,
    "uranus": GMU_C3,
    "neptune": GMN_C3,
    "venus": GMV_C3,
}


def tempo2_equ2ecl(
    vectors: np.ndarray,
    obl_arcsec: float = TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC,
) -> np.ndarray:
    """Rotate equatorial Cartesian vectors to ecliptic (tempo2 ``equ2ecl``)."""
    obl_rad = obl_arcsec * np.pi / (180.0 * 3600.0)
    return rotate_equatorial_to_ecliptic(np.asarray(vectors, dtype=np.float64), obl_rad)


def compute_tempo2_roemer_sec(
    ssb_obs_ls: np.ndarray,
    L_hat: np.ndarray,
    *,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
    vel_pulsar: np.ndarray | None = None,
    delt_centuries: np.ndarray | None = None,
) -> np.ndarray:
    """Roemer delay with tempo2 PM/parallax terms (``calculate_bclt.C``)."""
    ssb_obs_ls = np.asarray(ssb_obs_ls, dtype=np.float64)
    L_hat = np.asarray(L_hat, dtype=np.float64)

    rcos1 = np.sum(L_hat * ssb_obs_ls, axis=1)
    rr = np.sum(ssb_obs_ls * ssb_obs_ls, axis=1)

    roemer_ls = rcos1.copy()

    if vel_pulsar is not None and delt_centuries is not None:
        pmtrans_rcos2 = np.sum(vel_pulsar[None, :] * ssb_obs_ls, axis=1)
        pmtrans = float(np.linalg.norm(vel_pulsar))
        dt_pm = delt_centuries * pmtrans_rcos2
        dt_pmtt = -0.5 * pmtrans * pmtrans * (delt_centuries**2) * rcos1
        roemer_ls = roemer_ls + dt_pm + dt_pmtt
        if pmrv_rad_century != 0.0:
            dt_pmtr = -(delt_centuries**2) * pmrv_rad_century * pmtrans_rcos2
            roemer_ls = roemer_ls + dt_pmtr

    if parallax_mas != 0.0:
        dt_px = -0.5 * parallax_mas * PX_CONV * (rr - rcos1 * rcos1) / AULTSC
        roemer_ls = roemer_ls + dt_px

    # Match JUG/PINT geometric sign: negative projection onto pulsar direction.
    return -np.asarray(roemer_ls, dtype=np.float64)


def compute_tempo2_shapiro_sec(
    obs_body_ls: np.ndarray,
    L_hat: np.ndarray,
    gm_c3: float,
) -> np.ndarray:
    """Tempo2 Shapiro delay (``shapiro_delay.C``)."""
    obs_body_ls = np.asarray(obs_body_ls, dtype=np.float64)
    L_hat = np.asarray(L_hat, dtype=np.float64)
    r = np.linalg.norm(obs_body_ls, axis=1)
    rcostheta = np.sum(L_hat * obs_body_ls, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = -2.0 * gm_c3 * np.log(np.maximum(r - rcostheta, 1e-30) / AULTSC)
    return np.asarray(delay, dtype=np.float64)


def build_pulsar_direction(
    params: dict[str, Any],
    model_mjd: np.ndarray,
    *,
    use_native_ecliptic: bool,
    equatorial_to_ecliptic: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build pulsar unit vectors and PM velocity in the active coordinate frame."""
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    model_mjd_arr = np.asarray(model_mjd, dtype=np.float64)

    def _velocity_per_century() -> np.ndarray:
        """Finite-difference unit-vector derivative over one Julian century."""
        t0 = np.array([posepoch], dtype=np.float64)
        t1 = np.array([posepoch + 36525.0], dtype=np.float64)
        if use_native_ecliptic:
            d0 = compute_ecliptic_pulsar_direction(
                float(params["_ecliptic_lon_deg"]),
                float(params["_ecliptic_lat_deg"]),
                float(params.get("_ecliptic_pm_lon", 0.0)),
                float(params.get("_ecliptic_pm_lat", 0.0)),
                posepoch,
                t0,
            )[0]
            d1 = compute_ecliptic_pulsar_direction(
                float(params["_ecliptic_lon_deg"]),
                float(params["_ecliptic_lat_deg"]),
                float(params.get("_ecliptic_pm_lon", 0.0)),
                float(params.get("_ecliptic_pm_lat", 0.0)),
                posepoch,
                t1,
            )[0]
            return d1 - d0
        ra_rad = float(params.get("_raj_rad", parse_ra(params["RAJ"])))
        dec_rad = float(params.get("_decj_rad", parse_dec(params["DECJ"])))
        pmra_rad_day = params.get("PMRA", 0.0) * (np.pi / 180 / 3600000) / 365.25
        pmdec_rad_day = params.get("PMDEC", 0.0) * (np.pi / 180 / 3600000) / 365.25
        d0 = compute_pulsar_direction(
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, t0
        )[0]
        d1 = compute_pulsar_direction(
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, t1
        )[0]
        return d1 - d0

    if use_native_ecliptic:
        L_hat = compute_ecliptic_pulsar_direction(
            float(params["_ecliptic_lon_deg"]),
            float(params["_ecliptic_lat_deg"]),
            float(params.get("_ecliptic_pm_lon", 0.0)),
            float(params.get("_ecliptic_pm_lat", 0.0)),
            posepoch,
            model_mjd_arr,
        )
        vel = _velocity_per_century()
        pos = L_hat[0].copy()
    else:
        ra_rad = float(params.get("_raj_rad", parse_ra(params["RAJ"])))
        dec_rad = float(params.get("_decj_rad", parse_dec(params["DECJ"])))
        pmra_rad_day = params.get("PMRA", 0.0) * (np.pi / 180 / 3600000) / 365.25
        pmdec_rad_day = params.get("PMDEC", 0.0) * (np.pi / 180 / 3600000) / 365.25
        L_hat = compute_pulsar_direction(
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, model_mjd_arr
        )
        vel = _velocity_per_century()
        pos = L_hat[0].copy()
        if equatorial_to_ecliptic:
            obl = TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC * np.pi / (180.0 * 3600.0)
            L_hat = rotate_equatorial_to_ecliptic(L_hat, obl)
            pos = tempo2_equ2ecl(pos[None, :])[0]
            vel = tempo2_equ2ecl(vel[None, :])[0]

    return L_hat, pos, vel


def ssb_obs_light_seconds(ssb_obs_pos_km: np.ndarray) -> np.ndarray:
    return np.asarray(ssb_obs_pos_km, dtype=np.float64) / C_KM_S


def ecliptic_obliquity_rad(params: dict[str, Any], use_native_ecliptic: bool) -> float:
    if not use_native_ecliptic:
        return 0.0
    frame = str(params.get("_ecliptic_frame", "IERS2003")).upper()
    if frame in ("IERS2003", "DEFAULT"):
        return TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC * np.pi / (180.0 * 3600.0)
    arcsec = OBLIQUITY_ARCSEC.get(frame, TEMPO2_ECLIPTIC_OBLIQUITY_ARCSEC)
    return arcsec * np.pi / (180.0 * 3600.0)


def planet_shapiro_sec(
    planets_obs_ls: dict[str, np.ndarray],
    L_hat: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    """Sum planetary Shapiro contributions."""
    n = L_hat.shape[0]
    total = np.zeros(n, dtype=np.float64)
    if not enabled:
        return total
    for name, gm in _PLANET_GM.items():
        if name not in planets_obs_ls:
            continue
        total += compute_tempo2_shapiro_sec(planets_obs_ls[name], L_hat, gm)
    return total
