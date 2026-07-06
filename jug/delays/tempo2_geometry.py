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
from jug.utils.constants import C_KM_S, K_DM_SEC

# tempo2 ``tempo2.h``
GM_C3 = 4.925490947e-6
GMJ_C3 = 4.70255e-9
GMS_C3 = 1.47428e-8
GMU_C3 = 2.04269e-9
GMN_C3 = 2.45808e-9
GMV_C3 = 3.1764e-10
AULTSC = 499.00478364
PX_CONV = 1.74532925199432958e-2 / 3600.0e3  # mas → rad
# tempo2 ``tempo2.h`` dispersion / solar-wind constants
DM_CONST_TEMPO2 = 2.41e-4
DM_CONST_SI = 7.436e6
AU_DIST_M = 1.49598e11
SPEED_LIGHT_M = 299792458.0

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


def pmrv_rad_per_century(pmrv: float) -> float:
    """``calculate_bclt.C`` PMRV conversion to radians per century."""
    return float(pmrv) * (2.0 * np.pi / 360.0) / 36000.0


def psr_pos_at_delt(
    pos_pulsar: np.ndarray,
    vel_pulsar: np.ndarray,
    delt_centuries: float,
) -> np.ndarray:
    """Normalized ``psrPos`` at BCLT epoch (``shapiro_delay.C`` / ``dm_delays.C``)."""
    pos = np.asarray(pos_pulsar, dtype=np.float64) + float(delt_centuries) * np.asarray(
        vel_pulsar, dtype=np.float64
    )
    norm = float(np.linalg.norm(pos))
    if norm <= 0.0:
        return pos
    return pos / norm


def compute_tempo2_bclt_roemer_ls(
    ssb_obs_ls: np.ndarray,
    pos_pulsar: np.ndarray,
    vel_pulsar: np.ndarray,
    acc_pulsar: np.ndarray,
    *,
    delt_centuries: float,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
) -> float:
    """Positive Roemer in light-seconds (``calculate_bclt.C`` L129–152)."""
    rca = np.asarray(ssb_obs_ls, dtype=np.float64).reshape(3)
    pos = np.asarray(pos_pulsar, dtype=np.float64).reshape(3)
    vel = np.asarray(vel_pulsar, dtype=np.float64).reshape(3)
    acc = np.asarray(acc_pulsar, dtype=np.float64).reshape(3)
    delt = float(delt_centuries)

    rcos1 = float(np.dot(pos, rca))
    rr = float(np.dot(rca, rca))
    pmtrans_rcos2 = float(np.dot(vel, rca))
    pmtrans = float(np.linalg.norm(vel))
    dt_pm = delt * pmtrans_rcos2
    dt_pmtt = -0.5 * pmtrans * pmtrans * delt * delt * rcos1
    dt_acctrans = 0.5 * delt * delt * float(np.dot(acc, rca))
    dt_px = 0.0
    if parallax_mas != 0.0:
        dt_px = -0.5 * parallax_mas * PX_CONV * (rr - rcos1 * rcos1) / AULTSC
    dt_pmtr = -delt * delt * pmrv_rad_century * pmtrans_rcos2
    return rcos1 + dt_pm + dt_pmtt + dt_px + dt_pmtr + dt_acctrans


def compute_tempo2_roemer_sec(
    ssb_obs_ls: np.ndarray,
    L_hat: np.ndarray,
    *,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
    vel_pulsar: np.ndarray | None = None,
    delt_centuries: np.ndarray | None = None,
    pos_pulsar: np.ndarray | None = None,
    acc_pulsar: np.ndarray | None = None,
) -> np.ndarray:
    """Roemer delay with tempo2 PM/parallax terms (legacy batch API).

    When ``pos_pulsar`` is supplied, uses fixed POSEPOCH direction for ``rcos1``
    (BCLT convention). Otherwise falls back to ``L_hat`` for non-BCLT callers.
    """
    ssb_obs_ls = np.asarray(ssb_obs_ls, dtype=np.float64)
    n = ssb_obs_ls.shape[0]
    if pos_pulsar is not None and vel_pulsar is not None and delt_centuries is not None:
        pos = np.asarray(pos_pulsar, dtype=np.float64).reshape(3)
        vel = np.asarray(vel_pulsar, dtype=np.float64).reshape(3)
        acc = (
            np.zeros(3, dtype=np.float64)
            if acc_pulsar is None
            else np.asarray(acc_pulsar, dtype=np.float64).reshape(3)
        )
        pmrv = float(pmrv_rad_century)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            roemer_ls = compute_tempo2_bclt_roemer_ls(
                ssb_obs_ls[i],
                pos,
                vel,
                acc,
                delt_centuries=float(delt_centuries[i]),
                parallax_mas=parallax_mas,
                pmrv_rad_century=pmrv,
            )
            out[i] = -roemer_ls
        return out

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
    return -np.asarray(roemer_ls, dtype=np.float64)


def compute_tempo2_shapiro_sec(
    body_to_obs_ls: np.ndarray,
    psr_pos: np.ndarray,
    gm_c3: float,
) -> np.ndarray:
    """Tempo2 Shapiro delay (``shapiro_delay.C`` L78–80).

    ``body_to_obs_ls`` is the vector from the gravitating body to the observatory
    (tempo2 ``rsa``). ``psr_pos`` is the unit pulsar direction at ``delt``.
    """
    rsa = np.asarray(body_to_obs_ls, dtype=np.float64)
    pos = np.asarray(psr_pos, dtype=np.float64)
    if rsa.ndim == 1:
        rsa = rsa.reshape(1, 3)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    r = np.linalg.norm(rsa, axis=1)
    ctheta = np.sum(pos * rsa, axis=1) / np.maximum(r, 1e-30)
    with np.errstate(divide="ignore", invalid="ignore"):
        delay = -2.0 * gm_c3 * np.log(
            np.maximum(r / AULTSC * (1.0 + ctheta), 1e-30)
        )
    return np.asarray(delay, dtype=np.float64)


def build_tempo2_pulsar_vectors(
    params: dict[str, Any],
    *,
    use_native_ecliptic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port tempo2 ``vectorPulsar.C`` ``posPulsar`` / ``velPulsar`` / ``accPulsar``."""
    if use_native_ecliptic:
        lon = float(params["_ecliptic_lon_deg"]) * np.pi / 180.0
        lat = float(params["_ecliptic_lat_deg"]) * np.pi / 180.0
        ca, sa = np.cos(lon), np.sin(lon)
        cd, sd = np.cos(lat), np.sin(lat)
        pos = np.array([ca * cd, sa * cd, sd], dtype=np.float64)
        pm_lon = float(params.get("_ecliptic_pm_lon", params.get("PMRA", 0.0)))
        pm_lat = float(params.get("_ecliptic_pm_lat", params.get("PMDEC", 0.0)))
        pmra, pmdec = pm_lon, pm_lat
        lat_for_vel = lat
    else:
        alpha = float(params.get("_raj_rad", parse_ra(params["RAJ"])))
        delta = float(params.get("_decj_rad", parse_dec(params["DECJ"])))
        ca, sa = np.cos(alpha), np.sin(alpha)
        cd, sd = np.cos(delta), np.sin(delta)
        pos = np.array([ca * cd, sa * cd, sd], dtype=np.float64)
        pmra = float(params.get("PMRA", 0.0))
        pmdec = float(params.get("PMDEC", 0.0))
        lat_for_vel = delta

    convert = np.pi / 180.0 / 3600.0 / 1000.0 * 100.0
    cos_lat = np.cos(lat_for_vel)
    vel = convert * np.array(
        [
            -pmra / cos_lat * sa * cd - pmdec * ca * sd,
            pmra / cos_lat * ca * cd - pmdec * sa * sd,
            pmdec * cd,
        ],
        dtype=np.float64,
    )
    convert2 = convert * 100.0
    pmra2 = float(params.get("PMRA2", 0.0))
    pmdec2 = float(params.get("PMDEC2", 0.0))
    acc = convert2 * np.array(
        [
            -pmra2 / cos_lat * sa * cd - pmdec2 * ca * sd,
            pmra2 / cos_lat * ca * cd - pmdec2 * sa * sd,
            pmdec2 * cd,
        ],
        dtype=np.float64,
    )
    return pos, vel, acc


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
    psr_pos: np.ndarray,
    *,
    enabled: bool,
) -> np.ndarray:
    """Sum planetary Shapiro contributions at fixed ``psrPos(delt)``."""
    pos = np.asarray(psr_pos, dtype=np.float64)
    if pos.ndim == 1:
        pos = pos.reshape(1, 3)
    n = pos.shape[0]
    total = np.zeros(n, dtype=np.float64)
    if not enabled:
        return total
    for name, gm in _PLANET_GM.items():
        if name not in planets_obs_ls:
            continue
        obs_to_body = np.asarray(planets_obs_ls[name], dtype=np.float64)
        if obs_to_body.ndim == 1:
            obs_to_body = obs_to_body.reshape(1, 3)
        rsa = -obs_to_body
        total += compute_tempo2_shapiro_sec(rsa, pos, gm)
    return total


def _tempo2_spherical_solar_wind_sec(
    *,
    ctheta: float,
    r_ls: float,
    freq_hz: float,
) -> float:
    """``dm_delays.C`` L319 spherical_solar_wind factor (seconds)."""
    if freq_hz <= 1.0 or r_ls <= 0.0:
        return 0.0
    theta = float(np.arccos(np.clip(ctheta, -1.0, 1.0)))
    denom = r_ls * max(np.sqrt(max(1.0 - ctheta * ctheta, 0.0)), 1e-30)
    return (
        1.0e6
        * AU_DIST_M
        * AU_DIST_M
        / SPEED_LIGHT_M
        / DM_CONST_SI
        * theta
        / denom
        / freq_hz
        / freq_hz
    )


def compute_tempo2_dm_delays_sec(
    *,
    sat_mjd: float,
    freq_mhz: float,
    psr_pos: np.ndarray,
    obs_to_sun_ls: np.ndarray,
    earth_ssb_vel_km_s: np.ndarray,
    dm_val: float,
    ne_sw: float = 0.0,
    einstein_rate: float = 1.0,
    dilate_freq: bool = True,
    site_vel_km_s: np.ndarray | None = None,
) -> tuple[float, float]:
    """Port tempo2 ``dm_delays.C`` for one TOA at BCLT ``delt`` (host only).

    Uses fixed IFTE-era geometry; ``psr_pos`` is ``normalize(posPulsar + delt*vel)``.
    ``obs_to_sun_ls`` is observatory→Sun (JUG ``obs_sun_pos_ls``).
    """
    pos = np.asarray(psr_pos, dtype=np.float64).reshape(3)
    pos = pos / max(float(np.linalg.norm(pos)), 1e-30)
    obs_to_sun = np.asarray(obs_to_sun_ls, dtype=np.float64).reshape(3)
    rsa = -obs_to_sun
    vobs = np.asarray(earth_ssb_vel_km_s, dtype=np.float64).reshape(3) / C_KM_S
    if site_vel_km_s is not None:
        vobs = vobs + np.asarray(site_vel_km_s, dtype=np.float64).reshape(3) / C_KM_S
    r = float(np.linalg.norm(rsa))
    ctheta = float(np.dot(pos, rsa) / r) if r > 0 else 0.0
    voverc = float(np.dot(pos, vobs))
    freqf = float(freq_mhz) * 1.0e6 * (1.0 - voverc)
    if dilate_freq and freqf > 0.0 and einstein_rate != 0.0:
        freqf /= float(einstein_rate)
    if freqf <= 1.0:
        return 0.0, 0.0

    tdis1 = float(dm_val) * K_DM_SEC / ((freqf / 1.0e6) ** 2)
    if ne_sw == 0.0:
        return tdis1, 0.0

    spherical = _tempo2_spherical_solar_wind_sec(ctheta=ctheta, r_ls=r, freq_hz=freqf)
    tdis2 = float(ne_sw) * spherical
    return tdis1, tdis2
