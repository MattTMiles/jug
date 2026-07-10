"""Tempo2-native troposphere (``tropo.C``) for unified JAX path."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from functools import lru_cache
from typing import NamedTuple
from jug.utils.constants import C_M_S, SECS_PER_DAY

# Tempo2 ``NMF_hydrostatic`` / ``NMF_wet`` coefficient tables (tropo.C)
_AVGS_A = jnp.array(
    [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3],
    dtype=jnp.float64,
)
_AVGS_B = jnp.array(
    [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3],
    dtype=jnp.float64,
)
_AVGS_C = jnp.array(
    [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3],
    dtype=jnp.float64,
)
_AMPS_A = jnp.array(
    [0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5],
    dtype=jnp.float64,
)
_AMPS_B = jnp.array(
    [0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5],
    dtype=jnp.float64,
)
_AMPS_C = jnp.array(
    [0.0, 0.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5],
    dtype=jnp.float64,
)
_WET_A = jnp.array(
    [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4],
    dtype=jnp.float64,
)
_WET_B = jnp.array(
    [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3],
    dtype=jnp.float64,
)
_WET_C = jnp.array(
    [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2],
    dtype=jnp.float64,
)

class IersEopPacked(NamedTuple):
    """Static IERS Earth-orientation table for host-side interpolation."""

    mjd: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    dut1: np.ndarray


@lru_cache(maxsize=1)
def pack_iers_eop_jax() -> IersEopPacked:
    """Load Astropy IERS-B table once for tropo zenith host callbacks."""
    from astropy.utils.iers import IERS_B

    table = IERS_B.open()
    mjd = np.asarray(table["MJD"].value, dtype=np.float64)
    xp = np.asarray(table["PM_x"].value, dtype=np.float64)
    yp = np.asarray(table["PM_y"].value, dtype=np.float64)
    dut1 = np.asarray(table["UT1_UTC"].value, dtype=np.float64)
    return IersEopPacked(mjd=mjd, xp=xp, yp=yp, dut1=dut1)


_T2C_PHASE_MJD = 53398.0
_A_H = 2.53e-5
_B_H = 5.49e-3
_C_H = 1.14e-3


class TropoObsPacked(NamedTuple):
    """Static GRS80 site metadata for Tempo2 tropo."""

    latitude_rad: float
    longitude_rad: float
    height_m: float
    pressure_mbar: float


def pack_tropo_obs_static(
    *,
    obs_itrf_km: np.ndarray,
    pressure_mbar: float = 101.325,
) -> TropoObsPacked:
    """Pack geodetic site coordinates matching Tempo2 observatory tables."""
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    loc = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km,
    )
    return TropoObsPacked(
        latitude_rad=float(loc.lat.rad),
        longitude_rad=float(loc.lon.rad),
        height_m=float(loc.height.to(u.m).value),
        pressure_mbar=float(pressure_mbar),
    )


def _nmf_latitude_bins(site_latitude_rad: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Tempo2 tropo.C latitude binning (``ilat1``, ``ilat2``, ``frac``)."""
    abs_lat_deg = jnp.abs(jnp.degrees(site_latitude_rad))
    ilat1 = jnp.floor(abs_lat_deg / 15.0) - 1.0
    ilat1 = jnp.clip(ilat1, 0.0, 4.0).astype(jnp.int32)
    ilat2 = jnp.where(ilat1 >= 4, 4, jnp.minimum(ilat1 + 1, 4))
    same = ilat1 == ilat2
    frac = jnp.where(
        same,
        0.0,
        abs_lat_deg / 15.0 - 1.0 - ilat1.astype(jnp.float64),
    )
    return ilat1, ilat2, frac


def _nmf_hydrostatic_mapping(
    utc_mjd: jnp.ndarray,
    site_latitude_rad: jnp.ndarray,
    site_height_m: jnp.ndarray,
    source_elevation_rad: jnp.ndarray,
) -> jnp.ndarray:
    """Port of Tempo2 ``NMF_hydrostatic`` (returns basic + height correction)."""
    ilat1, ilat2, frac = _nmf_latitude_bins(site_latitude_rad)
    cos_phase = jnp.cos((utc_mjd - _T2C_PHASE_MJD) * 2.0 * jnp.pi / 365.25)
    cos_phase = cos_phase * jnp.where(site_latitude_rad < 0.0, -1.0, 1.0)

    def _interp(table: jnp.ndarray) -> jnp.ndarray:
        v1 = table[ilat1]
        v2 = table[ilat2]
        return jnp.where(ilat1 == ilat2, v1, frac * v2 + (1.0 - frac) * v1)

    a = _interp(_AVGS_A) - _interp(_AMPS_A) * cos_phase
    b = _interp(_AVGS_B) - _interp(_AMPS_B) * cos_phase
    c = _interp(_AVGS_C) - _interp(_AMPS_C) * cos_phase
    sin_el = jnp.sin(source_elevation_rad)
    basic = (1.0 + a / (1.0 + b / (1.0 + c))) / (
        sin_el + a / (sin_el + b / (sin_el + c))
    )
    height_correction = site_height_m * 1.0e-3 * (
        1.0 / sin_el
        - (1.0 + _A_H / (1.0 + _B_H / (1.0 + _C_H)))
        / (sin_el + _A_H / (sin_el + _B_H / (sin_el + _C_H)))
    )
    return basic + height_correction


def _nmf_wet_mapping(
    site_latitude_rad: jnp.ndarray,
    source_elevation_rad: jnp.ndarray,
) -> jnp.ndarray:
    """Port of Tempo2 ``NMF_wet`` (includes the ``cs`` typo at L205)."""
    ilat1, ilat2, frac = _nmf_latitude_bins(site_latitude_rad)
    sin_el = jnp.sin(source_elevation_rad)

    def _interp(table: jnp.ndarray) -> jnp.ndarray:
        v1 = table[ilat1]
        v2 = table[ilat2]
        return jnp.where(ilat1 == ilat2, v1, frac * v2 + (1.0 - frac) * v1)

    a = _interp(_WET_A)
    b = _interp(_WET_B)
    # Tempo2 L205 uses bs[ilat2] for the c interpolation branch.
    c = jnp.where(
        ilat1 == ilat2,
        _WET_C[ilat1],
        frac * _WET_B[ilat2] + (1.0 - frac) * _WET_C[ilat1],
    )
    return (1.0 + a / (1.0 + b / (1.0 + c))) / (sin_el + a / (sin_el + b / (sin_el + c)))


def tempo2_source_elevation_rad_jax(
    zenith_gcrs_m: jnp.ndarray,
    pos_pulsar_unit: jnp.ndarray,
    height_m: float,
) -> jnp.ndarray:
    """``asin(dot(zenith, posPulsar) / height_grs80)`` from tropo.C L426-428."""
    zenith = jnp.asarray(zenith_gcrs_m, dtype=jnp.float64)
    pos = jnp.asarray(pos_pulsar_unit, dtype=jnp.float64)
    dot = jnp.sum(zenith * pos, axis=-1)
    h = jnp.asarray(height_m, dtype=jnp.float64)
    return jnp.arcsin(jnp.clip(dot / h, -1.0, 1.0))


def _interp_eop_host(mjd: np.ndarray, table_mjd: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear EOP interpolation on host (IERS table MJD grid)."""
    mjd = np.asarray(mjd, dtype=np.float64)
    return np.interp(mjd, np.asarray(table_mjd, dtype=np.float64), np.asarray(values, dtype=np.float64))


def _host_zenith_gcrs_m(
    sat_mjd: np.ndarray,
    tt_sec: np.ndarray,
    latitude_rad: float,
    longitude_rad: float,
    height_m: float,
    utc_sec: np.ndarray | None = None,
) -> np.ndarray:
    """Transform geodetic zenith vector to GCRS (Tempo2 ``get_obsCoord_IAU2000B``)."""
    import erfa

    eop = pack_iers_eop_jax()
    arcsec_to_rad = np.pi / 180.0 / 3600.0

    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(tt_sec, dtype=np.float64)
    n = sat.size
    lat = float(latitude_rad)
    lon = float(longitude_rad)
    h = float(height_m)
    zenith_trs = np.array(
        [h * np.cos(lon) * np.cos(lat), h * np.sin(lon) * np.cos(lat), h * np.sin(lat)],
        dtype=np.float64,
    )
    if utc_sec is None:
        utc_mjd = sat
    else:
        utc_mjd = sat + np.asarray(utc_sec, dtype=np.float64) / SECS_PER_DAY
    dut1 = _interp_eop_host(utc_mjd, eop.mjd, eop.dut1)
    xp = _interp_eop_host(utc_mjd, eop.mjd, eop.xp) * arcsec_to_rad
    yp = _interp_eop_host(utc_mjd, eop.mjd, eop.yp) * arcsec_to_rad
    tt_jd = sat + tt / SECS_PER_DAY + 2400000.5
    tt_jd1 = np.floor(tt_jd).astype(np.int64)
    tt_jd2 = tt_jd - tt_jd1
    ut1_jd = utc_mjd + dut1 / SECS_PER_DAY + 2400000.5
    ut1_jd1 = np.floor(ut1_jd).astype(np.int64)
    ut1_jd2 = ut1_jd - ut1_jd1
    out = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        t2c = erfa.c2t00b(
            int(tt_jd1[i]), float(tt_jd2[i]),
            int(ut1_jd1[i]), float(ut1_jd2[i]),
            float(xp[i]), float(yp[i]),
        )
        out[i] = erfa.trxp(t2c, zenith_trs)
    return out


def compute_tempo2_zenith_gcrs_jax(
    sat_mjd: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    tropo: TropoObsPacked,
    *,
    utc_sec: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Zenith vector in GCRS (meters), batched via ``pure_callback``."""
    if utc_sec is None:
        utc_arg = jnp.zeros_like(jnp.asarray(sat_mjd, dtype=jnp.float64))
        use_utc = jnp.array(False)
    else:
        utc_arg = jnp.asarray(utc_sec, dtype=jnp.float64)
        use_utc = jnp.array(True)

    def callback(sat, tt, utc, use_utc_flag):
        utc_host = utc if bool(use_utc_flag) else None
        return _host_zenith_gcrs_m(
            sat,
            tt,
            float(tropo.latitude_rad),
            float(tropo.longitude_rad),
            float(tropo.height_m),
            utc_sec=utc_host,
        )

    return jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct(sat_mjd.shape + (3,), jnp.float64),
        sat_mjd,
        correction_tt_sec,
        utc_arg,
        use_utc,
        vmap_method="expand_dims",
    )


def tempo2_tropo_delay_jax(
    sat_mjd: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    source_elevation_rad: jnp.ndarray,
    tropo: TropoObsPacked,
    *,
    zenith_wet_delay_sec: jnp.ndarray | None = None,
    mapping_clock_sec: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Full Tempo2 ``compute_tropospheric_delays`` hydrostatic + wet path."""
    sat = jnp.asarray(sat_mjd, dtype=jnp.float64)
    tt = jnp.asarray(correction_tt_sec, dtype=jnp.float64)
    elev = jnp.asarray(source_elevation_rad, dtype=jnp.float64)
    lat = jnp.asarray(tropo.latitude_rad, dtype=jnp.float64)
    height = jnp.asarray(tropo.height_m, dtype=jnp.float64)
    pressure = jnp.asarray(tropo.pressure_mbar, dtype=jnp.float64)

    # Mapping epoch: SAT + UTC site-clock correction / SECDAY (``tropo.C`` L441-444).
    map_clock = (
        jnp.asarray(mapping_clock_sec, dtype=jnp.float64)
        if mapping_clock_sec is not None
        else tt
    )
    utc_mjd = sat + map_clock / SECS_PER_DAY
    mapping_h = _nmf_hydrostatic_mapping(utc_mjd, lat, height, elev)
    mapping_w = _nmf_wet_mapping(lat, elev)

    denom = 1.0 - 0.00266 * jnp.cos(lat) - 2.8e-7 * height
    zhd = 0.02268 * pressure / (jnp.asarray(C_M_S, dtype=jnp.float64) * denom)
    if zenith_wet_delay_sec is None:
        zwd = jnp.zeros_like(sat)
    else:
        zwd = jnp.asarray(zenith_wet_delay_sec, dtype=jnp.float64)
    return zhd * mapping_h + zwd * mapping_w


def compute_tempo2_tropo_delay_host(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    *,
    obs_itrf_km: np.ndarray,
    pos_pulsar: np.ndarray,
    pressure_mbar: float = 101.325,
    mapping_clock_sec: np.ndarray | None = None,
) -> np.ndarray:
    """Host batch wrapper around ``tempo2_tropo_delay_jax`` for legacy exports."""
    site = pack_tropo_obs_static(obs_itrf_km=obs_itrf_km, pressure_mbar=pressure_mbar)
    sat = jnp.asarray(sat_mjd, dtype=jnp.float64)
    tt = jnp.asarray(correction_tt_sec, dtype=jnp.float64)
    pos = jnp.asarray(pos_pulsar, dtype=jnp.float64)
    map_clock = None if mapping_clock_sec is None else jnp.asarray(
        mapping_clock_sec, dtype=jnp.float64
    )
    zenith = compute_tempo2_zenith_gcrs_jax(
        sat, tt, site, utc_sec=map_clock
    )
    elev = tempo2_source_elevation_rad_jax(zenith, pos, site.height_m)
    delay = tempo2_tropo_delay_jax(
        sat, tt, elev, site, mapping_clock_sec=map_clock
    )
    return np.asarray(jax.device_get(delay), dtype=np.float64)
