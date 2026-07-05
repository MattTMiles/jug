"""Tempo2-native site clock split and ``formBats`` arrival-time formation.

Implements ``getCorrectionTT`` + ``correctionTT_TB`` (``tt2tdb.C``) and
``formBats.C`` bat/bbat construction for ``compatibility='tempo2'``.

Diagnostic-only: production spin uses geometry ``model_mjd``, not
``model_clock`` from this module. See ``TEMPO2_NATIVE_CLOCK_STATUS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jug.utils.constants import C_KM_S, SECS_PER_DAY
from jug.utils.ifteph import IFTE_LC, IFTE_MJD0, IFTE_TEPH0_SEC, ifte_delta_t_mjd
from jug.utils.timescales import IFTE_K, IFTE_KM1, parse_timescale


@dataclass
class Tempo2ClockTerms:
    """Per-TOA tempo2 clock / arrival-time split (float64 MJD / seconds)."""

    sat_mjd: np.ndarray
    correction_tt_sec: np.ndarray
    correction_tt_tb_sec: np.ndarray
    model_clock_mjd: np.ndarray
    bat_mjd: np.ndarray
    bbat_mjd: np.ndarray
    shklovskii_sec: np.ndarray


def compute_site_clock_corrections_sec(
    mjd_utc: np.ndarray,
    *,
    obs_clocks: dict[str, dict],
    bipm_clock: dict,
    toas: list[Any],
    all_obs_codes: list[str],
    obs_clock_default: dict,
    time_offsets: np.ndarray | None = None,
) -> np.ndarray:
    """``getCorrectionTT`` — merged observatory + BIPM chain in seconds."""
    from jug.io.clock import interpolate_clock_vectorized

    mjd = np.asarray(mjd_utc, dtype=np.float64)
    out = np.zeros(len(toas), dtype=np.float64)
    for obs_code in all_obs_codes:
        idxs = [i for i, t in enumerate(toas) if t.observatory.lower() == obs_code]
        if not idxs:
            continue
        chain = obs_clocks.get(obs_code, obs_clock_default)
        mjd_obs = mjd[idxs]
        obs_corr = interpolate_clock_vectorized(chain, mjd_obs)
        bipm_corr = np.interp(mjd_obs, bipm_clock["mjd"], bipm_clock["offset"]) - 32.184
        out[idxs] = obs_corr + bipm_corr
    if time_offsets is not None:
        out = out + np.asarray(time_offsets, dtype=np.float64)
    return out


def compute_shklovskii_sec(
    bat_mjd: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """``secularMotion.C`` Shklovskii delay (seconds); zero without ``DSHK``."""
    if "DSHK" not in params:
        return np.zeros_like(bat_mjd, dtype=np.float64)
    if "PMRA" not in params and "PMDEC" not in params:
        return np.zeros_like(bat_mjd, dtype=np.float64)

    kpc2m = 3.08568025e19
    mas_yr2rad_s = 1.536281850e-16
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    dshk = float(params.get("DSHK", 0.0))
    pmra = float(params.get("PMRA", 0.0))
    pmdec = float(params.get("PMDEC", 0.0))
    bat = np.asarray(bat_mjd, dtype=np.float64)
    t0 = (bat - posepoch) * SECS_PER_DAY
    pm2 = (pmra * pmra + pmdec * pmdec) * mas_yr2rad_s * mas_yr2rad_s
    return (t0 * t0 / (2.0 * C_KM_S)) * (dshk * kpc2m) * pm2


def compute_correction_tt_tb_sec(
    mjd_tt: np.ndarray,
    *,
    observatory_earth_km: np.ndarray,
    earth_ssb_vel_km_s: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """``tt2tb.C`` ``correctionTT_TB`` for TCB (or TDB) par units."""
    mjd = np.asarray(mjd_tt, dtype=np.float64)
    delta_t = np.asarray(ifte_delta_t_mjd(mjd), dtype=np.float64)
    obs_km = np.asarray(observatory_earth_km, dtype=np.float64)
    earth_vel = np.asarray(earth_ssb_vel_km_s, dtype=np.float64)
    obs_term = np.sum(obs_km * earth_vel, axis=1) / (C_KM_S ** 2)
    obs_term = obs_term / (1.0 - IFTE_LC)

    units = parse_timescale(params)
    if units == "SI_UNITS":
        obs_term = obs_term / (IFTE_K * IFTE_K)
    else:
        obs_term = obs_term / IFTE_K

    correction_teph = IFTE_TEPH0_SEC + obs_term + delta_t / (1.0 - IFTE_LC)

    if units == "TDB":
        return correction_teph

    linear = IFTE_KM1 * (mjd - IFTE_MJD0) * SECS_PER_DAY
    return linear + IFTE_K * (correction_teph - IFTE_TEPH0_SEC)


def compute_formbats_arrival(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    correction_tt_tb_sec: np.ndarray,
    prebinary_delay_sec: np.ndarray,
    params: dict[str, Any],
) -> Tempo2ClockTerms:
    """Build tempo2 ``bat`` / ``bbat`` from ``formBats.C``."""
    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    tt_tb = np.asarray(correction_tt_tb_sec, dtype=np.float64)
    prebinary = np.asarray(prebinary_delay_sec, dtype=np.float64)

    clock_sec = tt + tt_tb
    model_clock = sat + clock_sec / SECS_PER_DAY

    bat = sat + (clock_sec - prebinary) / SECS_PER_DAY
    shk = compute_shklovskii_sec(bat, params)
    bbat = bat - shk / SECS_PER_DAY

    return Tempo2ClockTerms(
        sat_mjd=sat,
        correction_tt_sec=tt,
        correction_tt_tb_sec=tt_tb,
        model_clock_mjd=model_clock,
        bat_mjd=bat,
        bbat_mjd=bbat,
        shklovskii_sec=shk,
    )


def compute_tempo2_clock_terms(
    *,
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    observatory_earth_km: np.ndarray,
    earth_ssb_vel_km_s: np.ndarray,
    prebinary_delay_sec: np.ndarray,
    params: dict[str, Any],
) -> Tempo2ClockTerms:
    """Full native tempo2 clock split + ``formBats`` for one TOA batch."""
    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    mjd_tt = sat + tt / SECS_PER_DAY
    tt_tb = compute_correction_tt_tb_sec(
        mjd_tt,
        observatory_earth_km=observatory_earth_km,
        earth_ssb_vel_km_s=earth_ssb_vel_km_s,
        params=params,
    )
    return compute_formbats_arrival(
        sat,
        tt,
        tt_tb,
        prebinary_delay_sec,
        params,
    )
