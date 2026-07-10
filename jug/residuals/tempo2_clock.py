"""Tempo2-native site clock split and ``formBats`` arrival-time formation.

Implements ``getCorrectionTT`` + ``correctionTT_TB`` (``tt2tdb.C``) and
``formBats.C`` bat/bbat construction for ``compatibility='tempo2'``.

This module supplies the production host clock chain consumed by
``run_tempo2_host_stage`` in ``jug.residuals.tempo2.host``. See
``PARITY_ROADMAP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jug.utils.constants import C_KM_S, SECS_PER_DAY
from jug.utils.ifteph import IFTE_LC, IFTE_MJD0, IFTE_TEPH0_SEC, ifte_delta_t_mjd
from jug.utils.timescales import IFTE_K, IFTE_KM1, is_tempo2_si_units, parse_timescale


@dataclass
class Tempo2ClockTerms:
    """Per-TOA tempo2 clock / arrival-time split (float64 MJD / seconds)."""

    sat_mjd: np.ndarray
    correction_tt_sec: np.ndarray
    correction_tt_teph_sec: np.ndarray
    correction_tt_tb_sec: np.ndarray
    einstein_rate: np.ndarray
    model_clock_mjd: np.ndarray
    bat_mjd: np.ndarray
    bbat_mjd: np.ndarray
    shklovskii_sec: np.ndarray


def _pack_clock_chain_tables(
    obs_chain: dict,
    bipm_clock: dict,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    """Pack merged observatory + BIPM clock tables for feedback evaluation."""
    mjd_tables: list[np.ndarray] = []
    offset_tables: list[np.ndarray] = []
    if "mjd" in obs_chain and "offset" in obs_chain:
        mjd_tables.append(np.asarray(obs_chain["mjd"], dtype=np.float64))
        offset_tables.append(np.asarray(obs_chain["offset"], dtype=np.float64))
    for link in obs_chain.get("links", []):
        mjd_tables.append(np.asarray(link["mjd"], dtype=np.float64))
        offset_tables.append(np.asarray(link["offset"], dtype=np.float64))
    return (
        tuple(mjd_tables),
        tuple(offset_tables),
        np.asarray(bipm_clock["mjd"], dtype=np.float64),
        np.asarray(bipm_clock["offset"], dtype=np.float64),
    )


def compute_tempo2_get_correction_tt_sec_vectorized(
    sat_mjd: np.ndarray,
    *,
    chain_mjd_tables: tuple[np.ndarray, ...],
    chain_offset_tables: tuple[np.ndarray, ...],
    bipm_mjd: np.ndarray,
    bipm_offset: np.ndarray,
    feedback_iters: int = 3,
) -> np.ndarray:
    """Tempo2 ``clkcorr.C`` UTC→TT chain with ``sat+corr/SECDAY`` feedback."""
    from jug.io.clock import interpolate_clock_vectorized

    sat = np.asarray(sat_mjd, dtype=np.float64)
    corr = np.zeros_like(sat)

    def one_iter(prev_corr: np.ndarray) -> np.ndarray:
        mjd_eval = sat + prev_corr / SECS_PER_DAY
        total = np.zeros_like(sat)
        for mjd_tab, off_tab in zip(chain_mjd_tables, chain_offset_tables):
            total = total + interpolate_clock_vectorized(
                {"mjd": mjd_tab, "offset": off_tab},
                mjd_eval,
            )
        bipm = np.interp(mjd_eval, bipm_mjd, bipm_offset) - 32.184
        return total + bipm

    for _ in range(max(1, int(feedback_iters))):
        corr = one_iter(corr)
    return corr


def compute_get_correction_tt_sec(
    toas: list[Any],
    *,
    obs_clocks: dict[str, dict],
    obs_clock_default: dict,
    bipm_clock: dict,
    all_obs_codes: list[str],
    time_offsets: np.ndarray | None = None,
    feedback_iters: int = 3,
) -> np.ndarray:
    """Tempo2 ``getCorrectionTT`` for ``formBats.C`` (seconds, per TOA).

    Uses Astropy UTC→TT (leap-second aware) with ``sat+corr/SECDAY`` feedback
    matching ``clkcorr.C`` evaluation time.
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    from jug.io.tim_reader import compute_tt_correction_sec_vectorized
    from jug.utils.constants import OBSERVATORIES

    n = len(toas)
    out = np.zeros(n, dtype=np.float64)
    for obs_code in all_obs_codes:
        idxs = [i for i, t in enumerate(toas) if t.observatory.lower() == obs_code]
        if not idxs:
            continue
        chain = obs_clocks.get(obs_code, obs_clock_default)
        loc_km = OBSERVATORIES.get(obs_code)
        if loc_km is None:
            continue
        loc = EarthLocation.from_geocentric(
            loc_km[0] * u.km, loc_km[1] * u.km, loc_km[2] * u.km
        )
        offsets = None if time_offsets is None else time_offsets[idxs]
        mjd_ints = [toas[i].mjd_int for i in idxs]
        mjd_fracs = [toas[i].mjd_frac for i in idxs]
        mjd_strings = [toas[i].mjd_str for i in idxs]
        corr = np.zeros(len(idxs), dtype=np.float64)
        for _ in range(max(1, int(feedback_iters))):
            corr = compute_tt_correction_sec_vectorized(
                mjd_ints,
                mjd_fracs,
                chain,
                bipm_clock,
                loc,
                time_offsets=offsets,
                mjd_strings=mjd_strings,
                clock_eval_offset_sec=corr,
            )
        out[idxs] = corr
    return out


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
) -> tuple[np.ndarray, np.ndarray]:
    """``tt2tb.C`` ``correctionTT_TB`` and Teph component for TCB/TDB par units."""
    mjd = np.asarray(mjd_tt, dtype=np.float64)
    delta_t = np.asarray(ifte_delta_t_mjd(mjd), dtype=np.float64)
    obs_km = np.asarray(observatory_earth_km, dtype=np.float64)
    earth_vel = np.asarray(earth_ssb_vel_km_s, dtype=np.float64)
    obs_term = np.sum(obs_km * earth_vel, axis=1) / (C_KM_S ** 2)
    obs_term = obs_term / (1.0 - IFTE_LC)

    units = parse_timescale(params)
    if is_tempo2_si_units(units):
        obs_term = obs_term / (IFTE_K * IFTE_K)
    else:
        obs_term = obs_term / IFTE_K

    correction_teph = IFTE_TEPH0_SEC + obs_term + delta_t / (1.0 - IFTE_LC)

    if str(units).upper() == "TDB":
        return correction_teph, correction_teph

    linear = IFTE_KM1 * (mjd - IFTE_MJD0) * SECS_PER_DAY
    return linear + IFTE_K * (correction_teph - IFTE_TEPH0_SEC), correction_teph


def compute_formbats_arrival_from_components(
    sat_mjd,
    correction_tt_sec,
    correction_tt_teph_sec,
    correction_tt_tb_sec,
    einstein_rate,
    model_clock_mjd,
    tropo_sec,
    roemer_sec,
    shapiro_delay_sec,
    tdis1_sec,
    tdis2_sec,
    params,
) -> tuple[Tempo2ClockTerms, np.ndarray]:
    """Split longdouble ``formBats.C`` assembly (diagnostic / strict path)."""
    SECS_PER_DAY_LD = np.longdouble(86400.0)
    tt = np.asarray(correction_tt_sec, dtype=np.longdouble)
    rest = (
        np.asarray(correction_tt_tb_sec, dtype=np.longdouble)
        - np.asarray(tropo_sec, dtype=np.longdouble)
        + np.asarray(roemer_sec, dtype=np.longdouble)
        - np.asarray(shapiro_delay_sec, dtype=np.longdouble)
        - np.asarray(tdis1_sec, dtype=np.longdouble)
        - np.asarray(tdis2_sec, dtype=np.longdouble)
    )
    sat = np.asarray(sat_mjd, dtype=np.longdouble)
    bat_corr = tt / SECS_PER_DAY_LD + rest / SECS_PER_DAY_LD
    bat = sat + tt / SECS_PER_DAY_LD + rest / SECS_PER_DAY_LD
    shk = compute_shklovskii_sec(np.asarray(bat, dtype=np.float64), params)
    bbat = bat - np.asarray(shk, dtype=np.longdouble) / SECS_PER_DAY_LD
    terms = Tempo2ClockTerms(
        sat_mjd=np.asarray(sat, dtype=np.float64),
        correction_tt_sec=np.asarray(correction_tt_sec, dtype=np.float64),
        correction_tt_teph_sec=np.asarray(correction_tt_teph_sec, dtype=np.float64),
        correction_tt_tb_sec=np.asarray(correction_tt_tb_sec, dtype=np.float64),
        einstein_rate=np.asarray(einstein_rate, dtype=np.float64),
        model_clock_mjd=np.asarray(model_clock_mjd, dtype=np.float64),
        bat_mjd=np.asarray(bat, dtype=np.float64),
        bbat_mjd=np.asarray(bbat, dtype=np.float64),
        shklovskii_sec=np.asarray(shk, dtype=np.float64),
    )
    return terms, np.asarray(bat_corr, dtype=np.float64)


def compute_formbats_arrival(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    correction_tt_tb_sec: np.ndarray,
    prebinary_delay_sec: np.ndarray,
    params: dict[str, Any],
    *,
    correction_tt_teph_sec: np.ndarray | None = None,
    einstein_rate: np.ndarray | None = None,
) -> Tempo2ClockTerms:
    """Build tempo2 ``bat`` / ``bbat`` from ``formBats.C``."""
    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    tt_tb = np.asarray(correction_tt_tb_sec, dtype=np.float64)
    prebinary = np.asarray(prebinary_delay_sec, dtype=np.float64)
    tt_teph = (
        np.asarray(correction_tt_teph_sec, dtype=np.float64)
        if correction_tt_teph_sec is not None
        else tt_tb.copy()
    )
    if einstein_rate is None:
        from jug.delays.barycentric import compute_einstein_rate
        from jug.delays.tempo2_geometry import tempo2_dilate_freq_enabled

        dilate = tempo2_dilate_freq_enabled(params)
        if dilate:
            mjd_tt = sat + tt / SECS_PER_DAY
            units = parse_timescale(params)
            scale = "TCB" if is_tempo2_si_units(units) else "TDB"
            einstein = np.asarray(compute_einstein_rate(mjd_tt, units=scale), dtype=np.float64)
        else:
            einstein = np.ones_like(sat, dtype=np.float64)
    else:
        einstein = np.asarray(einstein_rate, dtype=np.float64)

    clock_sec = tt + tt_tb
    model_clock = sat + clock_sec / SECS_PER_DAY

    bat = sat + (clock_sec - prebinary) / SECS_PER_DAY
    shk = compute_shklovskii_sec(bat, params)
    bbat = bat - shk / SECS_PER_DAY

    return Tempo2ClockTerms(
        sat_mjd=sat,
        correction_tt_sec=tt,
        correction_tt_teph_sec=tt_teph,
        correction_tt_tb_sec=tt_tb,
        einstein_rate=einstein,
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
    tt_tb, tt_teph = compute_correction_tt_tb_sec(
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
        correction_tt_teph_sec=tt_teph,
    )
