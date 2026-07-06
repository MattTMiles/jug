"""DEV-ONLY NumPy reference for tempo2-native chain.

.. deprecated::
    Temporary development aid. Production tempo2 mode is JAX-only.
    Remove this module once native-chain gates are green.

Set ``JUG_DEV_NUMPY_TEMPO2_CHAIN=1`` to enable the reference implementation.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

from jug.residuals.tempo2_clock import compute_correction_tt_tb_sec, compute_shklovskii_sec
from jug.residuals.tempo2_native.calculate_bclt_jax import compute_bclt_terms_numpy
from jug.residuals.tempo2_native.probes import (
    compute_formbats_effective_shapiro_sec,
    formbats_correction_tt_sec,
)
from jug.utils.constants import SECS_PER_DAY

_NUMPY_NATIVE_CHAIN_DEPRECATION = (
    "JUG NumPy tempo2-native reference path is deprecated and dev-only. "
    "Production tempo2 mode is JAX-only; use compute_tempo2_native_terms_jax."
)


def _require_numpy_native_chain_env() -> None:
    if os.environ.get("JUG_DEV_NUMPY_TEMPO2_CHAIN", "").lower() not in ("1", "true", "yes"):
        raise RuntimeError(
            "NumPy tempo2-native reference is dev-only. "
            "Set JUG_DEV_NUMPY_TEMPO2_CHAIN=1 to run chain_numpy tests."
        )


def _formbats_numpy(
    sat_mjd,
    correction_tt_sec,
    correction_tt_tb_sec,
    tropospheric_sec,
    roemer_sec,
    shapiro_delay_sec,
    tdis1_sec,
    tdis2_sec,
    params,
    *,
    shklovskii_sec=None,
):
    """NumPy formBats assembly mirroring formbats_jax."""
    correction_sec = correction_tt_sec + (
        correction_tt_tb_sec
        - tropospheric_sec
        + roemer_sec
        - shapiro_delay_sec
        - tdis1_sec
        - tdis2_sec
    )
    bat_corr_day = correction_sec / SECS_PER_DAY
    bat_mjd = sat_mjd + correction_sec / SECS_PER_DAY
    if shklovskii_sec is None:
        shk = compute_shklovskii_sec(bat_mjd, params)
    else:
        shk = np.asarray(shklovskii_sec, dtype=np.float64)
    bbat_mjd = bat_mjd - shk / SECS_PER_DAY
    return bat_corr_day, bat_mjd, bbat_mjd, shk


def _compute_tempo2_native_terms_numpy_impl(
    *,
    sat_mjd,
    correction_tt_sec,
    params,
    ssb_obs_pos_km,
    observatory_earth_km,
    earth_ssb_vel_km_s,
    ephem_path,
    tdis1_sec,
    tdis2_sec,
    tropospheric_sec,
    dt_emission_sec,
    use_native_ecliptic,
    planet_shapiro_enabled,
    utc_to_tdb_sec=None,
    formbats_tt_sec=None,
    ssb_obs_ls_fixed=None,
    obs_sun_ls_fixed=None,
    obs_planets_ls_fixed=None,
    freq_mhz_topocentric=None,
    ne_sw=0.0,
    use_model_epoch_batcorr=False,
    model_mjd=None,
    prebinary_override_sec=None,
):
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    utc = None if utc_to_tdb_sec is None else np.asarray(utc_to_tdb_sec, dtype=np.float64)
    if formbats_tt_sec is not None:
        formbats_tt = np.asarray(formbats_tt_sec, dtype=np.float64)
    else:
        formbats_tt = formbats_correction_tt_sec(tt, utc_to_tdb_sec=utc)
    if ssb_obs_ls_fixed is None:
        from jug.utils.constants import C_KM_S

        ssb_obs_ls = np.asarray(ssb_obs_pos_km, dtype=np.float64) / C_KM_S
    else:
        ssb_obs_ls = np.asarray(ssb_obs_ls_fixed, dtype=np.float64)
    if obs_sun_ls_fixed is None:
        raise ValueError("NumPy native reference requires obs_sun_ls_fixed")
    if ne_sw == 0.0:
        profile = resolve_engine_profile(params, "tempo2")
        ne_sw = resolve_ne_sw_cm3(params, profile)
    mjd_tt = sat + formbats_tt / SECS_PER_DAY
    tt_tb = compute_correction_tt_tb_sec(
        mjd_tt,
        observatory_earth_km=observatory_earth_km,
        earth_ssb_vel_km_s=earth_ssb_vel_km_s,
        params=params,
    )
    if ephem_path is None:
        from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path

        ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    from jug.delays.barycentric import compute_einstein_rate
    from jug.utils.timescales import parse_timescale

    dilate_freq = str(params.get("DILATEFREQ", "N")).upper() in ("Y", "YES", "TRUE", "1")
    einstein = np.ones_like(sat, dtype=np.float64)
    if dilate_freq:
        units = parse_timescale(params)
        scale = "TCB" if units == "SI_UNITS" else "TDB"
        einstein = np.asarray(compute_einstein_rate(mjd_tt, units=scale), dtype=np.float64)
    bclt = compute_bclt_terms_numpy(
        sat_mjd=sat,
        correction_tt_sec=formbats_tt,
        correction_tt_tb_sec=tt_tb,
        observatory_earth_km=observatory_earth_km,
        params=params,
        use_native_ecliptic=use_native_ecliptic,
        planet_shapiro_enabled=planet_shapiro_enabled,
        ssb_obs_ls_fixed=ssb_obs_ls,
        obs_sun_ls_fixed=np.asarray(obs_sun_ls_fixed, dtype=np.float64),
        obs_planets_ls_fixed=obs_planets_ls_fixed,
        freq_mhz=np.asarray(freq_mhz_topocentric, dtype=np.float64),
        earth_ssb_vel_km_s=earth_ssb_vel_km_s,
        ne_sw=float(ne_sw),
        einstein_rate=einstein,
    )
    tdis1_sec = bclt.tdis1_sec
    tdis2_sec = bclt.tdis2_sec
    prebinary = (
        -bclt.roemer_sec
        + tdis1_sec
        + tdis2_sec
        + bclt.shapiro_sun_sec
        + bclt.shapiro_planets_sec
        + tropospheric_sec
    )
    planet_shapiro = 1.0 if planet_shapiro_enabled else 0.0
    shap_delay = compute_formbats_effective_shapiro_sec(
        bclt.shapiro_sun_sec,
        bclt.shapiro_planets_sec,
        planet_shapiro=planet_shapiro,
    )
    if use_model_epoch_batcorr and model_mjd is not None:
        model = np.asarray(model_mjd, dtype=np.float64)
        if prebinary_override_sec is not None:
            prebin_for_bat = np.asarray(prebinary_override_sec, dtype=np.float64)
        else:
            prebin_for_bat = prebinary
        bat_corr_day = (model - sat) - prebin_for_bat / SECS_PER_DAY
        bat_mjd = sat + bat_corr_day
        shk = compute_shklovskii_sec(bat_mjd, params)
        bbat_mjd = bat_mjd - shk / SECS_PER_DAY
    else:
        bat_corr_day, bat_mjd, bbat_mjd, shk = _formbats_numpy(
            sat,
            formbats_tt,
            tt_tb,
            tropospheric_sec,
            bclt.roemer_sec,
            shap_delay,
            tdis1_sec,
            tdis2_sec,
            params,
        )
    pepoch = float(params["PEPOCH"])
    torb = np.asarray(dt_emission_sec, dtype=np.float64) - (bbat_mjd - pepoch) * SECS_PER_DAY
    return {
        "sat_mjd": sat,
        "correction_tt_sec": formbats_tt,
        "correction_tt_tb_sec": tt_tb,
        "roemer_sec": bclt.roemer_sec,
        "tdis1_sec": tdis1_sec,
        "tdis2_sec": tdis2_sec,
        "shapiro_sun_sec": bclt.shapiro_sun_sec,
        "shapiro_planets_sec": bclt.shapiro_planets_sec,
        "shapiro_delay_sec": shap_delay,
        "tropospheric_sec": tropospheric_sec,
        "prebinary_sec": prebinary,
        "bat_corr_day": bat_corr_day,
        "bat_corr_day_residual": np.zeros_like(bat_corr_day),
        "bat_mjd": bat_mjd,
        "bbat_mjd": bbat_mjd,
        "shklovskii_sec": shk,
        "torb_sec": torb,
        "dt_emission_sec": np.asarray(dt_emission_sec, dtype=np.float64),
        "dt_ssb_sec": bclt.dt_ssb_sec,
        "bclt_iterations": bclt.bclt_iterations,
        "converged": bclt.converged,
    }


def compute_tempo2_native_terms_numpy(*args, **kwargs):
    _require_numpy_native_chain_env()
    warnings.warn(_NUMPY_NATIVE_CHAIN_DEPRECATION, DeprecationWarning, stacklevel=2)
    return _compute_tempo2_native_terms_numpy_impl(*args, **kwargs)
