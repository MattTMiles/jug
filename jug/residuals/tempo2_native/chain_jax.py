"""Production JAX orchestrator for tempo2-native clock/delay/spin chain."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.barycentric import compute_einstein_rate
from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path
from jug.residuals.tempo2_clock import compute_correction_tt_tb_sec
from jug.utils.timescales import parse_timescale
from jug.residuals.tempo2_native.calculate_bclt_jax import compute_bclt_terms_numpy
from jug.residuals.tempo2_native.formbats_jax import (
    compute_formbats_jax,
    compute_shklovskii_sec_jax,
    compute_torb_closure_jax,
)
from jug.residuals.tempo2_native.probes import (
    compute_formbats_effective_shapiro_sec,
    formbats_correction_tt_sec,
)
from jug.residuals.tempo2_native.spin_jax import (
    compute_tempo2_phase5_jax,
    spin_params_to_jax,
    track_minus2_frac_phase_jax,
)
from jug.residuals.tempo2_native.types import Tempo2NativeTerms


def _build_pn_add(toas: list[Any]) -> np.ndarray:
    pn_add = np.full(len(toas), -1, dtype=np.int64)
    running = np.int64(-1)
    for i, toa in enumerate(toas):
        pn_add[i] = running
        pnadd_val = toa.flags.get("pnadd")
        if pnadd_val is not None:
            running += np.int64(int(pnadd_val))
    return pn_add


def compute_tempo2_native_terms_jax(
    *,
    sat_mjd,
    correction_tt_sec,
    correction_tt_tb_sec,
    params,
    toas,
    observatory_earth_km,
    earth_ssb_km,
    earth_ssb_vel_km_s,
    ephem_path,
    freq_mhz,
    tdis1_sec,
    tdis2_sec,
    tropospheric_sec,
    dt_emission_sec,
    use_native_ecliptic: bool | None = None,
    utc_to_tdb_sec=None,
    formbats_tt_sec=None,
    ssb_obs_ls_fixed=None,
    obs_sun_ls_fixed=None,
    obs_planets_ls_fixed=None,
    freq_mhz_topocentric=None,
    ne_sw: float = 0.0,
    use_model_epoch_batcorr: bool = False,
    model_mjd=None,
    prebinary_override_sec=None,
    planet_shapiro_enabled: bool = True,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    addsat_sec=None,
) -> Tempo2NativeTerms:
    """Compute tempo2-native BCLT, formBats, bbat, torb in JAX.

    BCLT iteration runs on host with fixed IFTE geometry; formBats/spin export as JAX.
    """
    del pulse_numbers, pn_add, jump_phase, tzr_phase, addsat_sec, toas, ephem_path, earth_ssb_km
    del tdis1_sec, tdis2_sec

    if use_native_ecliptic is None:
        use_native_ecliptic = bool(params.get("_ecliptic_coords", False))

    sat_np = np.asarray(jax.device_get(sat_mjd), dtype=np.float64)
    tt_np = np.asarray(jax.device_get(correction_tt_sec), dtype=np.float64)
    tt_tb_np = np.asarray(jax.device_get(correction_tt_tb_sec), dtype=np.float64)
    tropo_np = np.asarray(jax.device_get(tropospheric_sec), dtype=np.float64)
    dt_emit_np = np.asarray(jax.device_get(dt_emission_sec), dtype=np.float64)
    obs_earth_np = np.asarray(jax.device_get(observatory_earth_km), dtype=np.float64)
    vel_np = np.asarray(jax.device_get(earth_ssb_vel_km_s), dtype=np.float64)
    utc_to_tdb_np = None
    if utc_to_tdb_sec is not None:
        utc_to_tdb_np = np.asarray(jax.device_get(utc_to_tdb_sec), dtype=np.float64)
    formbats_tt_np = None
    if formbats_tt_sec is not None:
        formbats_tt_np = np.asarray(jax.device_get(formbats_tt_sec), dtype=np.float64)

    formbats_tt_for_chain = formbats_correction_tt_sec(
        tt_np,
        utc_to_tdb_sec=utc_to_tdb_np,
        formbats_tt_sec=formbats_tt_np,
    )
    mjd_tt = sat_np + formbats_tt_for_chain / 86400.0
    tt_tb_np = compute_correction_tt_tb_sec(
        mjd_tt,
        observatory_earth_km=obs_earth_np,
        earth_ssb_vel_km_s=vel_np,
        params=params,
    )

    if ssb_obs_ls_fixed is None or obs_sun_ls_fixed is None:
        raise ValueError("tempo2-native BCLT requires fixed IFTE geometry arrays")
    ssb_obs_ls = np.asarray(jax.device_get(ssb_obs_ls_fixed), dtype=np.float64)
    obs_sun_ls = np.asarray(jax.device_get(obs_sun_ls_fixed), dtype=np.float64)
    planets_fixed = None
    if obs_planets_ls_fixed is not None:
        planets_fixed = {
            name: np.asarray(jax.device_get(arr), dtype=np.float64)
            for name, arr in obs_planets_ls_fixed.items()
        }
    if freq_mhz_topocentric is not None:
        freq_topo = np.asarray(jax.device_get(freq_mhz_topocentric), dtype=np.float64)
    else:
        freq_topo = np.asarray(jax.device_get(freq_mhz), dtype=np.float64)

    dilate_freq = str(params.get("DILATEFREQ", "N")).upper() in ("Y", "YES", "TRUE", "1")
    einstein_np = np.ones_like(sat_np, dtype=np.float64)
    if dilate_freq:
        units = parse_timescale(params)
        scale = "TCB" if units == "SI_UNITS" else "TDB"
        einstein_np = np.asarray(
            compute_einstein_rate(mjd_tt, units=scale), dtype=np.float64
        )

    bclt = compute_bclt_terms_numpy(
        sat_mjd=sat_np,
        correction_tt_sec=formbats_tt_for_chain,
        correction_tt_tb_sec=tt_tb_np,
        observatory_earth_km=obs_earth_np,
        params=params,
        use_native_ecliptic=use_native_ecliptic,
        planet_shapiro_enabled=planet_shapiro_enabled,
        ssb_obs_ls_fixed=ssb_obs_ls,
        obs_sun_ls_fixed=obs_sun_ls,
        obs_planets_ls_fixed=planets_fixed,
        freq_mhz=freq_topo,
        earth_ssb_vel_km_s=vel_np,
        ne_sw=float(ne_sw),
        einstein_rate=einstein_np,
    )

    tdis1_np = bclt.tdis1_sec
    tdis2_np = bclt.tdis2_sec
    prebinary_np = (
        -bclt.roemer_sec + tdis1_np + tdis2_np
        + bclt.shapiro_sun_sec + bclt.shapiro_planets_sec + tropo_np
    )

    planet_shapiro = 1.0 if planet_shapiro_enabled else 0.0
    formbats_tt_j = jnp.asarray(formbats_tt_for_chain, dtype=jnp.float64)
    shap_delay_np = compute_formbats_effective_shapiro_sec(
        bclt.shapiro_sun_sec,
        bclt.shapiro_planets_sec,
        planet_shapiro=planet_shapiro,
    )

    sat_j = jnp.asarray(sat_np, dtype=jnp.float64)
    tt_j = formbats_tt_j
    tt_tb_j = jnp.asarray(tt_tb_np, dtype=jnp.float64)
    roemer_j = jnp.asarray(bclt.roemer_sec, dtype=jnp.float64)
    tdis1_j = jnp.asarray(tdis1_np, dtype=jnp.float64)
    tdis2_j = jnp.asarray(tdis2_np, dtype=np.float64)
    shap_delay_j = jnp.asarray(shap_delay_np, dtype=jnp.float64)
    tropo_j = jnp.asarray(tropo_np, dtype=jnp.float64)

    if use_model_epoch_batcorr and model_mjd is not None:
        model_np = np.asarray(jax.device_get(model_mjd), dtype=np.float64)
        if prebinary_override_sec is not None:
            prebin_for_bat = np.asarray(jax.device_get(prebinary_override_sec), dtype=np.float64)
        else:
            prebin_for_bat = prebinary_np
        bat_corr_day_np = (model_np - sat_np) - prebin_for_bat / 86400.0
        bat_corr_day = jnp.asarray(bat_corr_day_np, dtype=jnp.float64)
        bat_corr_resid = jnp.zeros_like(bat_corr_day)
        bat_mjd = sat_j + bat_corr_day
        shk_j = compute_shklovskii_sec_jax(bat_mjd, params)
        bbat_mjd = bat_mjd - shk_j / jnp.asarray(86400.0, dtype=jnp.float64)
    else:
        bat_corr_day, bat_corr_resid, bat_mjd, bbat_mjd = compute_formbats_jax(
            sat_j,
            tt_j,
            tt_tb_j,
            tropo_j,
            roemer_j,
            shap_delay_j,
            tdis1_j,
            tdis2_j,
            jnp.zeros_like(sat_j),
        )
        shk_j = compute_shklovskii_sec_jax(bat_mjd, params)
        _, _, bat_mjd, bbat_mjd = compute_formbats_jax(
            sat_j,
            tt_j,
            tt_tb_j,
            tropo_j,
            roemer_j,
            shap_delay_j,
            tdis1_j,
            tdis2_j,
            shk_j,
        )

    pepoch = float(params["PEPOCH"])
    dt_emit_j = jnp.asarray(dt_emit_np, dtype=jnp.float64)
    torb_j = compute_torb_closure_jax(
        bbat_mjd, dt_emit_j, jnp.asarray(pepoch, dtype=jnp.float64)
    )

    return Tempo2NativeTerms(
        sat_mjd=sat_j,
        correction_tt_sec=formbats_tt_j,
        correction_tt_tb_sec=tt_tb_j,
        roemer_sec=roemer_j,
        tdis1_sec=tdis1_j,
        tdis2_sec=tdis2_j,
        shapiro_sun_sec=jnp.asarray(bclt.shapiro_sun_sec, dtype=jnp.float64),
        shapiro_planets_sec=jnp.asarray(bclt.shapiro_planets_sec, dtype=jnp.float64),
        shapiro_delay_sec=shap_delay_j,
        tropospheric_sec=tropo_j,
        prebinary_sec=jnp.asarray(prebinary_np, dtype=jnp.float64),
        bat_corr_day=bat_corr_day,
        bat_corr_day_residual=bat_corr_resid,
        bat_mjd=bat_mjd,
        bbat_mjd=bbat_mjd,
        shklovskii_sec=shk_j,
        torb_sec=torb_j,
        dt_emission_sec=dt_emit_j,
        dt_ssb_sec=jnp.asarray(bclt.dt_ssb_sec, dtype=jnp.float64),
        bclt_iterations=jnp.asarray(bclt.bclt_iterations, dtype=jnp.int32),
        converged=jnp.asarray(bclt.converged),
    )


def compute_tempo2_native_residuals_jax(
    *,
    native_terms: Tempo2NativeTerms,
    params,
    weights,
    pulse_numbers,
    pn_add,
    jump_phase,
    tzr_phase,
    subtract_mean: bool,
    mean_mode: str = "unweighted",
    track_val: int = -2,
):
    """Return residual seconds, pulse numbers, and native terms for tempo2 mode."""
    del weights, mean_mode
    f_terms, pepoch = spin_params_to_jax(params)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    phase5 = compute_tempo2_phase5_jax(
        native_terms.bbat_mjd,
        native_terms.torb_sec,
        f_terms,
        pepoch,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
    )
    if int(track_val) == -2 and pulse_numbers is not None and pn_add is not None:
        frac, pulse = track_minus2_frac_phase_jax(
            phase5,
            native_terms.bbat_mjd,
            f_terms[0],
            jnp.asarray(pulse_numbers, dtype=jnp.int64),
            jnp.asarray(pn_add, dtype=jnp.int64),
        )
    else:
        pulse = jnp.zeros_like(phase5)
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / f_terms[0]
    if subtract_mean:
        residual_sec = residual_sec - jnp.mean(residual_sec)
    return residual_sec, pulse, native_terms


def prepare_native_chain_from_simple_result(
    jug_result: dict,
    params: dict,
    toas: list[Any],
    *,
    observatory_earth_km: np.ndarray,
    earth_ssb_km: np.ndarray,
    earth_ssb_vel_km_s: np.ndarray,
    ephem_path: str | None = None,
    use_model_epoch_batcorr: bool = False,
) -> Tempo2NativeTerms:
    """Build native terms from ``compute_residuals_simple`` geometry exports."""
    td = jug_result["term_diagnostics"]
    tdis1 = np.asarray(td["dm_delay_sec"], dtype=np.float64) + np.asarray(
        td.get("dmx_delay_sec", 0.0), dtype=np.float64
    )
    prebinary_jug = np.asarray(td["prebinary_delay_sec"], dtype=np.float64)
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    ne_sw = resolve_ne_sw_cm3(params, profile)
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    planets = jug_result.get("obs_planet_pos_ls")
    formbats_tt = td.get("formbats_correction_tt_sec")
    return compute_tempo2_native_terms_jax(
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        correction_tt_sec=jnp.asarray(td["correction_tt_sec"], dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        params=params,
        toas=toas,
        observatory_earth_km=jnp.asarray(observatory_earth_km, dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(earth_ssb_km, dtype=jnp.float64),
        earth_ssb_vel_km_s=jnp.asarray(earth_ssb_vel_km_s, dtype=jnp.float64),
        ephem_path=ephem_path or resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405")),
        freq_mhz=jnp.asarray(jug_result.get("freq_bary_mhz", td.get("freq_bary_mhz", [])), dtype=jnp.float64),
        tdis1_sec=jnp.asarray(tdis1, dtype=jnp.float64),
        tdis2_sec=jnp.asarray(td["sw_delay_sec"], dtype=np.float64),
        tropospheric_sec=jnp.asarray(td["tropo_delay_sec"], dtype=np.float64),
        dt_emission_sec=jnp.asarray(jug_result["dt_sec"], dtype=np.float64),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        utc_to_tdb_sec=jnp.asarray(td.get("utc_to_tdb_sec", 0.0), dtype=jnp.float64),
        formbats_tt_sec=(
            jnp.asarray(formbats_tt, dtype=jnp.float64) if formbats_tt is not None else None
        ),
        ssb_obs_ls_fixed=jnp.asarray(jug_result["ssb_obs_pos_ls"], dtype=jnp.float64),
        obs_sun_ls_fixed=jnp.asarray(jug_result["obs_sun_pos_ls"], dtype=jnp.float64),
        obs_planets_ls_fixed=planets,
        freq_mhz_topocentric=jnp.asarray(freq_topo, dtype=jnp.float64),
        ne_sw=ne_sw,
        use_model_epoch_batcorr=use_model_epoch_batcorr,
        model_mjd=jnp.asarray(jug_result["model_mjd"], dtype=jnp.float64),
        prebinary_override_sec=jnp.asarray(prebinary_jug, dtype=jnp.float64),
    )
