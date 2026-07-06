"""Production JAX orchestrator for tempo2-native clock/delay/spin chain."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path
from jug.delays.tempo2_geometry import (
    build_tempo2_pulsar_vectors,
    pmrv_rad_per_century,
)
from jug.residuals.tempo2_native.calculate_bclt_jax import (
    _dm_vals_numpy as bclt_dm_vals_numpy,
    compute_bclt_terms_jax,
)
from jug.residuals.tempo2_native.clock_jax import compute_tempo2_correction_tt_tb_jax
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
from jug.utils.timescales import parse_timescale


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
    site_vel_km_s=None,
    tdb_mjd=None,
) -> Tempo2NativeTerms:
    """Compute tempo2-native BCLT, formBats, bbat, torb entirely in JAX."""
    del pulse_numbers, pn_add, jump_phase, tzr_phase, addsat_sec, toas, tdis1_sec, tdis2_sec
    del utc_to_tdb_sec, prebinary_override_sec, model_mjd, use_model_epoch_batcorr

    if use_native_ecliptic is None:
        use_native_ecliptic = bool(params.get("_ecliptic_coords", False))

    sat_j = jnp.asarray(sat_mjd, dtype=jnp.float64)
    tt_j = jnp.asarray(
        formbats_correction_tt_sec(
            np.asarray(correction_tt_sec, dtype=np.float64),
            formbats_tt_sec=(
                None
                if formbats_tt_sec is None
                else np.asarray(formbats_tt_sec, dtype=np.float64)
            ),
        ),
        dtype=jnp.float64,
    )
    tropo_j = jnp.asarray(tropospheric_sec, dtype=jnp.float64)
    dt_emit_j = jnp.asarray(dt_emission_sec, dtype=jnp.float64)
    obs_earth_j = jnp.asarray(observatory_earth_km, dtype=jnp.float64)
    earth_ssb_j = jnp.asarray(earth_ssb_km, dtype=jnp.float64)
    if ssb_obs_ls_fixed is None or obs_sun_ls_fixed is None:
        raise ValueError("tempo2-native BCLT requires fixed IFTE geometry arrays")
    ssb_obs_ls = jnp.asarray(ssb_obs_ls_fixed, dtype=jnp.float64)
    obs_sun_ls = jnp.asarray(obs_sun_ls_fixed, dtype=jnp.float64)
    site_vel_j = (
        jnp.zeros((ssb_obs_ls.shape[0], 3), dtype=jnp.float64)
        if site_vel_km_s is None
        else jnp.asarray(site_vel_km_s, dtype=jnp.float64)
    )
    if earth_ssb_j.ndim == 2 and earth_ssb_j.shape[1] == 3:
        earth_vel_j = jnp.asarray(earth_ssb_vel_km_s, dtype=jnp.float64)
    else:
        earth_vel_j = jnp.asarray(earth_ssb_vel_km_s, dtype=jnp.float64)

    if freq_mhz_topocentric is not None:
        freq_j = jnp.asarray(freq_mhz_topocentric, dtype=jnp.float64)
    else:
        freq_j = jnp.asarray(freq_mhz, dtype=jnp.float64)

    units = parse_timescale(params)
    mjd_tt = sat_j + tt_j / 86400.0
    tt_tb_j = compute_tempo2_correction_tt_tb_jax(
        mjd_tt,
        obs_earth_j,
        earth_vel_j,
        units_tdb=units == "TDB",
        si_units=units == "SI_UNITS",
    )

    pos, vel, acc = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=use_native_ecliptic
    )
    dm_vals = bclt_dm_vals_numpy(np.asarray(sat_mjd, dtype=np.float64), params)
    dilate_freq = str(params.get("DILATEFREQ", "N")).upper() in ("Y", "YES", "TRUE", "1")
    jup = None
    if obs_planets_ls_fixed and "jupiter" in obs_planets_ls_fixed:
        jup = jnp.asarray(obs_planets_ls_fixed["jupiter"], dtype=jnp.float64)

    bclt = compute_bclt_terms_jax(
        sat_mjd=sat_j,
        correction_tt_sec=tt_j,
        correction_tt_tb_sec=tt_tb_j,
        ssb_obs_ls=ssb_obs_ls,
        obs_sun_ls=obs_sun_ls,
        freq_mhz=freq_j,
        earth_ssb_vel_km_s=earth_vel_j,
        site_vel_km_s=site_vel_j,
        dm_vals=jnp.asarray(dm_vals, dtype=jnp.float64),
        pos_pulsar=jnp.asarray(pos, dtype=jnp.float64),
        vel_pulsar=jnp.asarray(vel, dtype=jnp.float64),
        acc_pulsar=jnp.asarray(acc, dtype=jnp.float64),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        parallax_mas=float(params.get("PX", 0.0)),
        pmrv_rad_century=pmrv_rad_per_century(float(params.get("PMRV", 0.0))),
        ne_sw=float(ne_sw),
        dilate_freq=dilate_freq,
        planet_shapiro_enabled=planet_shapiro_enabled,
        obs_jupiter_ls=jup,
    )

    planet_shapiro = 1.0 if planet_shapiro_enabled else 0.0
    shap_delay_j = (
        bclt.shapiro_sun_sec + planet_shapiro * bclt.shapiro_planets_sec
    )

    bat_corr_day, bat_corr_resid, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_j,
        tt_j,
        tt_tb_j,
        tropo_j,
        bclt.roemer_sec,
        shap_delay_j,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        jnp.zeros_like(sat_j),
    )
    shk_j = compute_shklovskii_sec_jax(bat_mjd, params)
    _, _, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_j,
        tt_j,
        tt_tb_j,
        tropo_j,
        bclt.roemer_sec,
        shap_delay_j,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        shk_j,
    )

    pepoch = float(params["PEPOCH"])
    torb_j = compute_torb_closure_jax(bbat_mjd, dt_emit_j, jnp.asarray(pepoch, dtype=jnp.float64))

    return Tempo2NativeTerms(
        sat_mjd=sat_j,
        correction_tt_sec=tt_j,
        correction_tt_tb_sec=tt_tb_j,
        roemer_sec=bclt.roemer_sec,
        tdis1_sec=bclt.tdis1_sec,
        tdis2_sec=bclt.tdis2_sec,
        shapiro_sun_sec=bclt.shapiro_sun_sec,
        shapiro_planets_sec=bclt.shapiro_planets_sec,
        shapiro_delay_sec=shap_delay_j,
        tropospheric_sec=tropo_j,
        prebinary_sec=jnp.zeros_like(sat_j),
        bat_corr_day=bat_corr_day,
        bat_corr_day_residual=bat_corr_resid,
        bat_mjd=bat_mjd,
        bbat_mjd=bbat_mjd,
        shklovskii_sec=shk_j,
        torb_sec=torb_j,
        dt_emission_sec=dt_emit_j,
        dt_ssb_sec=bclt.dt_ssb_sec,
        bclt_iterations=bclt.bclt_iterations,
        converged=bclt.converged,
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
    observatory_earth_km: np.ndarray | None = None,
    earth_ssb_km: np.ndarray | None = None,
    earth_ssb_vel_km_s: np.ndarray | None = None,
    ephem_path: str | None = None,
    use_model_epoch_batcorr: bool = False,
) -> Tempo2NativeTerms:
    """Build native terms from ``compute_residuals_simple`` geometry exports."""
    from jug.delays.tempo2_geometry import Tempo2ObservatoryState, tempo2_observatory_chain_vectors

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
    formbats_tt = td.get("formbats_correction_tt_sec", td.get("correction_tt_sec"))
    obs_state = td.get("tempo2_obs_state")
    site_vel = None
    ssb_obs_ls = jug_result.get("ssb_obs_pos_ls")
    obs_sun_ls = jug_result.get("obs_sun_pos_ls")
    if obs_state is not None:
        site_vel = obs_state.get("site_vel_km_s")
        if observatory_earth_km is None:
            observatory_earth_km = np.asarray(
                obs_state["observatory_earth_km"], dtype=np.float64
            )[:, :3]
        earth_ssb_arr = np.asarray(obs_state["earth_ssb_km"], dtype=np.float64)
        if earth_ssb_km is None:
            earth_ssb_km = earth_ssb_arr[:, :3]
        if earth_ssb_vel_km_s is None:
            earth_ssb_vel_km_s = earth_ssb_arr[:, 3:6]
        sun_ssb = obs_state.get("sun_ssb_km")
        planet_ssb = obs_state.get("planet_ssb_km", {})
        if sun_ssb is None:
            sun_ssb = np.zeros((len(toas), 6), dtype=np.float64)
        else:
            sun_ssb = np.asarray(sun_ssb, dtype=np.float64)
        if isinstance(planet_ssb, dict):
            planet_ssb = {
                name: np.asarray(arr, dtype=np.float64) for name, arr in planet_ssb.items()
            }
        else:
            planet_ssb = {}
        state = Tempo2ObservatoryState(
            earth_ssb_km=earth_ssb_arr,
            observatory_earth_km=np.asarray(
                obs_state["observatory_earth_km"], dtype=np.float64
            ),
            sun_ssb_km=sun_ssb,
            planet_ssb_km=planet_ssb,
            site_vel_km_s=np.asarray(site_vel, dtype=np.float64),
        )
        _, ssb_obs_ls, obs_sun_ls, planets_from_state = tempo2_observatory_chain_vectors(
            state
        )
        if planets is None:
            planets = planets_from_state
    if observatory_earth_km is None or earth_ssb_km is None or earth_ssb_vel_km_s is None:
        raise ValueError(
            "prepare_native_chain_from_simple_result requires tempo2_obs_state "
            "or explicit observatory_earth_km / earth_ssb_km / earth_ssb_vel_km_s"
        )
    return compute_tempo2_native_terms_jax(
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        correction_tt_sec=jnp.asarray(formbats_tt, dtype=jnp.float64),
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
        tropospheric_sec=jnp.asarray(td["tropo_delay_sec"], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(
            np.asarray(jug_result["dt_sec"], dtype=np.float64), dtype=jnp.float64
        ),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        utc_to_tdb_sec=jnp.asarray(td.get("utc_to_tdb_sec", 0.0), dtype=jnp.float64),
        formbats_tt_sec=jnp.asarray(formbats_tt, dtype=jnp.float64),
        ssb_obs_ls_fixed=jnp.asarray(ssb_obs_ls, dtype=jnp.float64),
        obs_sun_ls_fixed=jnp.asarray(obs_sun_ls, dtype=jnp.float64),
        obs_planets_ls_fixed=planets,
        freq_mhz_topocentric=jnp.asarray(freq_topo, dtype=jnp.float64),
        ne_sw=ne_sw,
        use_model_epoch_batcorr=use_model_epoch_batcorr,
        model_mjd=jnp.asarray(jug_result["model_mjd"], dtype=jnp.float64),
        prebinary_override_sec=jnp.asarray(prebinary_jug, dtype=jnp.float64),
        site_vel_km_s=None if site_vel is None else jnp.asarray(site_vel, dtype=jnp.float64),
    )


def compute_native_tempo2_residual_sec(
    params: dict,
    *,
    static: dict,
    weights,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    track_val: int = -2,
    pulse_numbers=None,
    pn_add=None,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the JAX chain for one param dict."""
    td = static["term_diagnostics"]
    jug_result = {
        "term_diagnostics": td,
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "model_mjd": td.get("sat_mjd", static.get("model_mjd")),
        "ssb_obs_pos_ls": static.get("ssb_obs_pos_ls"),
        "obs_sun_pos_ls": static.get("obs_sun_pos_ls"),
        "obs_planet_pos_ls": static.get("obs_planet_pos_ls"),
        "compatibility": "tempo2",
    }
    toas = static.get("toas") or []
    if not toas:
        n = len(np.asarray(td["sat_mjd"]))
        from jug.io.tim_reader import SimpleTOA

        toas = [
            SimpleTOA(
                mjd_int=int(np.floor(float(td["sat_mjd"][i]))),
                mjd_frac=float(td["sat_mjd"][i]) - int(np.floor(float(td["sat_mjd"][i]))),
                mjd_str=str(td["sat_mjd"][i]),
                freq_mhz=float(static["freq_bary_mhz"][i]),
                error_us=1.0,
                observatory="wsrt",
                flags={},
            )
            for i in range(n)
        ]
    native = prepare_native_chain_from_simple_result(jug_result, params, toas)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    residual_sec, _, _ = compute_tempo2_native_residuals_jax(
        native_terms=native,
        params=params,
        weights=jnp.asarray(weights, dtype=jnp.float64),
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
        subtract_mean=subtract_mean,
        track_val=track_val,
    )
    return residual_sec
