"""Tempo2-compatible host pipeline (setup + hybrid residual finalization).

Production host routing contract (strict-parity baseline, 2026-07-07)
----------------------------------------------------------------------
Host residuals in ``compute_residuals_simple`` use **hybrid** routing — not a
single native-eval path. See ``TEMPO2_PARITY.md`` § "Production fix".

| Condition | Route | Notes |
|-----------|-------|-------|
| ``TRACK == -2`` | ``compute_phase_residuals`` (Taylor + legacy TRACK −2) | wsrt167 ~15.5 ns RMS |
| ``TRACK`` absent (tempo2 default 0) | ``compute_phase_residuals`` (Taylor sequential) | no-TRACK nrt1400 ~4.4 ns |
| explicit non-(-2) ``TRACK`` | ``compute_native_eval_residuals_jax`` | native delay chain staging |

Do **not** promote ``phase5@bbat`` or split-longdouble formBats into this host
path (strict-parity probes ruled them out).

Fit / autodiff / design-matrix path
------------------------------------
``jug.fitting.jax_residual_delta`` uses the tempo2-native JAX graph
(``phase5@bbat`` via ``JUG_TEMPO2_NATIVE_GRAPH_MODE``). That is an **intentionally
different model** from the host Taylor path above. Do not merge them without
explicit parity work and documented gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
from jug.residuals.tempo2_graph_config import USE_NATIVE_BBAT_PHASE5
from jug.residuals.tempo2_native.types import Tempo2NativeTerms
from jug.utils.constants import SECS_PER_DAY
from jug.utils.timescales import is_tempo2_si_units, parse_timescale


@dataclass
class Tempo2HostSetupResult:
    """Outputs from tempo2 host clock/delay setup before phase residuals."""

    formbats_correction_tt: np.ndarray
    tempo2_clock_terms: Any
    tempo2_obs_state_export: dict
    earth_ssb_vel_km_s: np.ndarray
    dm_delay_sec: np.ndarray
    sw_delay_sec: np.ndarray
    tropo_delay_sec: np.ndarray
    roemer_sec: np.ndarray
    sun_shapiro_sec: np.ndarray
    planet_shapiro_sec: np.ndarray
    roemer_shapiro: np.ndarray
    prebinary_delay_sec: np.ndarray
    ifte_delta_t_sec: np.ndarray
    bbat_mjd: np.ndarray | None
    torb_sec: np.ndarray | None
    model_mjd: np.ndarray


@dataclass
class Tempo2HostFinalizeResult:
    """Host tempo2 residual finalization outputs."""

    residuals_us: np.ndarray
    residuals_sec: np.ndarray
    pulse_number: np.ndarray
    native: Tempo2NativeTerms
    dm_delay_sec: np.ndarray
    sw_delay_sec: np.ndarray


def compute_tempo2_host_setup(
    *,
    mjd_utc: np.ndarray,
    obs_clocks: dict,
    bipm_clock: dict,
    toas: list,
    all_obs_codes: list,
    obs_clock: dict,
    time_offsets: dict,
    params: dict,
    obs_itrf_km: np.ndarray,
    dm_eff: np.ndarray,
    freq_bary_mhz: np.ndarray,
    dt_sec: np.ndarray,
    model_mjd: np.ndarray,
    PEPOCH: np.longdouble,
    compatibility_mode: str,
    engine_profile,
    correct_troposphere: bool,
    roemer_sec: np.ndarray,
    sun_shapiro_sec: np.ndarray,
    planet_shapiro_sec: np.ndarray,
    roemer_shapiro: np.ndarray,
    dm_delay_sec: np.ndarray,
    sw_delay_sec: np.ndarray,
    tropo_delay_sec: np.ndarray,
    dmx_delay_sec: np.ndarray,
    skip_native_bclt_overlay: bool,
) -> Tempo2HostSetupResult:
    """Tempo2 host clock chain, DM/SW loop, optional native BCLT overlay, formBats."""
    from jug.delays.barycentric import compute_einstein_rate
    from jug.delays.tempo2_ephemeris import (
        bootstrap_tempo2_observatory_state,
        resolve_tempo2_ephemeris_path,
    )
    from jug.delays.tempo2_geometry import (
        build_tempo2_pulsar_vectors,
        compute_tempo2_dm_delays_sec,
        psr_pos_at_delt,
        tempo2_dilate_freq_enabled,
        tempo2_observatory_chain_vectors,
    )
    from jug.residuals.tempo2_clock import (
        compute_formbats_arrival,
        compute_get_correction_tt_sec,
        compute_site_clock_corrections_sec,
    )
    from jug.residuals.tempo2_native.chain_jax import (
        prepare_native_chain_from_simple_result,
    )
    from jug.residuals.tempo2_native.types import native_terms_to_numpy
    from jug.utils.ifteph import ifte_delta_t_mjd

    correction_tt = compute_site_clock_corrections_sec(
        mjd_utc,
        obs_clocks=obs_clocks,
        bipm_clock=bipm_clock,
        toas=toas,
        all_obs_codes=all_obs_codes,
        obs_clock_default=obs_clock,
        time_offsets=time_offsets,
    )
    formbats_correction_tt = compute_get_correction_tt_sec(
        toas,
        obs_clocks=obs_clocks,
        obs_clock_default=obs_clock,
        bipm_clock=bipm_clock,
        all_obs_codes=all_obs_codes,
        time_offsets=time_offsets,
    )

    ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    sat_arr = np.asarray(mjd_utc, dtype=np.float64)
    formbats_tt_arr = np.asarray(formbats_correction_tt, dtype=np.float64)
    obs_itrf = np.asarray(obs_itrf_km, dtype=np.float64).reshape(3)

    geo_boot = bootstrap_tempo2_observatory_state(
        sat_arr,
        formbats_tt_arr,
        obs_itrf,
        ephem_path=ephem_path,
        params=params,
    )
    tempo2_obs_state = geo_boot.state
    mjd_tt = geo_boot.site_mjd
    tt_tb = geo_boot.correction_tt_tb_sec
    tt_teph = geo_boot.correction_tt_teph_sec
    tempo2_obs_state_export = {
        "site_vel_km_s": tempo2_obs_state.site_vel_km_s,
        "earth_ssb_km": tempo2_obs_state.earth_ssb_km,
        "observatory_earth_km": tempo2_obs_state.observatory_earth_km,
        "sun_ssb_km": tempo2_obs_state.sun_ssb_km,
        "planet_ssb_km": tempo2_obs_state.planet_ssb_km,
    }
    earth_ssb_vel_km_s = tempo2_obs_state.earth_ssb_km[:, 3:6]

    _, _, obs_sun_ls_dm, _ = tempo2_observatory_chain_vectors(tempo2_obs_state)
    pos_pulsar, vel_pulsar, _ = build_tempo2_pulsar_vectors(
        params,
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
    )
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    delt_formbats = (
        sat_arr - posepoch + (formbats_tt_arr + tt_tb) / SECS_PER_DAY
    ) / 36525.0
    topo_freq_mhz = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    dilate_freq = tempo2_dilate_freq_enabled(params)
    if dilate_freq:
        units = parse_timescale(params)
        ein_scale = "TCB" if is_tempo2_si_units(units) else "TDB"
        einstein_rate = np.asarray(
            compute_einstein_rate(mjd_tt, units=ein_scale), dtype=np.float64
        )
    else:
        einstein_rate = np.ones_like(sat_arr, dtype=np.float64)
    ne_sw_val = resolve_ne_sw_cm3(params, engine_profile)
    dm_host = np.zeros(len(toas), dtype=np.float64)
    sw_host = np.zeros(len(toas), dtype=np.float64)
    for i in range(len(toas)):
        psr_pos_i = psr_pos_at_delt(pos_pulsar, vel_pulsar, float(delt_formbats[i]))
        dm_host[i], sw_host[i] = compute_tempo2_dm_delays_sec(
            sat_mjd=float(sat_arr[i]),
            freq_mhz=float(topo_freq_mhz[i]),
            psr_pos=psr_pos_i,
            obs_to_sun_ls=obs_sun_ls_dm[i],
            earth_ssb_vel_km_s=earth_ssb_vel_km_s[i],
            dm_val=float(dm_eff[i]),
            ne_sw=float(ne_sw_val),
            einstein_rate=float(einstein_rate[i]),
            dilate_freq=dilate_freq,
            site_vel_km_s=tempo2_obs_state.site_vel_km_s[i],
        )
    dm_delay_sec = dm_host
    sw_delay_sec = sw_host

    if correct_troposphere:
        from jug.delays.tropo_jax import compute_tempo2_tropo_delay_host

        tropo_delay_sec = compute_tempo2_tropo_delay_host(
            sat_arr,
            formbats_tt_arr,
            obs_itrf_km=obs_itrf,
            pos_pulsar=pos_pulsar,
            mapping_clock_sec=correction_tt,
        )

    if not skip_native_bclt_overlay:
        _overlay_td = {
            "sat_mjd": sat_arr,
            "correction_tt_sec": formbats_tt_arr,
            "correction_tt_tb_sec": tt_tb,
            "formbats_correction_tt_sec": formbats_tt_arr,
            "tropo_delay_sec": tropo_delay_sec,
            "dm_delay_sec": dm_delay_sec,
            "sw_delay_sec": sw_delay_sec,
            "freq_bary_mhz": freq_bary_mhz,
            "tempo2_obs_state": tempo2_obs_state_export,
        }
        _overlay_jug = {
            "term_diagnostics": _overlay_td,
            "dt_sec": np.asarray(dt_sec, dtype=np.float64),
            "freq_bary_mhz": freq_bary_mhz,
            "compatibility": compatibility_mode,
        }
        _native_overlay = prepare_native_chain_from_simple_result(
            _overlay_jug, params, toas
        )
        _native_np = native_terms_to_numpy(_native_overlay)
        formbats_tt_arr = np.asarray(_native_np["correction_tt_sec"], dtype=np.float64)
        formbats_correction_tt = formbats_tt_arr
        tropo_delay_sec = np.asarray(_native_np["tropospheric_sec"], dtype=np.float64)
        roemer_sec = -np.asarray(_native_np["roemer_sec"], dtype=np.float64)
        sun_shapiro_sec = np.asarray(_native_np["shapiro_sun_sec"], dtype=np.float64)
        planet_shapiro_sec = np.asarray(
            _native_np["shapiro_planets_sec"], dtype=np.float64
        )
        dm_delay_sec = np.asarray(_native_np["tdis1_sec"], dtype=np.float64)
        sw_delay_sec = np.asarray(_native_np["tdis2_sec"], dtype=np.float64)
        roemer_shapiro = roemer_sec + sun_shapiro_sec + planet_shapiro_sec

    prebinary_delay_sec = (
        roemer_shapiro + dm_delay_sec + dmx_delay_sec + sw_delay_sec + tropo_delay_sec
    )

    ifte_delta_t_sec = np.asarray(ifte_delta_t_mjd(mjd_tt), dtype=np.float64)
    tempo2_clock_terms = compute_formbats_arrival(
        sat_arr,
        formbats_tt_arr,
        tt_tb,
        prebinary_delay_sec,
        params,
        correction_tt_teph_sec=tt_teph,
    )

    bbat_mjd = None
    torb_sec = None
    model_mjd_out = np.asarray(model_mjd, dtype=np.float64)
    if USE_NATIVE_BBAT_PHASE5:
        from jug.residuals.tempo2_spin import compute_tempo2_torb_sec

        bbat_mjd = tempo2_clock_terms.bbat_mjd
        model_mjd_out = tempo2_clock_terms.model_clock_mjd
        torb_sec = compute_tempo2_torb_sec(bbat_mjd, dt_sec, PEPOCH)

    return Tempo2HostSetupResult(
        formbats_correction_tt=formbats_correction_tt,
        tempo2_clock_terms=tempo2_clock_terms,
        tempo2_obs_state_export=tempo2_obs_state_export,
        earth_ssb_vel_km_s=earth_ssb_vel_km_s,
        dm_delay_sec=dm_delay_sec,
        sw_delay_sec=sw_delay_sec,
        tropo_delay_sec=tropo_delay_sec,
        roemer_sec=roemer_sec,
        sun_shapiro_sec=sun_shapiro_sec,
        planet_shapiro_sec=planet_shapiro_sec,
        roemer_shapiro=roemer_shapiro,
        prebinary_delay_sec=prebinary_delay_sec,
        ifte_delta_t_sec=ifte_delta_t_sec,
        bbat_mjd=bbat_mjd,
        torb_sec=torb_sec,
        model_mjd=model_mjd_out,
    )


def finalize_tempo2_host_residuals(
    *,
    params: dict,
    toas: list,
    dt_sec: np.ndarray,
    compatibility_mode: str,
    tempo2_clock_terms: Any,
    formbats_correction_tt: np.ndarray,
    tempo2_obs_state_export: dict,
    tropo_delay_sec: np.ndarray,
    dm_delay_sec: np.ndarray,
    sw_delay_sec: np.ndarray,
    freq_bary_mhz: np.ndarray,
    weights_scaled: np.ndarray,
    subtract_mean_in_phase: bool,
    tzr_phase_for_residuals,
    jump_phase: np.ndarray,
    external_pn: np.ndarray | None,
    external_pn_add: np.ndarray | None,
    track_val,
    addsat_sec: np.ndarray | None,
    phase_mean_mode: str,
    phase_bbat_mjd,
    phase_torb_sec,
) -> Tempo2HostFinalizeResult:
    """Hybrid tempo2 host residual routing (strict-parity production contract)."""
    from jug.residuals.simple_calculator import compute_phase_residuals
    from jug.residuals.tempo2_native.chain_jax import (
        compute_native_eval_residuals_jax,
        prepare_native_chain_from_simple_result,
    )

    _native_td = {
        "sat_mjd": np.asarray(tempo2_clock_terms.sat_mjd, dtype=np.float64),
        "correction_tt_sec": np.asarray(
            tempo2_clock_terms.correction_tt_sec, dtype=np.float64
        ),
        "correction_tt_tb_sec": np.asarray(
            tempo2_clock_terms.correction_tt_tb_sec, dtype=np.float64
        ),
        "formbats_correction_tt_sec": np.asarray(
            formbats_correction_tt, dtype=np.float64
        ),
        "tropo_delay_sec": np.asarray(tropo_delay_sec, dtype=np.float64),
        "dm_delay_sec": np.asarray(dm_delay_sec, dtype=np.float64),
        "sw_delay_sec": np.asarray(sw_delay_sec, dtype=np.float64),
        "freq_bary_mhz": np.asarray(freq_bary_mhz, dtype=np.float64),
        "tempo2_obs_state": tempo2_obs_state_export,
    }
    _jug = {
        "term_diagnostics": _native_td,
        "dt_sec": np.asarray(dt_sec, dtype=np.float64),
        "freq_bary_mhz": freq_bary_mhz,
        "compatibility": compatibility_mode,
    }
    native = prepare_native_chain_from_simple_result(_jug, params, toas)
    use_taylor_track2_spin = track_val is not None and int(track_val) == -2
    use_taylor_nontrack_spin = track_val is None
    if use_taylor_track2_spin or use_taylor_nontrack_spin:
        if use_taylor_track2_spin:
            residuals_us, residuals_sec, pulse_number = compute_phase_residuals(
                dt_sec,
                params,
                weights_scaled,
                subtract_mean=subtract_mean_in_phase,
                tzr_phase=tzr_phase_for_residuals,
                jump_phase=jump_phase,
                external_pulse_numbers=external_pn,
                track_val=int(track_val),
                external_pn_add=external_pn_add,
                bbat_mjd=phase_bbat_mjd,
                torb_sec=phase_torb_sec,
                use_native_bbat_phase5=False,
                addsat_sec=addsat_sec,
                mean_mode=phase_mean_mode,
            )
        else:
            residuals_us, residuals_sec, pulse_number = compute_phase_residuals(
                dt_sec,
                params,
                weights_scaled,
                subtract_mean=subtract_mean_in_phase,
                tzr_phase=tzr_phase_for_residuals,
                jump_phase=jump_phase,
                mean_mode=phase_mean_mode,
            )
    else:
        jump_j = None if jump_phase is None else np.asarray(jump_phase, dtype=np.float64)
        tzr_j = None if tzr_phase_for_residuals is None else float(tzr_phase_for_residuals)
        residuals_sec_jax, pulse_number_jax, native = compute_native_eval_residuals_jax(
            params=params,
            toas=toas,
            jug_result=_jug,
            pulse_numbers=external_pn,
            pn_add=external_pn_add,
            jump_phase=jump_j,
            tzr_phase=tzr_j,
            subtract_mean=subtract_mean_in_phase,
            mean_mode=phase_mean_mode,
            track_val=int(track_val) if track_val is not None else -2,
            weights=jnp.asarray(weights_scaled, dtype=jnp.float64),
            addsat_sec=addsat_sec,
        )
        residuals_sec = np.asarray(jax.device_get(residuals_sec_jax), dtype=np.float64)
        pulse_number = np.asarray(jax.device_get(pulse_number_jax), dtype=np.longdouble)
        residuals_us = residuals_sec * 1e6

    dm_delay_sec = np.asarray(jax.device_get(native.tdis1_sec), dtype=np.float64)
    sw_delay_sec = np.asarray(jax.device_get(native.tdis2_sec), dtype=np.float64)
    return Tempo2HostFinalizeResult(
        residuals_us=residuals_us,
        residuals_sec=residuals_sec,
        pulse_number=pulse_number,
        native=native,
        dm_delay_sec=dm_delay_sec,
        sw_delay_sec=sw_delay_sec,
    )


def finalize_pint_host_residuals(
    *,
    dt_sec: np.ndarray,
    params: dict,
    weights_scaled: np.ndarray,
    subtract_mean_in_phase: bool,
    tzr_phase_for_residuals,
    jump_phase: np.ndarray,
    external_pn: np.ndarray | None,
    track_val,
    external_pn_add: np.ndarray | None,
    phase_bbat_mjd,
    phase_torb_sec,
    addsat_sec: np.ndarray | None,
    phase_mean_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PINT-family host residual finalization via ``compute_phase_residuals``."""
    from jug.residuals.simple_calculator import compute_phase_residuals

    return compute_phase_residuals(
        dt_sec,
        params,
        weights_scaled,
        subtract_mean=subtract_mean_in_phase,
        tzr_phase=tzr_phase_for_residuals,
        jump_phase=jump_phase,
        external_pulse_numbers=external_pn,
        track_val=int(track_val) if track_val is not None else None,
        external_pn_add=external_pn_add,
        bbat_mjd=phase_bbat_mjd,
        torb_sec=phase_torb_sec,
        use_native_bbat_phase5=USE_NATIVE_BBAT_PHASE5,
        addsat_sec=addsat_sec,
        mean_mode=phase_mean_mode,
    )
