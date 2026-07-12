"""Tempo2-compatible host pipeline (setup + native residual finalization).

Production host routing contract (two-part bbat parity, 2026-07-07)
--------------------------------------------------------------------
Host **residuals** (libstempo parity):

* ``TRACK`` absent  → Taylor ``compute_phase_residuals`` (sequential wrap)
* ``TRACK == -2``   → Taylor emission-time spin + legacy ``-pn`` wrapping
  (matches libstempo; ``phase5@bbat`` TRACK−2 mis-handles ``-addsat`` even when
  sat carries the read-time shift)
* other ``TRACK``   → ``compute_eval_residuals_jax`` (two-part formBats)

JAX fit/autodiff always uses the native two-part ``staged_bclt`` tail
(``phase5@bbat`` + TRACK−2 in-graph). Host vs fit spin routing therefore
differs only for ``TRACK == -2`` until ``-addsat`` + ``phase5@bbat`` coupling
matches tempo2 ``formResiduals.C`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
from jug.residuals.tempo2.graph_config import USE_NATIVE_BBAT_PHASE5
from jug.residuals.tempo2.types import Tempo2Terms
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
    clock_feedback_delta_sec: np.ndarray
    bclt_dt_ssb_sec: np.ndarray | None = None
    # The overlay's Tempo2Terms (native BCLT chain), threaded through so the
    # finalize stage can reuse it instead of rebuilding the whole chain.
    # None when skip_native_bclt_overlay=True.
    native_terms: Any | None = None


@dataclass
class Tempo2HostFinalizeResult:
    """Host tempo2 residual finalization outputs."""

    residuals_us: np.ndarray
    residuals_sec: np.ndarray
    pulse_number: np.ndarray
    native: Tempo2Terms
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
        per_toa_obs_itrf_km,
        resolve_tempo2_ephemeris_path,
    )
    from jug.delays.tempo2_geometry import (
        build_tempo2_pulsar_vectors,
        compute_tempo2_dm_delays_sec,
        ecliptic_obliquity_rad,
        psr_pos_at_delt,
        tempo2_dilate_freq_enabled,
        tempo2_observatory_chain_vectors,
    )
    from jug.residuals.tempo2_clock import (
        compute_formbats_arrival,
        compute_get_correction_tt_sec,
        compute_site_clock_corrections_sec,
    )
    from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
    from jug.residuals.tempo2.types import tempo2_terms_to_numpy
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
    formbats_correction_tt_nofeedback = compute_get_correction_tt_sec(
        toas,
        obs_clocks=obs_clocks,
        obs_clock_default=obs_clock,
        bipm_clock=bipm_clock,
        all_obs_codes=all_obs_codes,
        time_offsets=time_offsets,
        feedback_iters=1,
    )
    # clkcorr.C evaluates each UTC->TT hop at sat + accumulated_corr/SECDAY;
    # the production tdb_mjd/model_mjd path evaluates the same chain at raw SAT.
    # This per-TOA delta is the dt-chain piece missing from the Taylor dt_sec.
    clock_feedback_delta_sec = (
        np.asarray(formbats_correction_tt, dtype=np.float64)
        - np.asarray(formbats_correction_tt_nofeedback, dtype=np.float64)
    )

    ephem_path = resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))
    sat_int_arr = np.array([t.mjd_int for t in toas], dtype=np.float64)
    sat_sec_arr = np.array([t.mjd_frac * SECS_PER_DAY for t in toas], dtype=np.float64)
    sat_arr = sat_int_arr + sat_sec_arr / SECS_PER_DAY
    formbats_tt_arr = np.asarray(formbats_correction_tt, dtype=np.float64)
    obs_itrf = per_toa_obs_itrf_km(
        toas, np.asarray(obs_itrf_km, dtype=np.float64).reshape(3)
    )

    # readEphemeris.C scales one_au by IFTE_K only for SI_UNITS (TCB) pulsars.
    # For ELONG/ELAT pulsars tempo2 rotates the whole obsn[] geometry to
    # ecliptic (readEphemeris.C / get_obsCoord.C equ2ecl); posPulsar is built
    # in the ecliptic frame, so every dot product downstream needs the same.
    use_native_ecliptic = bool(params.get("_ecliptic_coords", False))
    ecl_obl_rad = ecliptic_obliquity_rad(params, use_native_ecliptic)
    geo_boot = bootstrap_tempo2_observatory_state(
        sat_arr,
        formbats_tt_arr,
        obs_itrf,
        ephem_path=ephem_path,
        params=params,
        si_units=is_tempo2_si_units(parse_timescale(params)),
        t2c_method=str(getattr(engine_profile, "t2cmethod", "IAU2000B")),
        ecl_obl_rad=ecl_obl_rad,
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
        use_native_ecliptic=use_native_ecliptic,
    )
    # tropo.C uses posPulsarEquatorial against the GCRS zenith regardless of
    # the pulsar coordinate frame.
    if use_native_ecliptic:
        from jug.delays.barycentric import rotate_ecliptic_to_equatorial

        pos_pulsar_equatorial = rotate_ecliptic_to_equatorial(
            pos_pulsar[None, :], ecl_obl_rad
        )[0]
    else:
        pos_pulsar_equatorial = pos_pulsar
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

        # Tropo mapping is site-specific; evaluate per observatory group.
        tropo_delay_sec = np.zeros(len(toas), dtype=np.float64)
        obs_codes_per_toa = [t.observatory.lower() for t in toas]
        for code in sorted(set(obs_codes_per_toa)):
            idxs = [i for i, c in enumerate(obs_codes_per_toa) if c == code]
            tropo_delay_sec[idxs] = compute_tempo2_tropo_delay_host(
                sat_arr[idxs],
                formbats_tt_arr[idxs],
                obs_itrf_km=obs_itrf[idxs[0]],
                pos_pulsar=pos_pulsar_equatorial,
                mapping_clock_sec=correction_tt[idxs],
            )

    bclt_dt_ssb_sec = None
    native_terms = None
    if not skip_native_bclt_overlay:
        _overlay_td = {
            "sat_mjd": sat_arr,
            "sat_int_day": sat_int_arr,
            "sat_sec_in_day": sat_sec_arr,
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
        _native_overlay = prepare_tempo2_chain_from_simple_result(
            _overlay_jug, params, toas
        )
        native_terms = _native_overlay
        _native_np = tempo2_terms_to_numpy(_native_overlay)
        bclt_dt_ssb_sec = np.asarray(_native_np["dt_ssb_sec"], dtype=np.float64)
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
        clock_feedback_delta_sec=clock_feedback_delta_sec,
        bclt_dt_ssb_sec=bclt_dt_ssb_sec,
        native_terms=native_terms,
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
    prebinary_delay_sec: np.ndarray,
    total_delay_sec: np.ndarray,
    native_terms: Tempo2Terms | None = None,
) -> Tempo2HostFinalizeResult:
    """Tempo2 host residuals: Taylor for TRACK−2/no-TRACK; native for other TRACK.

    ``native_terms`` is the overlay's already-built native chain (from
    ``run_tempo2_host_stage``). In the Taylor branch the chain is needed only
    for the ``tdis1/tdis2`` diagnostics, so reusing it skips a second full
    ``prepare_tempo2_chain_from_simple_result`` build; the other-TRACK branch
    builds its own chain inside ``compute_eval_residuals_jax`` either way.
    """
    from jug.residuals.phase import compute_phase_residuals
    from jug.residuals.tempo2.common import sat_daysec_numpy_from_td_and_toas
    from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
    from jug.residuals.tempo2.orchestrator import compute_eval_residuals_jax

    sat_int, sat_sec = sat_daysec_numpy_from_td_and_toas(
        {"sat_mjd": np.asarray(tempo2_clock_terms.sat_mjd, dtype=np.float64)},
        toas,
    )
    _native_td = {
        "sat_mjd": np.asarray(tempo2_clock_terms.sat_mjd, dtype=np.float64),
        "sat_int_day": sat_int,
        "sat_sec_in_day": sat_sec,
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
        "prebinary_delay_sec": np.asarray(prebinary_delay_sec, dtype=np.float64),
        "total_delay_sec": np.asarray(total_delay_sec, dtype=np.float64),
        "tempo2_obs_state": tempo2_obs_state_export,
    }
    _jug = {
        "term_diagnostics": _native_td,
        "dt_sec": np.asarray(dt_sec, dtype=np.float64),
        "freq_bary_mhz": freq_bary_mhz,
        "compatibility": compatibility_mode,
    }
    use_taylor_host_spin = track_val is None or int(track_val) == -2
    if use_taylor_host_spin:
        # Reuse the overlay chain when available; only skip_native_bclt_overlay
        # sessions need a fresh build here (and only for tdis diagnostics).
        if native_terms is not None:
            native = native_terms
        else:
            native = prepare_tempo2_chain_from_simple_result(_jug, params, toas)
        residuals_us, residuals_sec, pulse_number = compute_phase_residuals(
            dt_sec,
            params,
            weights_scaled,
            subtract_mean=subtract_mean_in_phase,
            tzr_phase=tzr_phase_for_residuals,
            jump_phase=jump_phase,
            external_pulse_numbers=external_pn,
            track_val=int(track_val) if track_val is not None else None,
            external_pn_add=external_pn_add,
            addsat_sec=addsat_sec,
            mean_mode=phase_mean_mode,
        )
    else:
        jump_j = None if jump_phase is None else np.asarray(jump_phase, dtype=np.float64)
        tzr_j = None if tzr_phase_for_residuals is None else float(tzr_phase_for_residuals)
        residuals_sec_jax, pulse_number_jax, native = compute_eval_residuals_jax(
            params=params,
            toas=toas,
            jug_result=_jug,
            pulse_numbers=external_pn,
            pn_add=external_pn_add,
            jump_phase=jump_j,
            tzr_phase=tzr_j,
            subtract_mean=subtract_mean_in_phase,
            mean_mode=phase_mean_mode,
            track_val=int(track_val),
            weights=jnp.asarray(weights_scaled, dtype=jnp.float64),
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


@dataclass
class Tempo2HostStageResult:
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
    total_delay_sec: np.ndarray
    delay_sec: np.ndarray
    dt_sec: np.ndarray
    bclt_dt_ssb_sec: np.ndarray | None = None
    native_terms: Any | None = None


def run_tempo2_host_stage(
    *,
    mjd_utc,
    obs_clocks,
    bipm_clock,
    toas,
    all_obs_codes,
    obs_clock,
    time_offsets,
    params,
    obs_itrf_km,
    dm_eff,
    freq_bary_mhz,
    dt_sec,
    model_mjd,
    PEPOCH,
    compatibility_mode,
    engine_profile,
    correct_troposphere,
    roemer_sec,
    sun_shapiro_sec,
    planet_shapiro_sec,
    roemer_shapiro,
    dm_delay_sec,
    sw_delay_sec,
    tropo_delay_sec,
    dmx_delay_sec,
    skip_native_bclt_overlay,
    total_delay_sec,
    delay_sec,
) -> Tempo2HostStageResult:
    # Keep the kernel-time delay terms so the native-overlay delta can be
    # folded into the total delay below (the kernel ran with the provider
    # geometry; the overlay recomputes Roemer/DM/SW from the exact
    # bootstrap chain).
    _pre_overlay_roemer_shapiro = np.asarray(roemer_shapiro, dtype=np.float64)
    _pre_overlay_dm = np.asarray(dm_delay_sec, dtype=np.float64)
    _pre_overlay_sw = np.asarray(sw_delay_sec, dtype=np.float64)

    _t2_setup = compute_tempo2_host_setup(
        mjd_utc=mjd_utc,
        obs_clocks=obs_clocks,
        bipm_clock=bipm_clock,
        toas=toas,
        all_obs_codes=all_obs_codes,
        obs_clock=obs_clock,
        time_offsets=time_offsets,
        params=params,
        obs_itrf_km=obs_itrf_km,
        dm_eff=dm_eff,
        freq_bary_mhz=freq_bary_mhz,
        dt_sec=dt_sec,
        model_mjd=model_mjd,
        PEPOCH=PEPOCH,
        compatibility_mode=compatibility_mode,
        engine_profile=engine_profile,
        correct_troposphere=correct_troposphere,
        roemer_sec=roemer_sec,
        sun_shapiro_sec=sun_shapiro_sec,
        planet_shapiro_sec=planet_shapiro_sec,
        roemer_shapiro=roemer_shapiro,
        dm_delay_sec=dm_delay_sec,
        sw_delay_sec=sw_delay_sec,
        tropo_delay_sec=tropo_delay_sec,
        dmx_delay_sec=dmx_delay_sec,
        skip_native_bclt_overlay=skip_native_bclt_overlay,
    )
    formbats_correction_tt = _t2_setup.formbats_correction_tt
    tempo2_clock_terms = _t2_setup.tempo2_clock_terms
    tempo2_obs_state_export = _t2_setup.tempo2_obs_state_export
    earth_ssb_vel_km_s = _t2_setup.earth_ssb_vel_km_s
    dm_delay_sec = _t2_setup.dm_delay_sec
    sw_delay_sec = _t2_setup.sw_delay_sec
    tropo_delay_sec = _t2_setup.tropo_delay_sec
    roemer_sec = _t2_setup.roemer_sec
    sun_shapiro_sec = _t2_setup.sun_shapiro_sec
    planet_shapiro_sec = _t2_setup.planet_shapiro_sec
    roemer_shapiro = _t2_setup.roemer_shapiro
    prebinary_delay_sec = _t2_setup.prebinary_delay_sec
    ifte_delta_t_sec = _t2_setup.ifte_delta_t_sec
    bbat_mjd = _t2_setup.bbat_mjd
    torb_sec = _t2_setup.torb_sec
    model_mjd = _t2_setup.model_mjd
    bclt_dt_ssb_sec = _t2_setup.bclt_dt_ssb_sec

    # The JAX kernel summed the provider-geometry Roemer/DM/SW; the native
    # BCLT overlay recomputes them from the exact bootstrap chain (fixed
    # POSEPOCH direction + dt_pm/dt_px terms, dt_SSB iteration).  Fold the
    # difference into the total so residuals use the native terms (for
    # high-PM/PX binaries like J0437-4715 the provider-vs-native gap is
    # ~20 ns and otherwise leaks into the residuals).
    _overlay_delta_sec = (
        (np.asarray(roemer_shapiro, dtype=np.float64) - _pre_overlay_roemer_shapiro)
        + (np.asarray(dm_delay_sec, dtype=np.float64) - _pre_overlay_dm)
        + (np.asarray(sw_delay_sec, dtype=np.float64) - _pre_overlay_sw)
    )
    if np.any(_overlay_delta_sec != 0.0):
        _overlay_delta_ld = np.asarray(_overlay_delta_sec, dtype=np.longdouble)
        total_delay_sec = total_delay_sec + _overlay_delta_ld
        delay_sec = total_delay_sec
        dt_sec = dt_sec - _overlay_delta_ld

    # formBats.C subtracts troposphericDelay inside ``bat``, so tempo2's
    # spin argument includes the troposphere.  The kernel stage above ran
    # with tropo=0 (the tempo2-native troposphere needs the formBats clock
    # chain computed inside the host setup), so fold it into the total
    # delay and emission dt now.  Delays enter dt linearly, so this
    # post-hoc adjustment is exact; without it the Taylor host residuals
    # are missing up to ~100 ns of troposphere vs tempo2 (wsrt167 floor).
    if np.any(np.asarray(tropo_delay_sec) != 0.0):
        _tropo_ld = np.asarray(tropo_delay_sec, dtype=np.longdouble)
        total_delay_sec = total_delay_sec + _tropo_ld
        delay_sec = total_delay_sec
        dt_sec = dt_sec - _tropo_ld

    # clkcorr.C feedback: the production model_mjd/tdb_mjd evaluate the UTC→TT
    # clock chain at raw SAT, but tempo2 evaluates each hop at sat+corr/SECDAY.
    # For the Taylor host spin (TRACK absent or TRACK -2) fold the per-TOA feedback
    # delta into the emission time so residuals match tempo2's bat-based epoch.
    # The native-JAX path (other TRACK) already consumes the feedback correction
    # via formbats terms, so gate on the Taylor host condition to leave the
    # full-JAX pipeline untouched.
    _track_val_fb = params.get("TRACK", None)
    _use_taylor_host = _track_val_fb is None or int(_track_val_fb) == -2
    if _use_taylor_host:
        _clock_fb_delta_sec = np.asarray(
            _t2_setup.clock_feedback_delta_sec, dtype=np.longdouble
        )
        dt_sec = dt_sec + _clock_fb_delta_sec
        model_mjd = (
            np.asarray(model_mjd, dtype=np.longdouble)
            + _clock_fb_delta_sec / np.longdouble(SECS_PER_DAY)
        )

    return Tempo2HostStageResult(
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
        model_mjd=model_mjd,
        total_delay_sec=total_delay_sec,
        delay_sec=delay_sec,
        dt_sec=dt_sec,
        bclt_dt_ssb_sec=bclt_dt_ssb_sec,
        native_terms=_t2_setup.native_terms,
    )
