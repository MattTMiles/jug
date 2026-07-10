"""Tempo2 native chain submodule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path
from jug.delays.tempo2_geometry import (
    build_tempo2_pulsar_vectors,
    pmrv_rad_per_century,
    tempo2_dilate_freq_enabled,
)
from jug.utils.constants import SECS_PER_DAY
from jug.residuals.tempo2.model import (
    Tempo2ModelStatic,
    _dm_coeffs_from_params,
    _eop_to_jax,
    _spk_to_jax,
    build_tempo2_model_static,
    compute_dm_vals_jax,
    compute_tempo2_get_correction_tt_jax,
    compute_tempo2_toa_model_fixed_state_nonlinear_jax,
    compute_tempo2_toa_model_jax,
    compute_tempo2_toa_model_staging_with_host_inputs_jax,
    host_frozen_vectors_from_tempo2_obs_state,
    run_tempo2_toa_model_with_fixed_ifte_geometry,
)
from jug.residuals.tempo2.graph_config import (
    TEMPO2_GRAPH_FIXED_STATE_BCLT,
    TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
    TEMPO2_GRAPH_FULL,
    TEMPO2_GRAPH_STAGED_BCLT,
    tempo2_graph_mode,
)
from jug.residuals.tempo2.model.bbat_lite import bbat_lite_daysec_from_pack
from jug.residuals.tempo2.spin_jax import spin_params_to_jax
from jug.residuals.tempo2.types import Tempo2Terms
from jug.utils.timescales import is_tempo2_si_units, parse_timescale
from .common import (
    NativeDeltaPack,
    _dm_coeffs_jax,
    _param_scalar_jax,
    _spin_f_terms_jax,
    pulsar_vectors_from_params_jax,
    track2_pulse_arrays_from_toas,
)

def compute_tempo2_terms_jax(
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
    model_static: Tempo2ModelStatic | None = None,
    tdb_mjd=None,
    sat_int_day=None,
    sat_sec_in_day=None,
    bclt_max_iter: int | None = None,
    torb_binary_sec=None,
) -> Tempo2Terms:
    """Compute tempo2-native terms through ``compute_tempo2_toa_model_jax``."""
    del (
        toas,
        tdis1_sec,
        tdis2_sec,
        utc_to_tdb_sec,
        prebinary_override_sec,
        model_mjd,
        use_model_epoch_batcorr,
        pulse_numbers,
        pn_add,
        jump_phase,
        tzr_phase,
        addsat_sec,
        ephem_path,
        tdb_mjd,
        correction_tt_sec,
        correction_tt_tb_sec,
        observatory_earth_km,
        earth_ssb_km,
        earth_ssb_vel_km_s,
        site_vel_km_s,
        obs_planets_ls_fixed,
        formbats_tt_sec,
    )

    if ssb_obs_ls_fixed is not None or obs_sun_ls_fixed is not None:
        raise ValueError(
            "Unified Phase 4: ssb_obs_ls and obs_sun_ls must be None; "
            "geometry computed in-graph. "
            "For host-precomputed geometry use "
            "compute_tempo2_toa_model_staging_with_host_inputs_jax."
        )
    if model_static is None:
        raise ValueError(
            "compute_tempo2_terms_jax requires model_static with "
            "clock, IFTE, and SPK tables"
        )

    if freq_mhz_topocentric is not None:
        freq = np.asarray(freq_mhz_topocentric, dtype=np.float64)
    else:
        freq = np.asarray(freq_mhz, dtype=np.float64)

    terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=freq,
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        use_native_ecliptic=use_native_ecliptic,
        sat_int_day=(
            None if sat_int_day is None else np.asarray(sat_int_day, dtype=np.float64)
        ),
        sat_sec_in_day=(
            None if sat_sec_in_day is None else np.asarray(sat_sec_in_day, dtype=np.float64)
        ),
        bclt_max_iter=bclt_max_iter,
        torb_binary_sec=(
            None
            if torb_binary_sec is None
            else np.asarray(torb_binary_sec, dtype=np.float64)
        ),
    )
    return terms


def compute_tempo2_residuals_jax(
    *,
    native_terms: Tempo2Terms,
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
    from jug.residuals.tempo2.spin_jax import (
        compute_tempo2_phase5_daysec,
        spin_params_to_jax,
        track_minus2_frac_phase_jax,
    )

    f_terms, _pepoch, pep_int, pep_frac = spin_params_to_jax(params)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    bbat_int = native_terms.bbat_int_day
    bbat_sec = native_terms.bbat_sec_in_day
    torb = native_terms.torb_sec
    if int(track_val) == -2 and pulse_numbers is not None and pn_add is not None:
        phase5 = compute_tempo2_phase5_daysec(
            bbat_int,
            bbat_sec,
            torb,
            f_terms,
            pep_int,
            pep_frac,
            jump_phase=jump_j,
            tzr_phase=tzr_j,
        )
        frac, pulse = track_minus2_frac_phase_jax(
            phase5,
            bbat_int,
            f_terms[0],
            jnp.asarray(pulse_numbers, dtype=jnp.int64),
            jnp.asarray(pn_add, dtype=jnp.int64),
        )
    else:
        pulse = jnp.zeros_like(torb)
        phase5 = compute_tempo2_phase5_daysec(
            bbat_int,
            bbat_sec,
            torb,
            f_terms,
            pep_int,
            pep_frac,
            jump_phase=jump_j,
            tzr_phase=tzr_j,
        )
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / f_terms[0]
    if subtract_mean:
        if mean_mode == "weighted":
            w = jnp.asarray(weights, dtype=jnp.float64)
            residual_sec = residual_sec - jnp.sum(residual_sec * w) / jnp.sum(w)
        else:
            residual_sec = residual_sec - jnp.mean(residual_sec)
    return residual_sec, pulse, native_terms


def compute_spin_residual_sec_jax(
    native_terms: Tempo2Terms,
    params,
    *,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    track_val: int = -2,
) -> jnp.ndarray:
    """Spin/track-only residual from precomputed delay terms (diagnostics helper)."""
    residual_sec, _, _ = compute_tempo2_residuals_jax(
        native_terms=native_terms,
        params=params,
        weights=jnp.ones(native_terms.sat_mjd.shape[0], dtype=jnp.float64),
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
        subtract_mean=subtract_mean,
        track_val=track_val,
    )
    return residual_sec

def compute_terms_and_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> tuple[Tempo2Terms, jnp.ndarray]:
    """Recompute tempo2-native terms and residuals for any graph mode pack."""
    pos, vel, acc = pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms = _spin_f_terms_jax(params)
    pepoch = jnp.asarray(_param_scalar_jax(params, "PEPOCH"), dtype=jnp.float64)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    common = dict(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=pos,
        vel_pulsar=vel,
        acc_pulsar=acc,
        dt_emission_sec=pack.dt_emission_sec,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        ne_sw=pack.ne_sw,
        posepoch_mjd=pack.posepoch_mjd,
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
        sat_int_day=pack.sat_int_day,
        sat_sec_in_day=pack.sat_sec_in_day,
        pep_int=pack.pep_int,
        pep_frac=pack.pep_frac,
        torb_binary_sec=pack.torb_binary_sec,
    )
    if pack.mode == TEMPO2_GRAPH_FIXED_STATE_BCLT:
        terms, residual_sec = compute_tempo2_toa_model_fixed_state_nonlinear_jax(
            tropo_sec=pack.tropo_sec,
            earth_ssb_km=pack.earth_ssb_km,
            observatory_earth_km=pack.observatory_earth_km,
            site_vel_km_s=pack.site_vel_km_s,
            ssb_obs_ls=pack.ssb_obs_ls,
            obs_sun_ls=pack.obs_sun_ls,
            obs_jupiter_ls=pack.obs_jupiter_ls,
            planet_obs_ls=pack.planet_obs_ls,
            correction_tt_sec_pre=pack.correction_tt_sec,
            correction_tt_tb_sec_pre=pack.correction_tt_tb_sec,
            einstein_rate=pack.einstein_rate,
            dt_ssb_ref_sec=pack.dt_ssb_ref_sec,
            **common,
        )
    elif pack.mode == TEMPO2_GRAPH_FULL:
        terms, residual_sec = compute_tempo2_toa_model_jax(
            obs_itrf_km=pack.obs_itrf_km,
            spk_packed=pack.spk_packed,
            eop_packed=pack.eop_packed,
            chain_mjd_tables=pack.chain_mjd_tables,
            chain_offset_tables=pack.chain_offset_tables,
            bipm_mjd=pack.bipm_mjd,
            bipm_offset=pack.bipm_offset,
            ifte_records=pack.ifte_records,
            ifte_start_jd=pack.ifte_start_jd,
            ifte_end_jd=pack.ifte_end_jd,
            ifte_step_jd=pack.ifte_step_jd,
            ifte_coef_offset=pack.ifte_coef_offset,
            ifte_ncf=pack.ifte_ncf,
            ifte_na=pack.ifte_na,
            obs_site_latitude_rad=pack.obs_site_latitude_rad,
            obs_site_longitude_rad=pack.obs_site_longitude_rad,
            obs_site_height_m=pack.obs_site_height_m,
            obs_site_pressure_mbar=pack.obs_site_pressure_mbar,
            correct_troposphere=pack.correct_troposphere,
            bclt_max_iter=pack.bclt_max_iter,
            **common,
        )
    else:
        terms, residual_sec = compute_tempo2_toa_model_staging_with_host_inputs_jax(
            tropo_sec=pack.tropo_sec,
            earth_ssb_km=pack.earth_ssb_km,
            observatory_earth_km=pack.observatory_earth_km,
            site_vel_km_s=pack.site_vel_km_s,
            ssb_obs_ls=pack.ssb_obs_ls,
            obs_sun_ls=pack.obs_sun_ls,
            obs_jupiter_ls=pack.obs_jupiter_ls,
            planet_obs_ls=pack.planet_obs_ls,
            correction_tt_sec_pre=pack.correction_tt_sec,
            correction_tt_tb_sec_pre=pack.correction_tt_tb_sec,
            einstein_rate=pack.einstein_rate,
            bclt_max_iter=pack.bclt_max_iter,
            **common,
        )
    return terms, residual_sec


def compute_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals for any graph mode pack."""
    _, residual_sec = compute_terms_and_residual_sec_jax(params, pack)
    return residual_sec


def compute_bbat_delay_change_sec_jax_stripped(
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """BBAT displacement for ``fixed_state_stripped`` (single pert vs host ref)."""
    bbat_int, bbat_sec = bbat_lite_daysec_from_pack(params_pert, pack)
    bbat_delta_sec = (
        (bbat_int - pack.bbat_ref_int_day) * SECS_PER_DAY
        + (bbat_sec - pack.bbat_ref_sec_in_day)
    )
    return -bbat_delta_sec


def compute_bbat_delay_change_sec_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Return delay change from native bbat motion for local phase deltas.

    The tempo2 ``phase5`` closure uses ``torb = dt_emit - (bbat - PEPOCH)``.
    With ``dt_emit`` frozen in the fitting setup, absolute native residuals are
    parity values, but delay changes cancel out of the forward residual.  For
    nonlinear fitting we recover the small local timing perturbation from the
    native bbat displacement.  A later bbat is equivalent to a smaller emission
    delay, hence the sign flip.
    """
    if pack.mode == TEMPO2_GRAPH_FIXED_STATE_STRIPPED:
        return compute_bbat_delay_change_sec_jax_stripped(params_pert, pack)
    terms_ref, _ = compute_terms_and_residual_sec_jax(params_ref, pack)
    terms_pert, _ = compute_terms_and_residual_sec_jax(params_pert, pack)
    bbat_delta_sec = (
        (terms_pert.bbat_int_day - terms_ref.bbat_int_day) * SECS_PER_DAY
        + (terms_pert.bbat_sec_in_day - terms_ref.bbat_sec_in_day)
    )
    return -bbat_delta_sec


def compute_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Native residual delta: ``res(θ+Δθ) − res(θ)`` with optional mean on delta."""
    res_ref = compute_residual_sec_jax(params_ref, pack)
    res_pert = compute_residual_sec_jax(params_pert, pack)
    delta = res_pert - res_ref
    if pack.subtract_mean:
        delta = delta - jnp.mean(delta)
    return delta


def compute_fixed_state_nonlinear_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the fixed-state nonlinear tail."""
    return compute_residual_sec_jax(params, pack)


def compute_fixed_state_nonlinear_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Fixed-state nonlinear residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    return compute_residual_delta_jax(params_ref, params_pert, pack)


def compute_staged_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the host-frozen staging tail."""
    return compute_residual_sec_jax(params, pack)


def compute_staged_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Host-frozen residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    return compute_residual_delta_jax(params_ref, params_pert, pack)


def compute_full_chain_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through ``compute_tempo2_toa_model_jax``."""
    return compute_residual_sec_jax(params, pack)


def compute_full_chain_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Full native-chain residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    return compute_residual_delta_jax(params_ref, params_pert, pack)

