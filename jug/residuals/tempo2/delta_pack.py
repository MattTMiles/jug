"""Tempo2 native chain submodule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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
    compute_tempo2_toa_model_fixed_state_nonlinear_jax,
    compute_tempo2_toa_model_jax,
    compute_tempo2_toa_model_staging_with_host_inputs_jax,
    host_frozen_vectors_from_tempo2_obs_state,
    run_tempo2_toa_model_with_fixed_ifte_geometry,
)
from jug.residuals.tempo2.graph_config import (
    TEMPO2_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_GRAPH_FULL,
    TEMPO2_GRAPH_STAGED_BCLT,
    tempo2_graph_mode,
)
from jug.residuals.tempo2.spin_jax import spin_params_to_jax
from jug.residuals.tempo2.types import Tempo2Terms
from jug.utils.timescales import is_tempo2_si_units, parse_timescale
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import GeneralFitSetup

from .common import (
    NativeDeltaPack,
    _load_model_static_for_native_chain,
    _chain_mode,
    _param_scalar_jax,
    pulsar_vectors_from_params_jax,
    sat_daysec_numpy_from_td_and_toas,
    track2_pulse_arrays_from_toas,
)
from .terms import compute_tempo2_terms_jax


def _bclt_max_iter_from_setup(setup: "GeneralFitSetup") -> int:
    """Resolve the fixed BCLT scan length from ``setup.tempo2_jug_options``."""
    from jug.residuals.tempo2.calculate_bclt_jax import bclt_jax_fixed_iter_count
    from jug.timing import resolve_tempo2_jug_options

    opts = resolve_tempo2_jug_options(getattr(setup, "tempo2_jug_options", None))
    return bclt_jax_fixed_iter_count(opts.get("bclt_fixed_iter"))


def build_delta_pack(setup: "GeneralFitSetup") -> NativeDeltaPack | None:
    """Build JAX-static cache for full-chain native residual deltas."""
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "model_mjd": static.get("model_mjd", static["term_diagnostics"].get("sat_mjd")),
        "ssb_obs_pos_ls": static.get("ssb_obs_pos_ls"),
        "obs_sun_pos_ls": static.get("obs_sun_pos_ls"),
        "obs_planet_pos_ls": static.get("obs_planet_pos_ls"),
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    units = parse_timescale(params)
    tropo = model_static.tropo_packed
    jump = getattr(setup, "jump_phase", None)
    tzr = getattr(setup, "tzr_phase", None)
    bclt_max_iter = _bclt_max_iter_from_setup(setup)
    return NativeDeltaPack(
        mode=TEMPO2_GRAPH_FULL,
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(static["dt_sec"], dtype=jnp.float64),
        obs_itrf_km=jnp.asarray(model_static.obs_itrf_km, dtype=jnp.float64),
        spk_packed=_spk_to_jax(model_static.spk_packed),
        eop_packed=_eop_to_jax(model_static.eop_packed),
        chain_mjd_tables=tuple(
            jnp.asarray(t, dtype=jnp.float64) for t in model_static.chain_mjd_tables
        ),
        chain_offset_tables=tuple(
            jnp.asarray(t, dtype=jnp.float64) for t in model_static.chain_offset_tables
        ),
        bipm_mjd=jnp.asarray(model_static.bipm_mjd, dtype=jnp.float64),
        bipm_offset=jnp.asarray(model_static.bipm_offset, dtype=jnp.float64),
        ifte_records=jnp.asarray(model_static.ifte_records, dtype=jnp.float64),
        ifte_start_jd=jnp.asarray(model_static.ifte_start_jd, dtype=jnp.float64),
        ifte_end_jd=jnp.asarray(model_static.ifte_end_jd, dtype=jnp.float64),
        ifte_step_jd=jnp.asarray(model_static.ifte_step_jd, dtype=jnp.float64),
        ifte_coef_offset=int(model_static.ifte_coef_offset),
        ifte_ncf=int(model_static.ifte_ncf),
        ifte_na=int(model_static.ifte_na),
        ne_sw=float(model_static.ne_sw),
        correct_troposphere=bool(model_static.correct_troposphere),
        obs_site_latitude_rad=(
            float(tropo.latitude_rad) if tropo is not None else 0.0
        ),
        obs_site_longitude_rad=(
            float(tropo.longitude_rad) if tropo is not None else 0.0
        ),
        obs_site_height_m=float(tropo.height_m) if tropo is not None else 0.0,
        obs_site_pressure_mbar=(
            float(tropo.pressure_mbar) if tropo is not None else 101.325
        ),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
        bclt_max_iter=bclt_max_iter,
    )


def build_staged_delta_pack(
    setup: "GeneralFitSetup",
) -> NativeDeltaPack | None:
    """Build host-staged cache for tempo2-native residual deltas (default path)."""
    from jug.residuals.tempo2.model import tempo2_einstein_rate_host

    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    frozen = host_frozen_vectors_from_tempo2_obs_state(td)
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    sat_int, sat_sec = sat_daysec_numpy_from_td_and_toas(td, toas)
    _f_terms, _pepoch, pep_int, pep_frac = spin_params_to_jax(params)
    tt_pre = np.asarray(
        td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
        dtype=np.float64,
    )
    site_mjd = sat + tt_pre / SECS_PER_DAY
    einstein = tempo2_einstein_rate_host(site_mjd, params)
    tropo = np.asarray(td.get("tropo_delay_sec", 0.0), dtype=np.float64)
    if tropo.ndim == 0:
        tropo = np.full(len(sat), float(tropo), dtype=np.float64)
    units = parse_timescale(params)
    jump = getattr(setup, "jump_phase", None)
    tzr = getattr(setup, "tzr_phase", None)
    planet_obs_ls = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in frozen["planet_obs_ls"].items()
    }
    bclt_max_iter = _bclt_max_iter_from_setup(setup)
    return NativeDeltaPack(
        mode=TEMPO2_GRAPH_STAGED_BCLT,
        sat_mjd=jnp.asarray(sat, dtype=jnp.float64),
        sat_int_day=jnp.asarray(sat_int, dtype=jnp.float64),
        sat_sec_in_day=jnp.asarray(sat_sec, dtype=jnp.float64),
        pep_int=pep_int,
        pep_frac=pep_frac,
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(static["dt_sec"], dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(frozen["earth_ssb_km"], dtype=jnp.float64),
        observatory_earth_km=jnp.asarray(frozen["observatory_earth_km"], dtype=jnp.float64),
        site_vel_km_s=jnp.asarray(frozen["site_vel_km_s"], dtype=jnp.float64),
        ssb_obs_ls=jnp.asarray(frozen["ssb_obs_ls"], dtype=jnp.float64),
        obs_sun_ls=jnp.asarray(frozen["obs_sun_ls"], dtype=jnp.float64),
        obs_jupiter_ls=jnp.asarray(frozen["obs_jupiter_ls"], dtype=jnp.float64),
        planet_obs_ls=planet_obs_ls,
        correction_tt_sec=jnp.asarray(tt_pre, dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        einstein_rate=jnp.asarray(einstein, dtype=jnp.float64),
        tropo_sec=jnp.asarray(tropo, dtype=jnp.float64),
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
        bclt_max_iter=bclt_max_iter,
    )


def _resolve_dt_ssb_ref_sec(
    td: dict,
    *,
    params: dict,
    sat_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    dt_emission_sec: np.ndarray,
    tropo_sec: np.ndarray,
    frozen: dict,
    model_static: Tempo2ModelStatic,
    ne_sw: float,
    use_native_ecliptic: bool,
    bclt_max_iter: int | None = None,
) -> np.ndarray:
    """Resolve reference BCLT ``dt_ssb`` for fixed-state nonlinear mode."""
    native_terms = td.get("tempo2_terms") or {}
    for candidate in (
        td.get("bclt_dt_ssb_sec"),
        td.get("dt_ssb_sec"),
        native_terms.get("dt_ssb_sec"),
    ):
        if candidate is not None:
            return np.asarray(candidate, dtype=np.float64)
    tt_pre = np.asarray(
        td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
        dtype=np.float64,
    )
    ref_terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=np.asarray(freq_mhz, dtype=np.float64),
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
        tropo_sec=np.asarray(tropo_sec, dtype=np.float64),
        ssb_obs_ls=frozen["ssb_obs_ls"],
        obs_sun_ls=frozen["obs_sun_ls"],
        obs_jupiter_ls=frozen["obs_jupiter_ls"],
        obs_planets_ls=frozen["planet_obs_ls"],
        earth_ssb_km=frozen["earth_ssb_km"],
        observatory_earth_km=frozen["observatory_earth_km"][:, :3],
        site_vel_km_s=frozen["site_vel_km_s"],
        correction_tt_sec_pre=tt_pre,
        correction_tt_tb_sec=np.asarray(td["correction_tt_tb_sec"], dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        use_native_ecliptic=use_native_ecliptic,
        bclt_max_iter=bclt_max_iter,
    )
    return np.asarray(ref_terms.dt_ssb_sec, dtype=np.float64)


def _build_fixed_state_pack_from_host(
    *,
    jug_result: dict,
    params: dict,
    toas: list[Any],
    model_static: Tempo2ModelStatic,
    frozen: dict,
    tropo: np.ndarray,
    dt_ssb_ref_sec: np.ndarray,
    bclt_max_iter: int | None = None,
) -> NativeDeltaPack:
    """Assemble a fixed-state pack from host residual-cache inputs."""
    from jug.residuals.tempo2.model import tempo2_einstein_rate_host

    td = jug_result["term_diagnostics"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    sat_int, sat_sec = sat_daysec_numpy_from_td_and_toas(td, toas)
    _f_terms, _pepoch, pep_int, pep_frac = spin_params_to_jax(params)
    tt_pre = np.asarray(
        td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
        dtype=np.float64,
    )
    site_mjd = sat + tt_pre / SECS_PER_DAY
    einstein = tempo2_einstein_rate_host(site_mjd, params)
    units = parse_timescale(params)
    jump = model_static.jump_phase
    tzr = model_static.tzr_phase
    planet_obs_ls = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in frozen["planet_obs_ls"].items()
    }
    return NativeDeltaPack(
        mode=TEMPO2_GRAPH_FIXED_STATE_NONLINEAR,
        sat_mjd=jnp.asarray(sat, dtype=jnp.float64),
        sat_int_day=jnp.asarray(sat_int, dtype=jnp.float64),
        sat_sec_in_day=jnp.asarray(sat_sec, dtype=jnp.float64),
        pep_int=pep_int,
        pep_frac=pep_frac,
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(jug_result["dt_sec"], dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(frozen["earth_ssb_km"], dtype=jnp.float64),
        observatory_earth_km=jnp.asarray(frozen["observatory_earth_km"], dtype=jnp.float64),
        site_vel_km_s=jnp.asarray(frozen["site_vel_km_s"], dtype=jnp.float64),
        ssb_obs_ls=jnp.asarray(frozen["ssb_obs_ls"], dtype=jnp.float64),
        obs_sun_ls=jnp.asarray(frozen["obs_sun_ls"], dtype=jnp.float64),
        obs_jupiter_ls=jnp.asarray(frozen["obs_jupiter_ls"], dtype=jnp.float64),
        planet_obs_ls=planet_obs_ls,
        correction_tt_sec=jnp.asarray(tt_pre, dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        einstein_rate=jnp.asarray(einstein, dtype=jnp.float64),
        tropo_sec=jnp.asarray(tropo, dtype=jnp.float64),
        dt_ssb_ref_sec=jnp.asarray(dt_ssb_ref_sec, dtype=np.float64),
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
        bclt_max_iter=bclt_max_iter,
    )


def _native_fixed_state_terms_from_pack(
    params: dict,
    pack: NativeDeltaPack,
) -> tuple[Tempo2Terms, jnp.ndarray]:
    pos, vel, acc = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms, pepoch, pep_int, pep_frac = spin_params_to_jax(params)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    terms, residual_sec = compute_tempo2_toa_model_fixed_state_nonlinear_jax(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=jnp.asarray(pos, dtype=jnp.float64),
        vel_pulsar=jnp.asarray(vel, dtype=jnp.float64),
        acc_pulsar=jnp.asarray(acc, dtype=jnp.float64),
        tropo_sec=pack.tropo_sec,
        dt_emission_sec=pack.dt_emission_sec,
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
    )
    return terms, residual_sec


def build_fixed_state_nonlinear_delta_pack(
    setup: "GeneralFitSetup",
) -> NativeDeltaPack | None:
    """Build fixed-state nonlinear cache for tempo2-native residual deltas."""
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    frozen = host_frozen_vectors_from_tempo2_obs_state(td)
    tropo = np.asarray(td.get("tropo_delay_sec", 0.0), dtype=np.float64)
    if tropo.ndim == 0:
        tropo = np.full(len(td["sat_mjd"]), float(tropo), dtype=np.float64)
    dt_ssb_ref = _resolve_dt_ssb_ref_sec(
        td,
        params=params,
        sat_mjd=np.asarray(td["sat_mjd"], dtype=np.float64),
        freq_mhz=np.array([t.freq_mhz for t in toas], dtype=np.float64),
        dt_emission_sec=np.asarray(static["dt_sec"], dtype=np.float64),
        tropo_sec=tropo,
        frozen=frozen,
        model_static=model_static,
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        bclt_max_iter=_bclt_max_iter_from_setup(setup),
    )
    return _build_fixed_state_pack_from_host(
        jug_result=jug_result,
        params=params,
        toas=toas,
        model_static=model_static,
        frozen=frozen,
        tropo=tropo,
        dt_ssb_ref_sec=dt_ssb_ref,
        bclt_max_iter=_bclt_max_iter_from_setup(setup),
    )


def build_delta_pack_for_setup(
    setup: "GeneralFitSetup",
) -> NativeDeltaPack | NativeDeltaPack | NativeDeltaPack | None:
    """Select native delta pack from ``setup.tempo2_native``."""
    config = getattr(setup, "tempo2_native", None)
    mode = _chain_mode(config)
    if mode == TEMPO2_GRAPH_FULL:
        return build_delta_pack(setup)
    if mode == TEMPO2_GRAPH_FIXED_STATE_NONLINEAR:
        return build_fixed_state_nonlinear_delta_pack(setup)
    if mode == TEMPO2_GRAPH_STAGED_BCLT:
        return build_staged_delta_pack(setup)
    raise ValueError(f"Unknown tempo2 native graph mode: {mode!r}")
