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
    TEMPO2_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_GRAPH_FULL,
    TEMPO2_GRAPH_STAGED_BCLT,
    tempo2_graph_mode,
)
from jug.residuals.tempo2.spin_jax import spin_params_to_jax
from jug.residuals.tempo2.types import Tempo2Terms
from jug.utils.timescales import is_tempo2_si_units, parse_timescale
from .common import NativeDeltaPack, _load_model_static_for_native_chain, _chain_mode, sat_daysec_numpy_from_td_and_toas
from .terms import compute_tempo2_terms_jax

def prepare_tempo2_chain_from_simple_result(
    jug_result: dict,
    params: dict,
    toas: list[Any],
) -> Tempo2Terms:
    """Build native terms for dev_oracle / diagnostics.

    Dispatches by ``jug_result['tempo2_native']`` graph mode:
    ``full``, ``fixed_state_nonlinear``, or ``staged_bclt`` (default).
    """
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    td = jug_result["term_diagnostics"]
    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    ne_sw = resolve_ne_sw_cm3(params, profile)
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    model_static = _load_model_static_for_native_chain(params, toas, jug_result)
    graph_mode = _chain_mode(jug_result.get("tempo2_native"))
    from jug.residuals.tempo2.calculate_bclt_jax import bclt_jax_fixed_iter_count
    from jug.timing import resolve_tempo2_jug_options

    bclt_opts = resolve_tempo2_jug_options(jug_result.get("tempo2_jug_options"))
    bclt_max_iter = bclt_jax_fixed_iter_count(bclt_opts.get("bclt_fixed_iter"))

    if graph_mode == TEMPO2_GRAPH_FULL:
        return compute_tempo2_terms_jax(
            sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
            correction_tt_sec=jnp.asarray(
                td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
                dtype=jnp.float64,
            ),
            correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
            params=params,
            toas=toas,
            observatory_earth_km=jnp.zeros((len(toas), 3), dtype=jnp.float64),
            earth_ssb_km=jnp.zeros((len(toas), 3), dtype=jnp.float64),
            earth_ssb_vel_km_s=jnp.zeros((len(toas), 3), dtype=jnp.float64),
            ephem_path=model_static.ephem_path,
            freq_mhz=jnp.asarray(
                jug_result.get("freq_bary_mhz", td.get("freq_bary_mhz", [])),
                dtype=jnp.float64,
            ),
            tdis1_sec=jnp.asarray(td["dm_delay_sec"], dtype=jnp.float64),
            tdis2_sec=jnp.asarray(td["sw_delay_sec"], dtype=np.float64),
            tropospheric_sec=jnp.asarray(td.get("tropo_delay_sec", 0.0), dtype=jnp.float64),
            dt_emission_sec=jnp.asarray(jug_result["dt_sec"], dtype=np.float64),
            use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
            freq_mhz_topocentric=jnp.asarray(freq_topo, dtype=jnp.float64),
            ne_sw=ne_sw,
            model_static=model_static,
        )

    frozen = host_frozen_vectors_from_tempo2_obs_state(td)
    tropo = np.asarray(td.get("tropo_delay_sec", 0.0), dtype=np.float64)
    if tropo.ndim == 0:
        tropo = np.full(len(td["sat_mjd"]), float(tropo), dtype=np.float64)
    sat_int, sat_sec = sat_daysec_numpy_from_td_and_toas(td, toas)

    mode = graph_mode
    if mode == TEMPO2_GRAPH_FIXED_STATE_NONLINEAR:
        dt_ssb_ref = _resolve_dt_ssb_ref_sec(
            td,
            params=params,
            sat_mjd=np.asarray(td["sat_mjd"], dtype=np.float64),
            freq_mhz=freq_topo,
            dt_emission_sec=np.asarray(jug_result["dt_sec"], dtype=np.float64),
            tropo_sec=tropo,
            frozen=frozen,
            model_static=model_static,
            ne_sw=float(ne_sw),
            use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        )
        pack = _build_fixed_state_pack_from_host(
            jug_result=jug_result,
            params=params,
            toas=toas,
            model_static=model_static,
            frozen=frozen,
            tropo=tropo,
            dt_ssb_ref_sec=dt_ssb_ref,
        )
        terms, _ = _native_fixed_state_terms_from_pack(params, pack)
        return terms

    terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(td["sat_mjd"], dtype=np.float64),
        freq_mhz=freq_topo,
        dt_emission_sec=np.asarray(jug_result["dt_sec"], dtype=np.float64),
        tropo_sec=tropo,
        ssb_obs_ls=frozen["ssb_obs_ls"],
        obs_sun_ls=frozen["obs_sun_ls"],
        obs_jupiter_ls=frozen["obs_jupiter_ls"],
        obs_planets_ls=frozen["planet_obs_ls"],
        earth_ssb_km=frozen["earth_ssb_km"],
        observatory_earth_km=frozen["observatory_earth_km"][:, :3],
        site_vel_km_s=frozen["site_vel_km_s"],
        correction_tt_sec_pre=np.asarray(
            td.get("formbats_correction_tt_sec", td["correction_tt_sec"]),
            dtype=np.float64,
        ),
        correction_tt_tb_sec=np.asarray(td["correction_tt_tb_sec"], dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        sat_int_day=sat_int,
        sat_sec_in_day=sat_sec,
        bclt_max_iter=bclt_max_iter,
    )
    return terms

