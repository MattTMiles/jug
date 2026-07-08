"""Tempo2 JAX model submodule."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tropo_jax import (
    TropoObsPacked,
    compute_tempo2_zenith_gcrs_jax,
    pack_tropo_obs_static,
    tempo2_source_elevation_rad_jax,
    tempo2_tropo_delay_jax,
)
from jug.delays.tempo2_geometry import (
    build_tempo2_pulsar_vectors,
    pmrv_rad_per_century,
)
from jug.delays.tempo2_geometry_jax import (
    _stack_planet_obs_ls_jax,
    bootstrap_tempo2_geometry_jax,
)
from jug.delays.tempo2_spk_jax import Tempo2SpkPacked, SpkSegmentPacked, pack_tempo2_spk_jax
from jug.delays.tempo2_site_jax import IersEopPacked, pack_iers_eop_jax
from jug.delays.tempo2_geometry import tempo2_dilate_freq_enabled
from jug.residuals.tempo2.calculate_bclt_jax import (
    compute_bclt_terms_fixed_state_jax,
    compute_bclt_terms_jax,
)
from jug.residuals.tempo2.clock_jax import (
    compute_einstein_rate_jax,
    compute_tempo2_correction_tt_tb_jax,
    compute_tempo2_get_correction_tt_jax,
)
from jug.residuals.tempo2.compensated import (
    mjd_view_from_daysec,
    split_mjd_to_daysec,
)
from jug.residuals.tempo2.formbats_jax import (
    compute_formbats_daysec,
    compute_shklovskii_sec_jax_pure_daysec,
    compute_torb_closure_daysec,
)
from jug.residuals.tempo2.probes import compute_formbats_effective_shapiro_sec
from jug.residuals.tempo2.spin_jax import (
    compute_tempo2_phase5_daysec,
    pepoch_parts_from_value,
    spin_params_to_jax,
    track_minus2_frac_phase_jax,
)
from jug.residuals.tempo2.types import Tempo2NativeTerms
from jug.utils.constants import SECS_PER_DAY
from jug.utils.timescales import is_tempo2_si_units, parse_timescale
def _tempo2_residual_tail_jax(
    *,
    bclt,
    sat_mjd: jnp.ndarray,
    tt: jnp.ndarray,
    tt_tb: jnp.ndarray,
    tropo: jnp.ndarray,
    dt_emission_sec: jnp.ndarray,
    params_f_terms: jnp.ndarray,
    params_pepoch: jnp.float64,
    planet_shapiro_enabled: bool,
    dshk: float,
    pmra,
    pmdec,
    shk_posepoch_mjd,
    track_val: int,
    subtract_mean: bool,
    jump_phase: jnp.ndarray | None,
    tzr_phase: jnp.float64 | None,
    pulse_numbers: jnp.ndarray | None,
    pn_add: jnp.ndarray | None,
    sat_int_day: jnp.ndarray | None = None,
    sat_sec_in_day: jnp.ndarray | None = None,
    pep_int: jnp.ndarray | None = None,
    pep_frac: jnp.ndarray | None = None,
) -> tuple[Tempo2NativeTerms, jnp.ndarray]:
    """Shared formBats → Shklovskii → spin tail for all tempo2 TOA model modes."""
    if sat_int_day is None or sat_sec_in_day is None:
        sat_int_day, sat_sec_in_day = split_mjd_to_daysec(sat_mjd)
    if pep_int is None or pep_frac is None:
        _, pep_int, pep_frac = pepoch_parts_from_value(params_pepoch)
    shk_posepoch = float(shk_posepoch_mjd if shk_posepoch_mjd is not None else params_pepoch)
    shk_pep_int = jnp.floor(jnp.asarray(shk_posepoch, dtype=jnp.float64))
    shk_pep_frac = jnp.asarray(shk_posepoch, dtype=jnp.float64) - shk_pep_int

    shap_delay = bclt.shapiro_sun_sec + jnp.where(
        planet_shapiro_enabled,
        bclt.shapiro_planets_sec,
        0.0,
    )
    (
        bat_corr_day,
        bat_corr_resid,
        bat_int,
        bat_sec,
        _bbat_int,
        _bbat_sec,
    ) = compute_formbats_daysec(
        sat_int_day,
        sat_sec_in_day,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        jnp.zeros_like(sat_mjd),
    )
    shk = compute_shklovskii_sec_jax_pure_daysec(
        bat_int,
        bat_sec,
        shk_pep_int,
        shk_pep_frac,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
    )
    (
        bat_corr_day,
        bat_corr_resid,
        bat_int,
        bat_sec,
        bbat_int,
        bbat_sec,
    ) = compute_formbats_daysec(
        sat_int_day,
        sat_sec_in_day,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        shk,
    )
    bat_mjd = mjd_view_from_daysec(bat_int, bat_sec)
    bbat_mjd = mjd_view_from_daysec(bbat_int, bbat_sec)
    dt_emit = jnp.asarray(dt_emission_sec, dtype=jnp.float64)
    torb = compute_torb_closure_daysec(bbat_int, bbat_sec, dt_emit, pep_int, pep_frac)
    terms = Tempo2NativeTerms(
        sat_mjd=sat_mjd,
        sat_int_day=sat_int_day,
        sat_sec_in_day=sat_sec_in_day,
        correction_tt_sec=tt,
        correction_tt_tb_sec=tt_tb,
        roemer_sec=bclt.roemer_sec,
        tdis1_sec=bclt.tdis1_sec,
        tdis2_sec=bclt.tdis2_sec,
        shapiro_sun_sec=bclt.shapiro_sun_sec,
        shapiro_planets_sec=bclt.shapiro_planets_sec,
        shapiro_delay_sec=shap_delay,
        tropospheric_sec=tropo,
        prebinary_sec=jnp.zeros_like(sat_mjd),
        bat_corr_day=bat_corr_day,
        bat_corr_day_residual=bat_corr_resid,
        bat_mjd=bat_mjd,
        bbat_mjd=bbat_mjd,
        bat_int_day=bat_int,
        bat_sec_in_day=bat_sec,
        bbat_int_day=bbat_int,
        bbat_sec_in_day=bbat_sec,
        shklovskii_sec=shk,
        torb_sec=torb,
        dt_emission_sec=dt_emit,
        dt_ssb_sec=bclt.dt_ssb_sec,
        bclt_iterations=bclt.bclt_iterations,
        converged=bclt.converged,
    )
    if track_val == -2 and pulse_numbers is not None and pn_add is not None:
        phase5 = compute_tempo2_phase5_daysec(
            bbat_int,
            bbat_sec,
            torb,
            params_f_terms,
            pep_int,
            pep_frac,
            jump_phase=jump_phase,
            tzr_phase=tzr_phase,
        )
        frac, _pulse = track_minus2_frac_phase_jax(
            phase5,
            bbat_int,
            params_f_terms[0],
            pulse_numbers,
            pn_add,
        )
    else:
        phase5 = compute_tempo2_phase5_daysec(
            bbat_int,
            bbat_sec,
            torb,
            params_f_terms,
            pep_int,
            pep_frac,
            jump_phase=jump_phase,
            tzr_phase=tzr_phase,
        )
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / params_f_terms[0]
    if subtract_mean:
        residual_sec = residual_sec - jnp.mean(residual_sec)
    return terms, residual_sec
