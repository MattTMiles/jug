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
from .static import Tempo2ModelStatic, compute_dm_vals_jax, planet_rsa_tuple_jax_from_dict
from .tail import _tempo2_residual_tail_jax

@partial(
    jax.jit,
    static_argnames=(
        "ne_sw",
        "posepoch_mjd",
        "pmrv_rad_century",
        "dilate_freq",
        "si_units",
        "units_tdb",
        "planet_shapiro_enabled",
        "track_val",
        "subtract_mean",
        "dshk",
        "shk_posepoch",
        "dm_epoch",
        "dm_coeffs",
    ),
)
def compute_tempo2_toa_model_staging_with_host_inputs_jax(
    *,
    sat_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    params_f_terms: jnp.ndarray,
    params_pepoch: jnp.float64,
    pos_pulsar: jnp.ndarray,
    vel_pulsar: jnp.ndarray,
    acc_pulsar: jnp.ndarray,
    tropo_sec: jnp.ndarray,
    dt_emission_sec: jnp.ndarray,
    earth_ssb_km: jnp.ndarray,
    observatory_earth_km: jnp.ndarray,
    site_vel_km_s: jnp.ndarray,
    ssb_obs_ls: jnp.ndarray,
    obs_sun_ls: jnp.ndarray,
    obs_jupiter_ls: jnp.ndarray,
    planet_obs_ls: dict[str, jnp.ndarray] | None = None,
    correction_tt_sec_pre: jnp.ndarray,
    correction_tt_tb_sec_pre: jnp.ndarray | None = None,
    einstein_rate: jnp.ndarray | None = None,
    ifte_records: jnp.ndarray | None = None,
    ifte_start_jd: jnp.ndarray | None = None,
    ifte_end_jd: jnp.ndarray | None = None,
    ifte_step_jd: jnp.ndarray | None = None,
    ifte_coef_offset: int | None = None,
    ifte_ncf: int | None = None,
    ifte_na: int | None = None,
    dm_vals: jnp.ndarray | None = None,
    dm_epoch: float = 0.0,
    dm_coeffs: tuple[float, ...] = (0.0,),
    ne_sw: float = 0.0,
    posepoch_mjd: float = 0.0,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
    dilate_freq: bool = False,
    si_units: bool = False,
    units_tdb: bool = True,
    planet_shapiro_enabled: bool = True,
    track_val: int = -2,
    subtract_mean: bool = True,
    dshk: float = 0.0,
    pmra: float = 0.0,
    pmdec: float = 0.0,
    shk_posepoch: float | None = None,
    jump_phase: jnp.ndarray | None = None,
    tzr_phase: jnp.float64 | None = None,
    pulse_numbers: jnp.ndarray | None = None,
    pn_add: jnp.ndarray | None = None,
    sat_int_day: jnp.ndarray | None = None,
    sat_sec_in_day: jnp.ndarray | None = None,
    pep_int: jnp.ndarray | None = None,
    pep_frac: jnp.ndarray | None = None,
) -> tuple[Tempo2NativeTerms, jnp.ndarray]:
    """Production Tempo2 TOA model with host-frozen static inputs.

    Accepts precomputed geometry, clocks, and ``einsteinRate`` from
    ``term_diagnostics['tempo2_obs_state']``. Only the parameter-dependent tail
    (BCLT, formBats, Shklovskii, spin) runs inside JAX. This is the **default**
    production path when ``JUG_TEMPO2_NATIVE_GRAPH_MODE=staged_bclt`` (default).

    For the slow unified in-graph reference, see ``compute_tempo2_toa_model_jax``.
    """
    if dm_vals is None:
        dm_vals = compute_dm_vals_jax(sat_mjd, dm_epoch=dm_epoch, dm_coeffs=dm_coeffs)
    tt = jnp.asarray(correction_tt_sec_pre, dtype=jnp.float64)
    mjd_tt = sat_mjd + tt / SECS_PER_DAY
    if correction_tt_tb_sec_pre is None:
        if ifte_records is None:
            raise ValueError("frozen staging helper requires IFTE tables or tt_tb pre")
        tt_tb, _teph = compute_tempo2_correction_tt_tb_jax(
            mjd_tt,
            observatory_earth_km,
            earth_ssb_km[:, 3:6],
            ifte_records=ifte_records,
            ifte_start_jd=ifte_start_jd,
            ifte_end_jd=ifte_end_jd,
            ifte_step_jd=ifte_step_jd,
            ifte_coef_offset=ifte_coef_offset,
            ifte_ncf=ifte_ncf,
            ifte_na=ifte_na,
            units_tdb=units_tdb,
            si_units=si_units,
        )
    else:
        tt_tb = jnp.asarray(correction_tt_tb_sec_pre, dtype=jnp.float64)
    if einstein_rate is None:
        raise ValueError("frozen staging helper requires precomputed einstein_rate")
    einstein = jnp.asarray(einstein_rate, dtype=jnp.float64)
    if planet_obs_ls is None:
        planet_obs_ls = {"jupiter": obs_jupiter_ls}
    planet_rsa = planet_rsa_tuple_jax_from_dict(
        planet_obs_ls,
        n_toa=int(sat_mjd.shape[0]),
        obs_jupiter_ls=obs_jupiter_ls,
    )
    bclt = compute_bclt_terms_jax(
        sat_mjd=sat_mjd,
        correction_tt_sec=tt,
        correction_tt_tb_sec=tt_tb,
        ssb_obs_ls=ssb_obs_ls,
        obs_sun_ls=obs_sun_ls,
        freq_mhz=freq_mhz,
        earth_ssb_vel_km_s=earth_ssb_km[:, 3:6],
        site_vel_km_s=site_vel_km_s,
        dm_vals=dm_vals,
        pos_pulsar=pos_pulsar,
        vel_pulsar=vel_pulsar,
        acc_pulsar=acc_pulsar,
        posepoch_mjd=posepoch_mjd,
        parallax_mas=parallax_mas,
        pmrv_rad_century=pmrv_rad_century,
        ne_sw=ne_sw,
        einstein_rate=einstein,
        dilate_freq=dilate_freq,
        planet_shapiro_enabled=planet_shapiro_enabled,
        obs_jupiter_ls=obs_jupiter_ls,
        planet_obs_ls=planet_rsa,
    )
    tropo = jnp.asarray(tropo_sec, dtype=jnp.float64)
    return _tempo2_residual_tail_jax(
        bclt=bclt,
        sat_mjd=sat_mjd,
        tt=tt,
        tt_tb=tt_tb,
        tropo=tropo,
        dt_emission_sec=dt_emission_sec,
        params_f_terms=params_f_terms,
        params_pepoch=params_pepoch,
        planet_shapiro_enabled=planet_shapiro_enabled,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
        shk_posepoch_mjd=shk_posepoch,
        track_val=track_val,
        subtract_mean=subtract_mean,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        sat_int_day=sat_int_day,
        sat_sec_in_day=sat_sec_in_day,
        pep_int=pep_int,
        pep_frac=pep_frac,
    )
