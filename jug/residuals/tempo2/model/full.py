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
from jug.residuals.tempo2.types import Tempo2Terms
from jug.utils.constants import SECS_PER_DAY
from jug.utils.timescales import is_tempo2_si_units, parse_timescale
from .static import (
    Tempo2ModelStatic,
    _dm_coeffs_from_params,
    _eop_to_jax,
    _spk_to_jax,
    build_tempo2_model_static,
    compute_dm_vals_jax,
    tempo2_einstein_rate_host,
)
from .staged import compute_tempo2_toa_model_staging_with_host_inputs_jax
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
        "ifte_coef_offset",
        "ifte_ncf",
        "ifte_na",
        "bootstrap_max_iter",
        "bclt_max_iter",
        "correct_troposphere",
        "obs_site_latitude_rad",
        "obs_site_longitude_rad",
        "obs_site_height_m",
        "obs_site_pressure_mbar",
        "ecl_obl_rad",
    ),
)
def compute_tempo2_toa_model_jax(
    *,
    sat_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    params_f_terms: jnp.ndarray,
    params_pepoch: jnp.float64,
    pos_pulsar: jnp.ndarray,
    vel_pulsar: jnp.ndarray,
    acc_pulsar: jnp.ndarray,
    obs_itrf_km: jnp.ndarray,
    spk_packed: Tempo2SpkPacked,
    eop_packed: IersEopPacked,
    dm_vals: jnp.ndarray | None = None,
    dm_epoch: float = 0.0,
    dm_coeffs: tuple[float, ...] = (0.0,),
    dt_emission_sec: jnp.ndarray,
    chain_mjd_tables: tuple[jnp.ndarray, ...],
    chain_offset_tables: tuple[jnp.ndarray, ...],
    bipm_mjd: jnp.ndarray,
    bipm_offset: jnp.ndarray,
    ifte_records: jnp.ndarray,
    ifte_start_jd: jnp.ndarray,
    ifte_end_jd: jnp.ndarray,
    ifte_step_jd: jnp.ndarray,
    ifte_coef_offset: int,
    ifte_ncf: int,
    ifte_na: int,
    ne_sw: float,
    obs_site_latitude_rad: float = 0.0,
    obs_site_longitude_rad: float = 0.0,
    obs_site_height_m: float = 0.0,
    obs_site_pressure_mbar: float = 101.325,
    posepoch_mjd: float,
    parallax_mas: jnp.ndarray | float = 0.0,
    pmrv_rad_century: float = 0.0,
    dilate_freq: bool = False,
    si_units: bool = False,
    units_tdb: bool = True,
    planet_shapiro_enabled: bool = True,
    track_val: int = -2,
    subtract_mean: bool = True,
    dshk: float = 0.0,
    pmra: jnp.ndarray | float = 0.0,
    pmdec: jnp.ndarray | float = 0.0,
    shk_posepoch: float | None = None,
    jump_phase: jnp.ndarray | None = None,
    tzr_phase: jnp.float64 | None = None,
    pulse_numbers: jnp.ndarray | None = None,
    pn_add: jnp.ndarray | None = None,
    bootstrap_max_iter: int = 8,
    bclt_max_iter: int | None = None,
    correct_troposphere: bool = False,
    ecl_obl_rad: float = 0.0,
    sat_int_day: jnp.ndarray | None = None,
    sat_sec_in_day: jnp.ndarray | None = None,
    pep_int: jnp.ndarray | None = None,
    pep_frac: jnp.ndarray | None = None,
) -> tuple[Tempo2Terms, jnp.ndarray]:
    """Full Tempo2 delay/spin chain in one JIT graph.

    .. warning::
        **Extremely slow first compile.** This function evaluates clocks, SPK
        ephemeris, EOP site motion, IFTE bootstrap, troposphere, BCLT, formBats,
        and spin inside a single ``@jax.jit`` boundary. On wsrt167 (167 TOAs) the
        initial compile can take **minutes**. Production fitting and fast dev loops
        should use ``compute_tempo2_toa_model_staging_with_host_inputs_jax`` with
        host-frozen inputs instead. Enable only via
        ``tempo2_native="full"``.

    Clock ``getCorrectionTT``, IFTE ``IF_deltaT``, ephemeris geometry
    (SPK + site motion + Teph bootstrap), troposphere, and ``einsteinRate``
    run inside the JIT graph.
    """
    if dm_vals is None:
        dm_vals = compute_dm_vals_jax(sat_mjd, dm_epoch=dm_epoch, dm_coeffs=dm_coeffs)
    tt = compute_tempo2_get_correction_tt_jax(
        sat_mjd,
        chain_mjd_tables=chain_mjd_tables,
        chain_offset_tables=chain_offset_tables,
        bipm_mjd=bipm_mjd,
        bipm_offset=bipm_offset,
    )
    mjd_tt = sat_mjd + tt / SECS_PER_DAY
    if dilate_freq:
        einstein = compute_einstein_rate_jax(mjd_tt, si_units=si_units)
    else:
        einstein = jnp.ones_like(sat_mjd, dtype=jnp.float64)
    if correct_troposphere:
        site = TropoObsPacked(
            latitude_rad=obs_site_latitude_rad,
            longitude_rad=obs_site_longitude_rad,
            height_m=obs_site_height_m,
            pressure_mbar=obs_site_pressure_mbar,
        )
        zenith_gcrs = compute_tempo2_zenith_gcrs_jax(sat_mjd, tt, site)
        # tropo.C uses posPulsarEquatorial; undo the ecliptic frame if needed.
        if ecl_obl_rad != 0.0:
            ce = jnp.cos(ecl_obl_rad)
            se = jnp.sin(ecl_obl_rad)
            pos_pulsar_equ = jnp.stack(
                [
                    pos_pulsar[0],
                    ce * pos_pulsar[1] - se * pos_pulsar[2],
                    se * pos_pulsar[1] + ce * pos_pulsar[2],
                ]
            )
        else:
            pos_pulsar_equ = pos_pulsar
        elevation_rad = tempo2_source_elevation_rad_jax(
            zenith_gcrs,
            pos_pulsar_equ,
            obs_site_height_m,
        )
        tropo = tempo2_tropo_delay_jax(sat_mjd, tt, elevation_rad, site)
    else:
        tropo = jnp.zeros_like(sat_mjd, dtype=jnp.float64)
    tt_tb, geom = bootstrap_tempo2_geometry_jax(
        sat_mjd,
        tt,
        obs_itrf_km=obs_itrf_km,
        spk=spk_packed,
        eop=eop_packed,
        ifte_records=ifte_records,
        ifte_start_jd=ifte_start_jd,
        ifte_end_jd=ifte_end_jd,
        ifte_step_jd=ifte_step_jd,
        ifte_coef_offset=ifte_coef_offset,
        ifte_ncf=ifte_ncf,
        ifte_na=ifte_na,
        si_units=si_units,
        units_tdb=units_tdb,
        max_iter=bootstrap_max_iter,
        ecl_obl_rad=ecl_obl_rad,
    )
    bclt = compute_bclt_terms_jax(
        sat_mjd=sat_mjd,
        correction_tt_sec=tt,
        correction_tt_tb_sec=tt_tb,
        ssb_obs_ls=geom.ssb_obs_ls,
        obs_sun_ls=geom.obs_sun_ls,
        freq_mhz=freq_mhz,
        earth_ssb_vel_km_s=geom.earth_ssb_km[:, 3:6],
        site_vel_km_s=geom.site_vel_km_s,
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
        obs_jupiter_ls=geom.obs_jupiter_ls,
        planet_obs_ls=_stack_planet_obs_ls_jax(geom),
        max_iter=bclt_max_iter,
    )
    tropo = jnp.asarray(tropo, dtype=jnp.float64)
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

def run_tempo2_toa_model(
    *,
    params: dict,
    sat_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    ephem_mjd: np.ndarray,
    static: Tempo2ModelStatic,
) -> tuple[Tempo2Terms, np.ndarray]:
    """Host wrapper: run unified JIT model with prepacked static tables."""
    del ephem_mjd
    terms, res = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=np.asarray(freq_mhz, dtype=np.float64),
        dt_emission_sec=np.asarray(static.dt_emission_sec, dtype=np.float64),
        model_static=static,
        ne_sw=float(static.ne_sw),
        planet_shapiro_enabled=bool(static.planet_shapiro_enabled),
        use_native_ecliptic=bool(static.use_native_ecliptic),
        track_val=int(static.track_val),
        subtract_mean=bool(static.subtract_mean),
        pulse_numbers=static.pulse_numbers,
        pn_add=static.pn_add,
        jump_phase=static.jump_phase,
        tzr_phase=static.tzr_phase,
        compute_residuals=True,
    )
    return terms, jax.device_get(res)


def run_tempo2_toa_model_with_fixed_ifte_geometry(
    *,
    params: dict,
    sat_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    dt_emission_sec: np.ndarray,
    tropo_sec: np.ndarray | None = None,
    ssb_obs_ls: np.ndarray | None = None,
    obs_sun_ls: np.ndarray | None = None,
    obs_jupiter_ls: np.ndarray | None = None,
    obs_planets_ls: dict[str, np.ndarray] | None = None,
    earth_ssb_km: np.ndarray | None = None,
    observatory_earth_km: np.ndarray | None = None,
    site_vel_km_s: np.ndarray | None = None,
    earth_ssb_vel_km_s: np.ndarray | None = None,
    correction_tt_sec_pre: np.ndarray | None = None,
    correction_tt_tb_sec: np.ndarray | None = None,
    model_static: Tempo2ModelStatic | None = None,
    ne_sw: float = 0.0,
    planet_shapiro_enabled: bool = True,
    use_native_ecliptic: bool | None = None,
    track_val: int = -2,
    subtract_mean: bool = True,
    pulse_numbers: np.ndarray | None = None,
    pn_add: np.ndarray | None = None,
    jump_phase: np.ndarray | None = None,
    tzr_phase: float | None = None,
    compute_residuals: bool = False,
    sat_int_day: np.ndarray | None = None,
    sat_sec_in_day: np.ndarray | None = None,
    bclt_max_iter: int | None = None,
) -> tuple[Tempo2Terms, np.ndarray | None]:
    """Run unified or staging JAX model.

    When host geometry arrays are omitted, ephemeris geometry is derived
    in-graph from static SPK/IERS tables (Phase 4 production path).
    """
    if model_static is None:
        raise ValueError(
            "run_tempo2_toa_model_with_fixed_ifte_geometry requires model_static "
            "with clock, IFTE, and SPK tables"
        )
    if use_native_ecliptic is None:
        use_native_ecliptic = bool(params.get("_ecliptic_coords", False))

    pos, vel, acc = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=use_native_ecliptic
    )
    f_terms, pepoch, pep_int, pep_frac = spin_params_to_jax(params)
    dm_epoch = float(params.get("DMEPOCH", params["PEPOCH"]))
    dm_coeffs = _dm_coeffs_from_params(params)
    units = parse_timescale(params)
    dilate = tempo2_dilate_freq_enabled(params)
    pmrv = pmrv_rad_per_century(float(params.get("PMRV", 0.0)))
    common = dict(
        sat_mjd=jnp.asarray(sat_mjd, dtype=jnp.float64),
        freq_mhz=jnp.asarray(freq_mhz, dtype=jnp.float64),
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=jnp.asarray(pos, dtype=jnp.float64),
        vel_pulsar=jnp.asarray(vel, dtype=jnp.float64),
        acc_pulsar=jnp.asarray(acc, dtype=jnp.float64),
        dm_vals=None,
        dm_epoch=dm_epoch,
        dm_coeffs=dm_coeffs,
        dt_emission_sec=jnp.asarray(dt_emission_sec, dtype=jnp.float64),
        ne_sw=float(ne_sw),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        parallax_mas=float(params.get("PX", 0.0)),
        pmrv_rad_century=pmrv,
        dilate_freq=dilate,
        si_units=is_tempo2_si_units(units),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=planet_shapiro_enabled,
        track_val=int(track_val),
        subtract_mean=bool(subtract_mean) if compute_residuals else False,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        pmra=float(params.get("PMRA", 0.0)),
        pmdec=float(params.get("PMDEC", 0.0)),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        jump_phase=(
            None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
        ),
        tzr_phase=None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64),
        pulse_numbers=(
            None if pulse_numbers is None else jnp.asarray(pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=None if pn_add is None else jnp.asarray(pn_add, dtype=jnp.int64),
        sat_int_day=(
            None
            if sat_int_day is None
            else jnp.asarray(sat_int_day, dtype=jnp.float64)
        ),
        sat_sec_in_day=(
            None
            if sat_sec_in_day is None
            else jnp.asarray(sat_sec_in_day, dtype=jnp.float64)
        ),
        pep_int=pep_int,
        pep_frac=pep_frac,
    )
    if ssb_obs_ls is not None or obs_sun_ls is not None:
        if ssb_obs_ls is None or obs_sun_ls is None:
            raise ValueError("staging path requires both ssb_obs_ls and obs_sun_ls")
        earth_ssb = np.asarray(earth_ssb_km, dtype=np.float64)
        if earth_ssb.ndim == 1:
            earth_ssb = np.broadcast_to(earth_ssb, (len(sat_mjd), earth_ssb.shape[0]))
        if earth_ssb.shape[1] == 3:
            earth_vel = (
                np.asarray(earth_ssb_vel_km_s, dtype=np.float64)
                if earth_ssb_vel_km_s is not None
                else np.zeros((len(sat_mjd), 3), dtype=np.float64)
            )
            earth_ssb = np.concatenate([earth_ssb, earth_vel], axis=1)
        jup = (
            np.zeros((len(sat_mjd), 3), dtype=np.float64)
            if obs_jupiter_ls is None
            else np.asarray(obs_jupiter_ls, dtype=np.float64)
        )
        site_vel = (
            np.zeros((len(sat_mjd), 3), dtype=np.float64)
            if site_vel_km_s is None
            else np.asarray(site_vel_km_s, dtype=np.float64)
        )
        if tropo_sec is None:
            raise ValueError("staging path requires host tropo_sec")
        if correction_tt_sec_pre is None:
            tt_pre = np.asarray(
                jax.device_get(
                    compute_tempo2_get_correction_tt_jax(
                        jnp.asarray(sat_mjd, dtype=jnp.float64),
                        chain_mjd_tables=tuple(
                            jnp.asarray(t, dtype=jnp.float64)
                            for t in model_static.chain_mjd_tables
                        ),
                        chain_offset_tables=tuple(
                            jnp.asarray(t, dtype=jnp.float64)
                            for t in model_static.chain_offset_tables
                        ),
                        bipm_mjd=jnp.asarray(model_static.bipm_mjd, dtype=jnp.float64),
                        bipm_offset=jnp.asarray(model_static.bipm_offset, dtype=jnp.float64),
                    )
                ),
                dtype=np.float64,
            )
        else:
            tt_pre = np.asarray(correction_tt_sec_pre, dtype=np.float64)
        site_mjd = np.asarray(sat_mjd, dtype=np.float64) + tt_pre / SECS_PER_DAY
        einstein = tempo2_einstein_rate_host(site_mjd, params)
        planet_obs_ls = None
        if obs_planets_ls is not None:
            planet_obs_ls = {
                k: jnp.asarray(v, dtype=jnp.float64) for k, v in obs_planets_ls.items()
            }
        terms, residual_sec = compute_tempo2_toa_model_staging_with_host_inputs_jax(
            **common,
            tropo_sec=jnp.asarray(tropo_sec, dtype=jnp.float64),
            earth_ssb_km=jnp.asarray(earth_ssb, dtype=jnp.float64),
            observatory_earth_km=jnp.asarray(observatory_earth_km, dtype=jnp.float64),
            site_vel_km_s=jnp.asarray(site_vel, dtype=jnp.float64),
            ssb_obs_ls=jnp.asarray(ssb_obs_ls, dtype=jnp.float64),
            obs_sun_ls=jnp.asarray(obs_sun_ls, dtype=jnp.float64),
            obs_jupiter_ls=jnp.asarray(jup, dtype=jnp.float64),
            planet_obs_ls=planet_obs_ls,
            correction_tt_sec_pre=jnp.asarray(tt_pre, dtype=jnp.float64),
            correction_tt_tb_sec_pre=(
                None
                if correction_tt_tb_sec is None
                else jnp.asarray(correction_tt_tb_sec, dtype=jnp.float64)
            ),
            einstein_rate=jnp.asarray(einstein, dtype=jnp.float64),
            ifte_records=jnp.asarray(model_static.ifte_records, dtype=jnp.float64),
            ifte_start_jd=jnp.asarray(model_static.ifte_start_jd, dtype=jnp.float64),
            ifte_end_jd=jnp.asarray(model_static.ifte_end_jd, dtype=jnp.float64),
            ifte_step_jd=jnp.asarray(model_static.ifte_step_jd, dtype=jnp.float64),
            ifte_coef_offset=int(model_static.ifte_coef_offset),
            ifte_ncf=int(model_static.ifte_ncf),
            ifte_na=int(model_static.ifte_na),
            bclt_max_iter=bclt_max_iter,
        )
    else:
        from jug.delays.tempo2_geometry import ecliptic_obliquity_rad

        terms, residual_sec = compute_tempo2_toa_model_jax(
            **common,
            ecl_obl_rad=float(ecliptic_obliquity_rad(params, use_native_ecliptic)),
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
            correct_troposphere=bool(model_static.correct_troposphere),
            obs_site_latitude_rad=(
                float(model_static.tropo_packed.latitude_rad)
                if model_static.tropo_packed is not None
                else 0.0
            ),
            obs_site_longitude_rad=(
                float(model_static.tropo_packed.longitude_rad)
                if model_static.tropo_packed is not None
                else 0.0
            ),
            obs_site_height_m=(
                float(model_static.tropo_packed.height_m)
                if model_static.tropo_packed is not None
                else 0.0
            ),
            obs_site_pressure_mbar=(
                float(model_static.tropo_packed.pressure_mbar)
                if model_static.tropo_packed is not None
                else 101.325
            ),
            bclt_max_iter=bclt_max_iter,
        )
    if compute_residuals:
        return terms, jax.device_get(residual_sec)
    return terms, None
