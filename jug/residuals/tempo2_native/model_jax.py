"""Unified JAX Tempo2 TOA model (``formBatsAll`` semantics)."""

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
from jug.residuals.tempo2_native.calculate_bclt_jax import compute_bclt_terms_jax
from jug.residuals.tempo2_native.clock_jax import (
    compute_einstein_rate_jax,
    compute_tempo2_correction_tt_tb_jax,
    compute_tempo2_get_correction_tt_jax,
)
from jug.residuals.tempo2_native.formbats_jax import (
    compute_formbats_jax,
    compute_shklovskii_sec_jax_pure,
    compute_torb_closure_jax,
)
from jug.residuals.tempo2_native.probes import compute_formbats_effective_shapiro_sec
from jug.residuals.tempo2_native.spin_jax import (
    compute_tempo2_phase5_jax,
    spin_params_to_jax,
    track_minus2_frac_phase_jax,
)
from jug.residuals.tempo2_native.types import Tempo2NativeTerms
from jug.utils.constants import SECS_PER_DAY
from jug.utils.timescales import is_tempo2_si_units, parse_timescale


@dataclass(frozen=True)
class Tempo2ModelStatic:
    """Host-loaded static inputs for one TOA batch."""

    obs_itrf_km: np.ndarray
    ephem_path: str
    spk_packed: Tempo2SpkPacked
    eop_packed: IersEopPacked
    chain_mjd_tables: tuple
    chain_offset_tables: tuple
    bipm_mjd: np.ndarray
    bipm_offset: np.ndarray
    ifte_records: np.ndarray
    ifte_start_jd: float
    ifte_end_jd: float
    ifte_step_jd: float
    ifte_coef_offset: int
    ifte_ncf: int
    ifte_na: int
    correct_troposphere: bool
    tropo_packed: TropoObsPacked | None
    dt_emission_sec: np.ndarray
    pulse_numbers: np.ndarray | None
    pn_add: np.ndarray | None
    jump_phase: np.ndarray | None
    tzr_phase: float | None
    ne_sw: float
    planet_shapiro_enabled: bool
    use_native_ecliptic: bool
    track_val: int
    subtract_mean: bool


def _dm_coeffs_from_params(params: dict) -> tuple[float, ...]:
    coeffs: list[float] = []
    k = 0
    while True:
        key = "DM" if k == 0 else f"DM{k}"
        if key not in params:
            break
        coeffs.append(float(params[key]))
        k += 1
    return tuple(coeffs) if coeffs else (0.0,)


def compute_dm_vals_jax(
    sat_mjd: jnp.ndarray,
    *,
    dm_epoch: float,
    dm_coeffs: tuple[float, ...],
) -> jnp.ndarray:
    """Taylor DM model at ``sat`` (JAX-safe, static coefficient order)."""
    dt_years = (sat_mjd - dm_epoch) / 365.25
    out = jnp.zeros_like(sat_mjd)
    for i, coeff in enumerate(dm_coeffs):
        out = out + coeff * (dt_years**i) / math.factorial(i)
    return out


def _dm_vals_numpy(sat_mjd: np.ndarray, params: dict) -> np.ndarray:
    dm_epoch = float(params.get("DMEPOCH", params["PEPOCH"]))
    coeffs = _dm_coeffs_from_params(params)
    return np.asarray(
        compute_dm_vals_jax(
            jnp.asarray(sat_mjd, dtype=jnp.float64),
            dm_epoch=dm_epoch,
            dm_coeffs=coeffs,
        ),
        dtype=np.float64,
    )


def build_tempo2_model_static(
    *,
    params: dict,
    toas: list[Any],
    dt_emission_sec: np.ndarray,
    obs_clocks: dict,
    obs_clock_default: dict,
    bipm_clock: dict,
    obs_code: str,
    ephem_path: str,
    obs_itrf_km: np.ndarray,
    correct_troposphere: bool = False,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    ne_sw: float = 0.0,
    planet_shapiro_enabled: bool = True,
    track_val: int = -2,
    subtract_mean: bool = True,
) -> Tempo2ModelStatic:
    from jug.residuals.tempo2_native.clock_jax import pack_clock_chain_jax
    from jug.utils.ifteph import load_ifte_coeff_tables

    chain = obs_clocks.get(obs_code, obs_clock_default)
    mjd_t, off_t, bipm_mjd, bipm_off = pack_clock_chain_jax(chain, bipm_clock)
    ifte = load_ifte_coeff_tables()
    spk = pack_tempo2_spk_jax(ephem_path)
    eop = pack_iers_eop_jax()
    tropo_packed = None
    if correct_troposphere:
        tropo_packed = pack_tropo_obs_static(
            obs_itrf_km=np.asarray(obs_itrf_km, dtype=np.float64),
        )
    return Tempo2ModelStatic(
        obs_itrf_km=np.asarray(obs_itrf_km, dtype=np.float64),
        ephem_path=str(ephem_path),
        spk_packed=spk,
        eop_packed=eop,
        chain_mjd_tables=tuple(np.asarray(t, dtype=np.float64) for t in mjd_t),
        chain_offset_tables=tuple(np.asarray(t, dtype=np.float64) for t in off_t),
        bipm_mjd=np.asarray(bipm_mjd, dtype=np.float64),
        bipm_offset=np.asarray(bipm_off, dtype=np.float64),
        ifte_records=np.asarray(ifte.records, dtype=np.float64),
        ifte_start_jd=float(ifte.start_jd),
        ifte_end_jd=float(ifte.end_jd),
        ifte_step_jd=float(ifte.step_jd),
        ifte_coef_offset=int(ifte.coef_offset),
        ifte_ncf=int(ifte.ncf),
        ifte_na=int(ifte.na),
        correct_troposphere=bool(correct_troposphere),
        tropo_packed=tropo_packed,
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
        pulse_numbers=None if pulse_numbers is None else np.asarray(pulse_numbers, dtype=np.int64),
        pn_add=None if pn_add is None else np.asarray(pn_add, dtype=np.int64),
        jump_phase=None if jump_phase is None else np.asarray(jump_phase, dtype=np.float64),
        tzr_phase=tzr_phase,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(planet_shapiro_enabled),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        track_val=int(track_val),
        subtract_mean=bool(subtract_mean),
    )


def tempo2_einstein_rate_host(mjd_tt: np.ndarray, params: dict) -> np.ndarray:
    """Host ``einsteinRate`` for ``dm_delays.C`` when ``dilateFreq`` is enabled."""
    from jug.delays.barycentric import compute_einstein_rate

    mjd = np.asarray(mjd_tt, dtype=np.float64)
    if not tempo2_dilate_freq_enabled(params):
        return np.ones_like(mjd, dtype=np.float64)
    units = parse_timescale(params)
    scale = "TCB" if is_tempo2_si_units(units) else "TDB"
    return np.asarray(compute_einstein_rate(mjd, units=scale), dtype=np.float64)


def _spk_segment_to_jax(seg: SpkSegmentPacked) -> SpkSegmentPacked:
    return SpkSegmentPacked(
        init=jnp.asarray(seg.init, dtype=jnp.float64),
        intlen=jnp.asarray(seg.intlen, dtype=jnp.float64),
        coefficients=jnp.asarray(seg.coefficients, dtype=jnp.float64),
    )


def _spk_to_jax(spk: Tempo2SpkPacked) -> Tempo2SpkPacked:
    return Tempo2SpkPacked(
        emb_ssb=_spk_segment_to_jax(spk.emb_ssb),
        earth_emb=_spk_segment_to_jax(spk.earth_emb),
        sun_ssb=_spk_segment_to_jax(spk.sun_ssb),
        planets_ssb={k: _spk_segment_to_jax(v) for k, v in spk.planets_ssb.items()},
    )


def _eop_to_jax(eop: IersEopPacked) -> IersEopPacked:
    return IersEopPacked(
        mjd=jnp.asarray(eop.mjd, dtype=jnp.float64),
        xp=jnp.asarray(eop.xp, dtype=jnp.float64),
        yp=jnp.asarray(eop.yp, dtype=jnp.float64),
        dut1=jnp.asarray(eop.dut1, dtype=jnp.float64),
    )


_PLANET_RSA_NAMES = ("venus", "jupiter", "saturn", "uranus", "neptune")


def host_frozen_vectors_from_tempo2_obs_state(
    td: dict,
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Build staging vectors from ``term_diagnostics['tempo2_obs_state']``.

    Do not use top-level ``jug['ssb_obs_pos_ls']``; that is legacy geometry and is
    known to be metres off the Tempo2-native state.
    """
    from jug.utils.constants import C_KM_S

    state = td.get("tempo2_obs_state")
    if state is None:
        raise ValueError(
            "host-frozen native path requires term_diagnostics['tempo2_obs_state']"
        )

    earth = np.asarray(state["earth_ssb_km"], dtype=np.float64)
    obs = np.asarray(state["observatory_earth_km"], dtype=np.float64)
    sun = np.asarray(state["sun_ssb_km"], dtype=np.float64)
    site_vel = np.asarray(state["site_vel_km_s"], dtype=np.float64)
    planets_ssb = state.get("planet_ssb_km", {}) or {}

    ssb_obs_km = earth[:, :3] + obs[:, :3]
    planet_obs_ls: dict[str, np.ndarray] = {}
    for name, pv in planets_ssb.items():
        pv = np.asarray(pv, dtype=np.float64)
        planet_geo = pv[:, :3] - earth[:, :3]
        planet_obs_ls[name] = (obs[:, :3] - planet_geo) / C_KM_S

    return {
        "earth_ssb_km": earth,
        "observatory_earth_km": obs,
        "site_vel_km_s": site_vel,
        "ssb_obs_ls": ssb_obs_km / C_KM_S,
        "obs_sun_ls": (sun[:, :3] - ssb_obs_km) / C_KM_S,
        "planet_obs_ls": planet_obs_ls,
        "obs_jupiter_ls": planet_obs_ls.get(
            "jupiter", np.zeros((earth.shape[0], 3), dtype=np.float64)
        ),
    }


def planet_rsa_tuple_from_dict(
    planet_obs_ls: dict[str, np.ndarray] | None,
    *,
    n_toa: int,
    obs_jupiter_ls: np.ndarray | None = None,
) -> tuple[np.ndarray, ...]:
    """Tempo2 BCLT rsa tuple (venus … neptune) in light-seconds."""
    zeros = np.zeros((n_toa, 3), dtype=np.float64)
    if planet_obs_ls is None:
        if obs_jupiter_ls is None:
            return tuple(zeros for _ in _PLANET_RSA_NAMES)
        jup = -np.asarray(obs_jupiter_ls, dtype=np.float64)
        return (zeros, jup, zeros, zeros, zeros)
    out: list[np.ndarray] = []
    for name in _PLANET_RSA_NAMES:
        arr = planet_obs_ls.get(name)
        if arr is None:
            out.append(zeros)
        else:
            out.append(np.asarray(arr, dtype=np.float64))
    return tuple(out)


def planet_rsa_tuple_jax_from_dict(
    planet_obs_ls: dict[str, jnp.ndarray] | None,
    *,
    n_toa: int,
    obs_jupiter_ls: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, ...]:
    """JAX variant of :func:`planet_rsa_tuple_from_dict`."""
    zeros = jnp.zeros((n_toa, 3), dtype=jnp.float64)
    if planet_obs_ls is None:
        if obs_jupiter_ls is None:
            return tuple(zeros for _ in _PLANET_RSA_NAMES)
        jup = -jnp.asarray(obs_jupiter_ls, dtype=jnp.float64)
        return (zeros, jup, zeros, zeros, zeros)
    out: list[jnp.ndarray] = []
    for name in _PLANET_RSA_NAMES:
        arr = None if planet_obs_ls is None else planet_obs_ls.get(name)
        if arr is None:
            out.append(zeros)
        else:
            out.append(jnp.asarray(arr, dtype=jnp.float64))
    return tuple(out)


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
        "correct_troposphere",
        "obs_site_latitude_rad",
        "obs_site_longitude_rad",
        "obs_site_height_m",
        "obs_site_pressure_mbar",
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
    correct_troposphere: bool = False,
) -> tuple[Tempo2NativeTerms, jnp.ndarray]:
    """Full Tempo2 delay/spin chain in one JIT graph.

    .. warning::
        **Extremely slow first compile.** This function evaluates clocks, SPK
        ephemeris, EOP site motion, IFTE bootstrap, troposphere, BCLT, formBats,
        and spin inside a single ``@jax.jit`` boundary. On wsrt167 (167 TOAs) the
        initial compile can take **minutes**. Production fitting and fast dev loops
        should use ``compute_tempo2_toa_model_staging_with_host_inputs_jax`` with
        host-frozen inputs instead. Enable only via
        ``USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH`` or ``JUG_TEMPO2_NATIVE_FULL_INGRAPH=1``.

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
        elevation_rad = tempo2_source_elevation_rad_jax(
            zenith_gcrs,
            pos_pulsar,
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
    )
    shap_delay = bclt.shapiro_sun_sec + jnp.where(
        planet_shapiro_enabled,
        bclt.shapiro_planets_sec,
        0.0,
    )
    bat_corr_day, bat_corr_resid, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_mjd,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        jnp.zeros_like(sat_mjd),
    )
    shk = compute_shklovskii_sec_jax_pure(
        bat_mjd,
        params_pepoch,
        params_f_terms,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
        posepoch_mjd=shk_posepoch,
    )
    _, _, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_mjd,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        shk,
    )
    dt_emit = jnp.asarray(dt_emission_sec, dtype=jnp.float64)
    torb = compute_torb_closure_jax(bbat_mjd, dt_emit, params_pepoch)
    terms = Tempo2NativeTerms(
        sat_mjd=sat_mjd,
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
        shklovskii_sec=shk,
        torb_sec=torb,
        dt_emission_sec=dt_emit,
        dt_ssb_sec=bclt.dt_ssb_sec,
        bclt_iterations=bclt.bclt_iterations,
        converged=bclt.converged,
    )
    phase5 = compute_tempo2_phase5_jax(
        bbat_mjd,
        torb,
        params_f_terms,
        params_pepoch,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
    )
    if track_val == -2 and pulse_numbers is not None and pn_add is not None:
        frac, _pulse = track_minus2_frac_phase_jax(
            phase5, bbat_mjd, params_f_terms[0], pulse_numbers, pn_add
        )
    else:
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / params_f_terms[0]
    if subtract_mean:
        residual_sec = residual_sec - jnp.mean(residual_sec)
    return terms, residual_sec


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
) -> tuple[Tempo2NativeTerms, jnp.ndarray]:
    """Production Tempo2 TOA model with host-frozen static inputs.

    Accepts precomputed geometry, clocks, and ``einsteinRate`` from
    ``term_diagnostics['tempo2_obs_state']``. Only the parameter-dependent tail
    (BCLT, formBats, Shklovskii, spin) runs inside JAX. This is the **default**
    production path when ``USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH`` is False.

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
    shap_delay = bclt.shapiro_sun_sec + jnp.where(
        planet_shapiro_enabled,
        bclt.shapiro_planets_sec,
        0.0,
    )
    bat_corr_day, bat_corr_resid, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_mjd,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        jnp.zeros_like(sat_mjd),
    )
    shk = compute_shklovskii_sec_jax_pure(
        bat_mjd,
        params_pepoch,
        params_f_terms,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
        posepoch_mjd=shk_posepoch,
    )
    _, _, bat_mjd, bbat_mjd = compute_formbats_jax(
        sat_mjd,
        tt,
        tt_tb,
        tropo,
        bclt.roemer_sec,
        shap_delay,
        bclt.tdis1_sec,
        bclt.tdis2_sec,
        shk,
    )
    dt_emit = jnp.asarray(dt_emission_sec, dtype=jnp.float64)
    torb = compute_torb_closure_jax(bbat_mjd, dt_emit, params_pepoch)
    terms = Tempo2NativeTerms(
        sat_mjd=sat_mjd,
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
        shklovskii_sec=shk,
        torb_sec=torb,
        dt_emission_sec=dt_emit,
        dt_ssb_sec=bclt.dt_ssb_sec,
        bclt_iterations=bclt.bclt_iterations,
        converged=bclt.converged,
    )
    phase5 = compute_tempo2_phase5_jax(
        bbat_mjd,
        torb,
        params_f_terms,
        params_pepoch,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
    )
    if track_val == -2 and pulse_numbers is not None and pn_add is not None:
        frac, _pulse = track_minus2_frac_phase_jax(
            phase5, bbat_mjd, params_f_terms[0], pulse_numbers, pn_add
        )
    else:
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / params_f_terms[0]
    if subtract_mean:
        residual_sec = residual_sec - jnp.mean(residual_sec)
    return terms, residual_sec


compute_tempo2_toa_model_with_frozen_terms_for_tests = (
    compute_tempo2_toa_model_staging_with_host_inputs_jax
)


def prepare_ephemeris_inputs_jax(
    ephem_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    ephem_path: str,
    *,
    site_mjd: np.ndarray | None = None,
    site_time_scale: str = "tt",
) -> dict[str, jnp.ndarray]:
    """Host ephemeris setup → JAX arrays (staging / tests only)."""
    from jug.delays.tempo2_ephemeris import compute_tempo2_observatory_state
    from jug.delays.tempo2_geometry import tempo2_observatory_chain_vectors

    state = compute_tempo2_observatory_state(
        np.asarray(ephem_mjd, dtype=np.float64),
        np.asarray(obs_itrf_km, dtype=np.float64).reshape(3),
        ephem_path=ephem_path,
        site_mjd=site_mjd,
        site_time_scale=site_time_scale,
    )
    ssb_obs_km, ssb_obs_ls, obs_sun_ls, planets = tempo2_observatory_chain_vectors(state)
    jup = planets.get("jupiter", np.zeros((len(ephem_mjd), 3)))
    return {
        "earth_ssb_km": jnp.asarray(state.earth_ssb_km, dtype=jnp.float64),
        "observatory_earth_km": jnp.asarray(state.observatory_earth_km[:, :3], dtype=jnp.float64),
        "site_vel_km_s": jnp.asarray(state.site_vel_km_s, dtype=jnp.float64),
        "ssb_obs_ls": jnp.asarray(ssb_obs_ls, dtype=jnp.float64),
        "obs_sun_ls": jnp.asarray(obs_sun_ls, dtype=jnp.float64),
        "obs_jupiter_ls": jnp.asarray(jup, dtype=jnp.float64),
        "ssb_obs_km": jnp.asarray(ssb_obs_km, dtype=jnp.float64),
    }


def run_tempo2_toa_model(
    *,
    params: dict,
    sat_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    ephem_mjd: np.ndarray,
    static: Tempo2ModelStatic,
) -> tuple[Tempo2NativeTerms, np.ndarray]:
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
) -> tuple[Tempo2NativeTerms, np.ndarray | None]:
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
    f_terms, pepoch = spin_params_to_jax(params)
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
        )
    else:
        terms, residual_sec = compute_tempo2_toa_model_jax(
            **common,
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
        )
    if compute_residuals:
        return terms, jax.device_get(residual_sec)
    return terms, None
