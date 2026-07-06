"""Unified JAX Tempo2 TOA model (``formBatsAll`` semantics)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_geometry import (
    build_tempo2_pulsar_vectors,
    pmrv_rad_per_century,
    tempo2_observatory_chain_vectors,
)
from jug.delays.tempo2_ephemeris import compute_tempo2_observatory_state
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
from jug.utils.timescales import parse_timescale


@dataclass(frozen=True)
class Tempo2ModelStatic:
    """Host-loaded static inputs for one TOA batch."""

    obs_itrf_km: np.ndarray
    ephem_path: str
    chain_mjd_tables: tuple
    chain_offset_tables: tuple
    bipm_mjd: np.ndarray
    bipm_offset: np.ndarray
    tropo_sec: np.ndarray
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


def _dm_vals_numpy(sat_mjd: np.ndarray, params: dict) -> np.ndarray:
    import math

    dm_epoch = float(params.get("DMEPOCH", params["PEPOCH"]))
    coeffs = []
    k = 0
    while True:
        key = "DM" if k == 0 else f"DM{k}"
        if key not in params:
            break
        coeffs.append(float(params[key]))
        k += 1
    if not coeffs:
        coeffs = [0.0]
    out = np.zeros(len(sat_mjd), dtype=np.float64)
    for i, sat in enumerate(sat_mjd):
        dt_years = (float(sat) - dm_epoch) / 365.25
        out[i] = sum(coeffs[j] * (dt_years**j) / math.factorial(j) for j in range(len(coeffs)))
    return out


def build_tempo2_model_static(
    *,
    params: dict,
    toas: list[Any],
    tropo_sec: np.ndarray,
    dt_emission_sec: np.ndarray,
    obs_clocks: dict,
    obs_clock_default: dict,
    bipm_clock: dict,
    obs_code: str,
    ephem_path: str,
    obs_itrf_km: np.ndarray,
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

    chain = obs_clocks.get(obs_code, obs_clock_default)
    mjd_t, off_t, bipm_mjd, bipm_off = pack_clock_chain_jax(chain, bipm_clock)
    return Tempo2ModelStatic(
        obs_itrf_km=np.asarray(obs_itrf_km, dtype=np.float64),
        ephem_path=str(ephem_path),
        chain_mjd_tables=mjd_t,
        chain_offset_tables=off_t,
        bipm_mjd=np.asarray(jax.device_get(bipm_mjd)),
        bipm_offset=np.asarray(jax.device_get(bipm_off)),
        tropo_sec=np.asarray(tropo_sec, dtype=np.float64),
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


@jax.jit
def compute_tempo2_toa_model_jax(
    *,
    sat_mjd: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    params_f_terms: jnp.ndarray,
    params_pepoch: jnp.float64,
    pos_pulsar: jnp.ndarray,
    vel_pulsar: jnp.ndarray,
    acc_pulsar: jnp.ndarray,
    dm_vals: jnp.ndarray,
    tropo_sec: jnp.ndarray,
    dt_emission_sec: jnp.ndarray,
    chain_mjd_tables: tuple[jnp.ndarray, ...],
    chain_offset_tables: tuple[jnp.ndarray, ...],
    bipm_mjd: jnp.ndarray,
    bipm_offset: jnp.ndarray,
    ne_sw: float,
    earth_ssb_km: jnp.ndarray,
    observatory_earth_km: jnp.ndarray,
    site_vel_km_s: jnp.ndarray,
    ssb_obs_ls: jnp.ndarray,
    obs_sun_ls: jnp.ndarray,
    obs_jupiter_ls: jnp.ndarray,
    posepoch_mjd: float,
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
    """Full Tempo2 delay/spin chain in one JIT graph."""
    tt = compute_tempo2_get_correction_tt_jax(
        sat_mjd,
        chain_mjd_tables=chain_mjd_tables,
        chain_offset_tables=chain_offset_tables,
        bipm_mjd=bipm_mjd,
        bipm_offset=bipm_offset,
    )
    mjd_tt = sat_mjd + tt / SECS_PER_DAY
    tt_tb = compute_tempo2_correction_tt_tb_jax(
        mjd_tt,
        observatory_earth_km,
        earth_ssb_km[:, 3:6],
        units_tdb=units_tdb,
        si_units=si_units,
    )
    einstein = (
        compute_einstein_rate_jax(mjd_tt, si_units=si_units)
        if dilate_freq
        else jnp.ones_like(sat_mjd)
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
    )
    tropo = jnp.asarray(tropo_sec, dtype=jnp.float64)
    shap_delay = bclt.shapiro_sun_sec + bclt.shapiro_planets_sec
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


def prepare_ephemeris_inputs_jax(
    tdb_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    ephem_path: str,
) -> dict[str, jnp.ndarray]:
    """Host ephemeris setup → JAX arrays for ``compute_tempo2_toa_model_jax``."""
    state = compute_tempo2_observatory_state(
        np.asarray(tdb_mjd, dtype=np.float64),
        np.asarray(obs_itrf_km, dtype=np.float64).reshape(3),
        ephem_path=ephem_path,
    )
    ssb_obs_km, ssb_obs_ls, obs_sun_ls, planets = tempo2_observatory_chain_vectors(state)
    jup = planets.get("jupiter", np.zeros((len(tdb_mjd), 3)))
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
    tdb_mjd: np.ndarray,
    static: Tempo2ModelStatic,
) -> tuple[Tempo2NativeTerms, np.ndarray]:
    """Host wrapper: build pulsar vectors + ephemeris, run JIT model."""
    pos, vel, acc = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=static.use_native_ecliptic
    )
    f_terms, pepoch = spin_params_to_jax(params)
    dm_vals = _dm_vals_numpy(sat_mjd, params)
    eph = prepare_ephemeris_inputs_jax(tdb_mjd, static.obs_itrf_km, static.ephem_path)
    units = parse_timescale(params)
    dilate = str(params.get("DILATEFREQ", "N")).upper() in ("Y", "YES", "TRUE", "1")
    pmrv = pmrv_rad_per_century(float(params.get("PMRV", 0.0)))
    terms, res = compute_tempo2_toa_model_jax(
        sat_mjd=jnp.asarray(sat_mjd, dtype=jnp.float64),
        freq_mhz=jnp.asarray(freq_mhz, dtype=jnp.float64),
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=jnp.asarray(pos, dtype=jnp.float64),
        vel_pulsar=jnp.asarray(vel, dtype=jnp.float64),
        acc_pulsar=jnp.asarray(acc, dtype=jnp.float64),
        dm_vals=jnp.asarray(dm_vals, dtype=jnp.float64),
        tropo_sec=jnp.asarray(static.tropo_sec, dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(static.dt_emission_sec, dtype=jnp.float64),
        chain_mjd_tables=tuple(jnp.asarray(t, dtype=jnp.float64) for t in static.chain_mjd_tables),
        chain_offset_tables=tuple(
            jnp.asarray(t, dtype=jnp.float64) for t in static.chain_offset_tables
        ),
        bipm_mjd=jnp.asarray(static.bipm_mjd, dtype=jnp.float64),
        bipm_offset=jnp.asarray(static.bipm_offset, dtype=jnp.float64),
        ne_sw=static.ne_sw,
        earth_ssb_km=eph["earth_ssb_km"],
        observatory_earth_km=eph["observatory_earth_km"],
        site_vel_km_s=eph["site_vel_km_s"],
        ssb_obs_ls=eph["ssb_obs_ls"],
        obs_sun_ls=eph["obs_sun_ls"],
        obs_jupiter_ls=eph["obs_jupiter_ls"],
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        parallax_mas=float(params.get("PX", 0.0)),
        pmrv_rad_century=pmrv,
        dilate_freq=dilate,
        si_units=units == "SI_UNITS",
        units_tdb=units == "TDB",
        planet_shapiro_enabled=static.planet_shapiro_enabled,
        track_val=static.track_val,
        subtract_mean=static.subtract_mean,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        pmra=float(params.get("PMRA", 0.0)),
        pmdec=float(params.get("PMDEC", 0.0)),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        jump_phase=(
            None
            if static.jump_phase is None
            else jnp.asarray(static.jump_phase, dtype=jnp.float64)
        ),
        tzr_phase=(
            None if static.tzr_phase is None else jnp.asarray(static.tzr_phase, dtype=jnp.float64)
        ),
        pulse_numbers=(
            None
            if static.pulse_numbers is None
            else jnp.asarray(static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=None if static.pn_add is None else jnp.asarray(static.pn_add, dtype=jnp.int64),
    )
    return terms, jax.device_get(res)
