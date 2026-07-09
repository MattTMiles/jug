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

def sat_daysec_numpy_from_td_and_toas(td: dict, toas: list[Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``(int_day, sec_in_day)`` SAT from term diagnostics or TOA reader."""
    if "sat_int_day" in td and "sat_sec_in_day" in td:
        return (
            np.asarray(td["sat_int_day"], dtype=np.float64),
            np.asarray(td["sat_sec_in_day"], dtype=np.float64),
        )
    if toas:
        sat_int = np.array([t.mjd_int for t in toas], dtype=np.float64)
        sat_sec = np.array([t.mjd_frac * SECS_PER_DAY for t in toas], dtype=np.float64)
        return sat_int, sat_sec
    sat_mjd = np.asarray(td["sat_mjd"], dtype=np.float64)
    sat_int = np.floor(sat_mjd)
    return sat_int, (sat_mjd - sat_int) * SECS_PER_DAY


def _chain_mode(mode: str | None = None) -> str:
    """Return the active tempo2-native JAX graph mode."""
    return tempo2_graph_mode(mode)


@dataclass(frozen=True)
class NativeDeltaPack:
    """Prepacked static inputs for tempo2-native residual evaluation (all graph modes)."""

    mode: str
    sat_mjd: Any
    freq_mhz: Any
    dt_emission_sec: Any
    ne_sw: float
    use_native_ecliptic: bool
    dm_epoch: float
    dm_coeffs_ref: tuple[float, ...]
    posepoch_mjd: float
    shk_posepoch: float
    pmrv_rad_century: float
    dilate_freq: bool
    si_units: bool
    units_tdb: bool
    planet_shapiro_enabled: bool
    track_val: int
    subtract_mean: bool
    dshk: float
    jump_phase: Any | None
    tzr_phase: Any | None
    pulse_numbers: Any | None
    pn_add: Any | None
    sat_int_day: Any | None = None
    sat_sec_in_day: Any | None = None
    pep_int: Any | None = None
    pep_frac: Any | None = None
    # staged_bclt / fixed_state_nonlinear (host-frozen staging)
    earth_ssb_km: Any | None = None
    observatory_earth_km: Any | None = None
    site_vel_km_s: Any | None = None
    ssb_obs_ls: Any | None = None
    obs_sun_ls: Any | None = None
    obs_jupiter_ls: Any | None = None
    planet_obs_ls: dict[str, Any] | None = None
    correction_tt_sec: Any | None = None
    correction_tt_tb_sec: Any | None = None
    einstein_rate: Any | None = None
    tropo_sec: Any | None = None
    dt_ssb_ref_sec: Any | None = None
    # full in-graph chain only
    obs_itrf_km: Any | None = None
    spk_packed: Any | None = None
    eop_packed: Any | None = None
    chain_mjd_tables: tuple[Any, ...] | None = None
    chain_offset_tables: tuple[Any, ...] | None = None
    bipm_mjd: Any | None = None
    bipm_offset: Any | None = None
    ifte_records: Any | None = None
    ifte_start_jd: Any | None = None
    ifte_end_jd: Any | None = None
    ifte_step_jd: Any | None = None
    ifte_coef_offset: int | None = None
    ifte_ncf: int | None = None
    ifte_na: int | None = None
    correct_troposphere: bool | None = None
    obs_site_latitude_rad: float | None = None
    obs_site_longitude_rad: float | None = None
    obs_site_height_m: float | None = None
    obs_site_pressure_mbar: float | None = None
    bclt_max_iter: int | None = None


def _param_scalar_jax(params: dict, name: str, default: float = 0.0):
    key = name.upper()
    if key in params:
        return params[key]
    return default


def _spin_f_terms_jax(params: dict) -> jnp.ndarray:
    terms = []
    for i in range(10):
        key = f"F{i}"
        if key in params:
            terms.append(jnp.asarray(_param_scalar_jax(params, key), dtype=jnp.float64))
        elif i == 0:
            terms.append(jnp.asarray(_param_scalar_jax(params, "F0", 1.0), dtype=jnp.float64))
        else:
            break
    return jnp.stack(terms)


def _raj_decj_rad_jax(params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    if "_raj_rad" in params:
        alpha = jnp.asarray(params["_raj_rad"], dtype=jnp.float64)
    else:
        alpha = jnp.asarray(params["RAJ"], dtype=jnp.float64)
    if "_decj_rad" in params:
        delta = jnp.asarray(params["_decj_rad"], dtype=jnp.float64)
    else:
        delta = jnp.asarray(params["DECJ"], dtype=jnp.float64)
    return alpha, delta


def pulsar_vectors_from_params_jax(
    params: dict,
    *,
    use_native_ecliptic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX port of ``build_tempo2_pulsar_vectors`` for traced fit parameters."""
    if use_native_ecliptic:
        lon = jnp.deg2rad(jnp.asarray(params["_ecliptic_lon_deg"], dtype=jnp.float64))
        lat = jnp.deg2rad(jnp.asarray(params["_ecliptic_lat_deg"], dtype=jnp.float64))
        pmra = jnp.asarray(
            params.get("_ecliptic_pm_lon", params.get("PMRA", 0.0)), dtype=jnp.float64
        )
        pmdec = jnp.asarray(
            params.get("_ecliptic_pm_lat", params.get("PMDEC", 0.0)), dtype=jnp.float64
        )
        lat_for_vel = lat
    else:
        alpha, delta = _raj_decj_rad_jax(params)
        lon = alpha
        lat = delta
        pmra = jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64)
        pmdec = jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64)
        lat_for_vel = delta

    ca, sa = jnp.cos(lon), jnp.sin(lon)
    cd, sd = jnp.cos(lat), jnp.sin(lat)
    pos = jnp.stack([ca * cd, sa * cd, sd])
    convert = jnp.asarray(np.pi / 180.0 / 3600.0 / 1000.0 * 100.0, dtype=jnp.float64)
    cos_lat = jnp.cos(lat_for_vel)
    vel = convert * jnp.stack(
        [
            -pmra / cos_lat * sa * cd - pmdec * ca * sd,
            pmra / cos_lat * ca * cd - pmdec * sa * sd,
            pmdec * cd,
        ]
    )
    convert2 = convert * 100.0
    pmra2 = jnp.asarray(_param_scalar_jax(params, "PMRA2"), dtype=jnp.float64)
    pmdec2 = jnp.asarray(_param_scalar_jax(params, "PMDEC2"), dtype=jnp.float64)
    acc = convert2 * jnp.stack(
        [
            -pmra2 / cos_lat * sa * cd - pmdec2 * ca * sd,
            pmra2 / cos_lat * ca * cd - pmdec2 * sa * sd,
            pmdec2 * cd,
        ]
    )
    return pos, vel, acc


def _dm_coeffs_jax(params: dict) -> tuple[jnp.ndarray, ...]:
    coeffs: list[jnp.ndarray] = []
    k = 0
    while True:
        key = "DM" if k == 0 else f"DM{k}"
        if key not in params:
            break
        coeffs.append(jnp.asarray(_param_scalar_jax(params, key), dtype=jnp.float64))
        k += 1
    if not coeffs:
        coeffs = [jnp.asarray(0.0, dtype=jnp.float64)]
    return tuple(coeffs)


def track2_pulse_arrays_from_toas(
    toas: list[Any],
    params: dict,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract TRACK−2 ``-pn`` / ``-pnadd`` arrays when present on all TOAs."""
    track_val = params.get("TRACK", None)
    if track_val is None or int(track_val) != -2:
        return None, None
    pn_flags = [toa.flags.get("pn") for toa in toas]
    if not all(pn is not None for pn in pn_flags):
        return None, None
    pulse_numbers = np.array([int(pn) for pn in pn_flags], dtype=np.int64)
    pn_add_running = np.int64(-1)
    pn_add = np.empty(len(toas), dtype=np.int64)
    for i, toa in enumerate(toas):
        pn_add[i] = pn_add_running
        pnadd_val = toa.flags.get("pnadd")
        if pnadd_val is not None:
            pn_add_running += np.int64(int(pnadd_val))
    return pulse_numbers, pn_add


def _load_model_static_for_native_chain(
    params: dict,
    toas: list[Any],
    jug_result: dict,
    *,
    clock_dir=None,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    track_val: int = -2,
) -> Tempo2ModelStatic:
    """Load clock tables and pack static inputs for the unified JIT model."""
    from jug.io.clock import resolve_clock_dir
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile
    from jug.residuals.simple_calculator import _load_clock_corrections
    from jug.utils.constants import OBSERVATORIES

    compatibility = jug_result.get("compatibility", "tempo2")
    clock_dir = resolve_clock_dir(clock_dir, compatibility=compatibility)
    observatory = toas[0].observatory if toas else "wsrt"
    obs_itrf = OBSERVATORIES.get(observatory.lower())
    if obs_itrf is None:
        raise ValueError(f"Unknown observatory for native chain: {observatory}")
    all_obs_codes = sorted(set(t.observatory.lower() for t in toas))
    mjd_utc = np.array([t.mjd_int + t.mjd_frac for t in toas], dtype=np.float64)
    clk = _load_clock_corrections(
        observatory, all_obs_codes, clock_dir, params, mjd_utc, verbose=False
    )
    from jug.residuals.diagnostic_conventions import default_conventions
    from jug.residuals.engine_conventions import _flag_from_par

    diagnostic_conv = default_conventions(compatibility)
    profile = resolve_engine_profile(
        params,
        compatibility,
        implicit_tempo2_defaults=diagnostic_conv.apply_tempo2_implicit_defaults(
            compatibility
        ),
    )
    correct_tropo = bool(profile.correct_troposphere)
    if compatibility == "tempo2" and not correct_tropo:
        if "CORRECT_TROPOSPHERE" in params:
            correct_tropo = _flag_from_par(params, "CORRECT_TROPOSPHERE")
        else:
            correct_tropo = True
    from jug.residuals.diagnostic_conventions import resolve_planet_shapiro_enabled

    return build_tempo2_model_static(
        params=params,
        toas=toas,
        dt_emission_sec=np.asarray(jug_result["dt_sec"], dtype=np.float64),
        obs_clocks=clk["obs_clocks"],
        obs_clock_default=clk["obs_clock"],
        bipm_clock=clk["bipm_clock"],
        obs_code=observatory.lower(),
        ephem_path=resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405")),
        obs_itrf_km=np.asarray(obs_itrf, dtype=np.float64),
        correct_troposphere=correct_tropo,
        ne_sw=resolve_ne_sw_cm3(params, profile),
        planet_shapiro_enabled=resolve_planet_shapiro_enabled(params, profile),
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
        track_val=int(track_val),
    )

