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
def _host_frozen_placeholder_spk_segment() -> SpkSegmentPacked:
    z = jnp.zeros((1,), dtype=jnp.float64)
    return SpkSegmentPacked(
        init=z,
        intlen=z,
        coefficients=jnp.zeros((1, 1, 1), dtype=jnp.float64),
    )


def _host_frozen_placeholder_spk() -> Tempo2SpkPacked:
    seg = _host_frozen_placeholder_spk_segment()
    return Tempo2SpkPacked(
        emb_ssb=seg,
        earth_emb=seg,
        sun_ssb=seg,
        planets_ssb={},
    )


def _host_frozen_placeholder_eop() -> IersEopPacked:
    z = np.zeros(1, dtype=np.float64)
    return IersEopPacked(mjd=z, xp=z, yp=z, dut1=z)


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
    host_frozen: bool = False


def build_tempo2_model_static_host_frozen(
    *,
    params: dict,
    toas: list[Any],
    dt_emission_sec: np.ndarray,
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
    """Metadata-only static pack for host-frozen JAX graph modes.

    Skips SPK, EOP, IFTE, and clock-chain loading — those inputs are frozen in
    ``term_diagnostics`` / ``tempo2_obs_state`` at pack-build time.
    """
    from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path

    tropo_packed = None
    if correct_troposphere:
        tropo_packed = pack_tropo_obs_static(
            obs_itrf_km=np.asarray(obs_itrf_km, dtype=np.float64),
        )
    z = np.zeros(1, dtype=np.float64)
    return Tempo2ModelStatic(
        obs_itrf_km=np.asarray(obs_itrf_km, dtype=np.float64),
        ephem_path=str(resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405"))),
        spk_packed=_host_frozen_placeholder_spk(),
        eop_packed=_host_frozen_placeholder_eop(),
        chain_mjd_tables=(z,),
        chain_offset_tables=(z,),
        bipm_mjd=z,
        bipm_offset=z,
        ifte_records=z,
        ifte_start_jd=0.0,
        ifte_end_jd=0.0,
        ifte_step_jd=1.0,
        ifte_coef_offset=0,
        ifte_ncf=0,
        ifte_na=0,
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
        host_frozen=True,
    )


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
    from jug.residuals.tempo2.clock_jax import pack_clock_chain_jax
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
