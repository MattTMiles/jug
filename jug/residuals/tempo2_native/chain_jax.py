"""Production JAX orchestrator for tempo2-native clock/delay/spin chain."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_ephemeris import resolve_tempo2_ephemeris_path
from jug.utils.constants import SECS_PER_DAY
from jug.residuals.tempo2_native.model_jax import (
    Tempo2ModelStatic,
    build_tempo2_model_static,
    run_tempo2_toa_model_with_fixed_ifte_geometry,
)
from jug.residuals.tempo2_native.types import Tempo2NativeTerms


def _load_model_static_for_native_chain(
    params: dict,
    toas: list[Any],
    jug_result: dict,
    *,
    clock_dir=None,
) -> Tempo2ModelStatic:
    """Load clock tables and pack static inputs for the unified JIT model."""
    from pathlib import Path

    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile
    from jug.residuals.simple_calculator import _load_clock_corrections
    from jug.utils.constants import OBSERVATORIES

    if clock_dir is None:
        clock_dir = Path(__file__).resolve().parents[3] / "data" / "clock"
    observatory = toas[0].observatory if toas else "wsrt"
    obs_itrf = OBSERVATORIES.get(observatory.lower())
    if obs_itrf is None:
        raise ValueError(f"Unknown observatory for native chain: {observatory}")
    all_obs_codes = sorted(set(t.observatory.lower() for t in toas))
    mjd_utc = np.array([t.mjd_int + t.mjd_frac for t in toas], dtype=np.float64)
    clk = _load_clock_corrections(
        observatory, all_obs_codes, clock_dir, params, mjd_utc, verbose=False
    )
    td = jug_result["term_diagnostics"]
    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    return build_tempo2_model_static(
        params=params,
        toas=toas,
        tropo_sec=np.asarray(td["tropo_delay_sec"], dtype=np.float64),
        dt_emission_sec=np.asarray(jug_result["dt_sec"], dtype=np.float64),
        obs_clocks=clk["obs_clocks"],
        obs_clock_default=clk["obs_clock"],
        bipm_clock=clk["bipm_clock"],
        obs_code=observatory.lower(),
        ephem_path=resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405")),
        obs_itrf_km=np.asarray(obs_itrf, dtype=np.float64),
        ne_sw=resolve_ne_sw_cm3(params, profile),
    )


def compute_tempo2_native_terms_jax(
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
) -> Tempo2NativeTerms:
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
    )

    if ssb_obs_ls_fixed is None or obs_sun_ls_fixed is None:
        raise ValueError("tempo2-native BCLT requires fixed IFTE geometry arrays")
    if model_static is None:
        raise ValueError(
            "compute_tempo2_native_terms_jax requires model_static with clock and IFTE tables"
        )

    if freq_mhz_topocentric is not None:
        freq = np.asarray(freq_mhz_topocentric, dtype=np.float64)
    else:
        freq = np.asarray(freq_mhz, dtype=np.float64)

    planets = obs_planets_ls_fixed or {}
    jup = planets.get("jupiter")
    if jup is None:
        jup = np.zeros((len(np.asarray(sat_mjd)), 3), dtype=np.float64)
    else:
        jup = np.asarray(jup, dtype=np.float64)

    site_vel = (
        np.zeros((len(np.asarray(sat_mjd)), 3), dtype=np.float64)
        if site_vel_km_s is None
        else np.asarray(site_vel_km_s, dtype=np.float64)
    )

    terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=freq,
        tropo_sec=np.asarray(tropospheric_sec, dtype=np.float64),
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
        ssb_obs_ls=np.asarray(ssb_obs_ls_fixed, dtype=np.float64),
        obs_sun_ls=np.asarray(obs_sun_ls_fixed, dtype=np.float64),
        obs_jupiter_ls=jup,
        earth_ssb_km=np.asarray(earth_ssb_km, dtype=np.float64),
        observatory_earth_km=np.asarray(observatory_earth_km, dtype=np.float64),
        site_vel_km_s=site_vel,
        earth_ssb_vel_km_s=np.asarray(earth_ssb_vel_km_s, dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=planet_shapiro_enabled,
        use_native_ecliptic=use_native_ecliptic,
    )
    return terms


def compute_tempo2_native_residuals_jax(
    *,
    native_terms: Tempo2NativeTerms,
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
    del weights, mean_mode
    from jug.residuals.tempo2_native.spin_jax import (
        compute_tempo2_phase5_jax,
        spin_params_to_jax,
        track_minus2_frac_phase_jax,
    )

    f_terms, pepoch = spin_params_to_jax(params)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    phase5 = compute_tempo2_phase5_jax(
        native_terms.bbat_mjd,
        native_terms.torb_sec,
        f_terms,
        pepoch,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
    )
    if int(track_val) == -2 and pulse_numbers is not None and pn_add is not None:
        frac, pulse = track_minus2_frac_phase_jax(
            phase5,
            native_terms.bbat_mjd,
            f_terms[0],
            jnp.asarray(pulse_numbers, dtype=jnp.int64),
            jnp.asarray(pn_add, dtype=jnp.int64),
        )
    else:
        pulse = jnp.zeros_like(phase5)
        frac = phase5 - jnp.trunc(phase5)
    residual_sec = frac / f_terms[0]
    if subtract_mean:
        residual_sec = residual_sec - jnp.mean(residual_sec)
    return residual_sec, pulse, native_terms


def compute_native_spin_residual_sec_jax(
    native_terms: Tempo2NativeTerms,
    params,
    *,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    track_val: int = -2,
) -> jnp.ndarray:
    """Spin/track residual from precomputed unified-model terms (JAX-safe)."""
    residual_sec, _, _ = compute_tempo2_native_residuals_jax(
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


def prepare_native_terms_for_setup(
    jug_result: dict,
    params: dict,
    toas: list[Any],
) -> Tempo2NativeTerms:
    """Build unified native terms once for fit setup / residual_delta."""
    return prepare_native_chain_from_simple_result(jug_result, params, toas)


def prepare_native_chain_from_simple_result(
    jug_result: dict,
    params: dict,
    toas: list[Any],
    *,
    observatory_earth_km: np.ndarray | None = None,
    earth_ssb_km: np.ndarray | None = None,
    earth_ssb_vel_km_s: np.ndarray | None = None,
    ephem_path: str | None = None,
    use_model_epoch_batcorr: bool = False,
) -> Tempo2NativeTerms:
    """Build native terms from ``compute_residuals_simple`` geometry exports."""
    from jug.delays.tempo2_geometry import Tempo2ObservatoryState, tempo2_observatory_chain_vectors
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    td = jug_result["term_diagnostics"]
    tdis1 = np.asarray(td["dm_delay_sec"], dtype=np.float64) + np.asarray(
        td.get("dmx_delay_sec", 0.0), dtype=np.float64
    )
    prebinary_jug = np.asarray(td["prebinary_delay_sec"], dtype=np.float64)
    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    ne_sw = resolve_ne_sw_cm3(params, profile)
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    planets = jug_result.get("obs_planet_pos_ls")
    formbats_tt = td.get("formbats_correction_tt_sec", td.get("correction_tt_sec"))
    obs_state = td.get("tempo2_obs_state")
    site_vel = None
    ssb_obs_ls = jug_result.get("ssb_obs_pos_ls")
    obs_sun_ls = jug_result.get("obs_sun_pos_ls")
    if obs_state is not None:
        site_vel = obs_state.get("site_vel_km_s")
        if observatory_earth_km is None:
            observatory_earth_km = np.asarray(
                obs_state["observatory_earth_km"], dtype=np.float64
            )[:, :3]
        earth_ssb_arr = np.asarray(obs_state["earth_ssb_km"], dtype=np.float64)
        if earth_ssb_km is None:
            earth_ssb_km = earth_ssb_arr[:, :3]
        if earth_ssb_vel_km_s is None:
            earth_ssb_vel_km_s = earth_ssb_arr[:, 3:6]
        sun_ssb = obs_state.get("sun_ssb_km")
        planet_ssb = obs_state.get("planet_ssb_km", {})
        if sun_ssb is None:
            sun_ssb = np.zeros((len(toas), 6), dtype=np.float64)
        else:
            sun_ssb = np.asarray(sun_ssb, dtype=np.float64)
        if isinstance(planet_ssb, dict):
            planet_ssb = {
                name: np.asarray(arr, dtype=np.float64) for name, arr in planet_ssb.items()
            }
        else:
            planet_ssb = {}
        state = Tempo2ObservatoryState(
            earth_ssb_km=earth_ssb_arr,
            observatory_earth_km=np.asarray(
                obs_state["observatory_earth_km"], dtype=np.float64
            ),
            sun_ssb_km=sun_ssb,
            planet_ssb_km=planet_ssb,
            site_vel_km_s=np.asarray(site_vel, dtype=np.float64),
        )
        _, ssb_obs_ls, obs_sun_ls, planets_from_state = tempo2_observatory_chain_vectors(
            state
        )
        if planets is None:
            planets = planets_from_state
    if observatory_earth_km is None or earth_ssb_km is None or earth_ssb_vel_km_s is None:
        raise ValueError(
            "prepare_native_chain_from_simple_result requires tempo2_obs_state "
            "or explicit observatory_earth_km / earth_ssb_km / earth_ssb_vel_km_s"
        )
    model_static = _load_model_static_for_native_chain(params, toas, jug_result)
    return compute_tempo2_native_terms_jax(
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        correction_tt_sec=jnp.asarray(formbats_tt, dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        params=params,
        toas=toas,
        observatory_earth_km=jnp.asarray(observatory_earth_km, dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(earth_ssb_km, dtype=jnp.float64),
        earth_ssb_vel_km_s=jnp.asarray(earth_ssb_vel_km_s, dtype=jnp.float64),
        ephem_path=ephem_path or resolve_tempo2_ephemeris_path(params.get("EPHEM", "DE405")),
        freq_mhz=jnp.asarray(jug_result.get("freq_bary_mhz", td.get("freq_bary_mhz", [])), dtype=jnp.float64),
        tdis1_sec=jnp.asarray(tdis1, dtype=jnp.float64),
        tdis2_sec=jnp.asarray(td["sw_delay_sec"], dtype=np.float64),
        tropospheric_sec=jnp.asarray(td["tropo_delay_sec"], dtype=np.float64),
        dt_emission_sec=jnp.asarray(
            np.asarray(jug_result["dt_sec"], dtype=np.float64), dtype=np.float64
        ),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        utc_to_tdb_sec=jnp.asarray(td.get("utc_to_tdb_sec", 0.0), dtype=np.float64),
        formbats_tt_sec=jnp.asarray(formbats_tt, dtype=np.float64),
        ssb_obs_ls_fixed=jnp.asarray(ssb_obs_ls, dtype=np.float64),
        obs_sun_ls_fixed=jnp.asarray(obs_sun_ls, dtype=np.float64),
        obs_planets_ls_fixed=planets,
        freq_mhz_topocentric=jnp.asarray(freq_topo, dtype=np.float64),
        ne_sw=ne_sw,
        use_model_epoch_batcorr=use_model_epoch_batcorr,
        model_mjd=jnp.asarray(jug_result["model_mjd"], dtype=np.float64),
        prebinary_override_sec=jnp.asarray(prebinary_jug, dtype=np.float64),
        site_vel_km_s=None if site_vel is None else jnp.asarray(site_vel, dtype=jnp.float64),
        model_static=model_static,
    )


def compute_native_tempo2_residual_sec(
    params: dict,
    *,
    static: dict,
    weights,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    track_val: int = -2,
    pulse_numbers=None,
    pn_add=None,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the unified JAX model."""
    del weights
    toas = static.get("toas")
    if not toas:
        raise ValueError(
            "native_chain_static must include TOAs; rebuild GeneralFitSetup with "
            "USE_JAX_TEMPO2_NATIVE_CHAIN enabled"
        )
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "model_mjd": static.get("model_mjd", static["term_diagnostics"].get("sat_mjd")),
        "ssb_obs_pos_ls": static.get("ssb_obs_pos_ls"),
        "obs_sun_pos_ls": static.get("obs_sun_pos_ls"),
        "obs_planet_pos_ls": static.get("obs_planet_pos_ls"),
        "compatibility": "tempo2",
    }
    native = prepare_native_chain_from_simple_result(jug_result, params, toas)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    residual_sec, _, _ = compute_tempo2_native_residuals_jax(
        native_terms=native,
        params=params,
        weights=jnp.asarray(weights, dtype=jnp.float64),
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
        subtract_mean=subtract_mean,
        track_val=track_val,
    )
    return residual_sec
