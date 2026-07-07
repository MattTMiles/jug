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
    from jug.residuals.diagnostic_conventions import default_conventions
    from jug.residuals.engine_conventions import _flag_from_par

    compatibility = jug_result.get("compatibility", "tempo2")
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
        observatory_earth_km,
        earth_ssb_km,
        earth_ssb_vel_km_s,
        site_vel_km_s,
        obs_planets_ls_fixed,
        formbats_tt_sec,
    )

    if ssb_obs_ls_fixed is not None or obs_sun_ls_fixed is not None:
        raise ValueError(
            "Unified Phase 4: ssb_obs_ls and obs_sun_ls must be None; "
            "geometry computed in-graph. "
            "For host-precomputed geometry use "
            "compute_tempo2_toa_model_staging_with_host_inputs_jax."
        )
    if model_static is None:
        raise ValueError(
            "compute_tempo2_native_terms_jax requires model_static with "
            "clock, IFTE, and SPK tables"
        )

    if freq_mhz_topocentric is not None:
        freq = np.asarray(freq_mhz_topocentric, dtype=np.float64)
    else:
        freq = np.asarray(freq_mhz, dtype=np.float64)

    terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=freq,
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
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
    use_model_epoch_batcorr: bool = False,
) -> Tempo2NativeTerms:
    """Build native terms through unified in-graph geometry (Phase 4)."""
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    del use_model_epoch_batcorr
    td = jug_result["term_diagnostics"]
    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    ne_sw = resolve_ne_sw_cm3(params, profile)
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    model_static = _load_model_static_for_native_chain(params, toas, jug_result)
    return compute_tempo2_native_terms_jax(
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        correction_tt_sec=jnp.asarray(
            td.get("formbats_correction_tt_sec", td["correction_tt_sec"]), dtype=jnp.float64
        ),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        params=params,
        toas=toas,
        observatory_earth_km=jnp.zeros((len(toas), 3), dtype=jnp.float64),
        earth_ssb_km=jnp.zeros((len(toas), 3), dtype=jnp.float64),
        earth_ssb_vel_km_s=jnp.zeros((len(toas), 3), dtype=jnp.float64),
        ephem_path=model_static.ephem_path,
        freq_mhz=jnp.asarray(
            jug_result.get("freq_bary_mhz", td.get("freq_bary_mhz", [])), dtype=jnp.float64
        ),
        tdis1_sec=jnp.asarray(td["dm_delay_sec"], dtype=jnp.float64),
        tdis2_sec=jnp.asarray(td["sw_delay_sec"], dtype=np.float64),
        tropospheric_sec=jnp.asarray(td["tropo_delay_sec"], dtype=np.float64),
        dt_emission_sec=jnp.asarray(jug_result["dt_sec"], dtype=np.float64),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
        freq_mhz_topocentric=jnp.asarray(freq_topo, dtype=jnp.float64),
        ne_sw=ne_sw,
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
