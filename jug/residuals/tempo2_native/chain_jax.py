"""Production JAX orchestrator for tempo2-native clock/delay/spin chain."""

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
from jug.residuals.tempo2_native.model_jax import (
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
from jug.residuals.tempo2_native_quarantine import (
    TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_NATIVE_GRAPH_FULL,
    TEMPO2_NATIVE_GRAPH_STAGED_BCLT,
    tempo2_native_graph_mode,
)
from jug.residuals.tempo2_native.spin_jax import spin_params_to_jax
from jug.residuals.tempo2_native.types import Tempo2NativeTerms
from jug.utils.timescales import is_tempo2_si_units, parse_timescale

if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import GeneralFitSetup


def _native_chain_mode() -> str:
    """Return the active tempo2-native JAX graph mode."""
    return tempo2_native_graph_mode()


@dataclass(frozen=True)
class NativeFrozenDeltaPack:
    """Host-frozen static inputs for tempo2-native residual evaluation."""

    sat_mjd: Any
    freq_mhz: Any
    dt_emission_sec: Any
    earth_ssb_km: Any
    observatory_earth_km: Any
    site_vel_km_s: Any
    ssb_obs_ls: Any
    obs_sun_ls: Any
    obs_jupiter_ls: Any
    planet_obs_ls: dict[str, Any]
    correction_tt_sec: Any
    correction_tt_tb_sec: Any | None
    einstein_rate: Any
    tropo_sec: Any
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


@dataclass(frozen=True)
class NativeFixedStateNonlinearDeltaPack:
    """Fast nonlinear tempo2 residuals with fixed host/BCLT reference state."""

    sat_mjd: Any
    freq_mhz: Any
    dt_emission_sec: Any
    earth_ssb_km: Any
    observatory_earth_km: Any
    site_vel_km_s: Any
    ssb_obs_ls: Any
    obs_sun_ls: Any
    obs_jupiter_ls: Any
    planet_obs_ls: dict[str, Any]
    correction_tt_sec: Any
    correction_tt_tb_sec: Any
    einstein_rate: Any
    tropo_sec: Any
    dt_ssb_ref_sec: Any
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


@dataclass(frozen=True)
class NativeDeltaPack:
    """Prepacked static inputs for tempo2-native full-chain residual evaluation."""

    sat_mjd: Any
    freq_mhz: Any
    dt_emission_sec: Any
    obs_itrf_km: Any
    spk_packed: Any
    eop_packed: Any
    chain_mjd_tables: tuple[Any, ...]
    chain_offset_tables: tuple[Any, ...]
    bipm_mjd: Any
    bipm_offset: Any
    ifte_records: Any
    ifte_start_jd: Any
    ifte_end_jd: Any
    ifte_step_jd: Any
    ifte_coef_offset: int
    ifte_ncf: int
    ifte_na: int
    ne_sw: float
    correct_troposphere: bool
    obs_site_latitude_rad: float
    obs_site_longitude_rad: float
    obs_site_height_m: float
    obs_site_pressure_mbar: float
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
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
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
    addsat_sec=None,
    emission_phase5=None,
    emission_nphase=None,
):
    """Return residual seconds, pulse numbers, and native terms for tempo2 mode."""
    from jug.residuals.tempo2_native.spin_jax import (
        apply_addsat_track2_frac_phase_jax,
        compute_tempo2_phase5_jax,
        fortran_nlong_jax,
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
        if addsat_sec is not None and emission_phase5 is not None and emission_nphase is not None:
            frac = apply_addsat_track2_frac_phase_jax(
                frac,
                emission_phase5,
                emission_nphase,
                addsat_sec,
                f_terms[0],
            )
    else:
        pulse = jnp.zeros_like(phase5)
        frac = phase5 - jnp.trunc(phase5)
        if addsat_sec is not None:
            addsat = jnp.asarray(addsat_sec, dtype=jnp.float64)
            if bool(jnp.any(addsat != 0.0)):
                addsat_turns = f_terms[0] * addsat
                addsat_int = fortran_nlong_jax(addsat_turns).astype(jnp.float64)
                frac = frac + jnp.where(
                    addsat != 0.0, addsat_turns - addsat_int, 0.0
                )
    residual_sec = frac / f_terms[0]
    if subtract_mean:
        if mean_mode == "weighted":
            w = jnp.asarray(weights, dtype=jnp.float64)
            residual_sec = residual_sec - jnp.sum(residual_sec * w) / jnp.sum(w)
        else:
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
    """Spin/track-only residual from precomputed delay terms (diagnostics helper)."""
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


def prepare_native_chain_from_simple_result(
    jug_result: dict,
    params: dict,
    toas: list[Any],
) -> Tempo2NativeTerms:
    """Build native terms for dev_oracle / diagnostics.

    Dispatches by ``JUG_TEMPO2_NATIVE_GRAPH_MODE``:
    ``full``, ``fixed_state_nonlinear``, or ``staged_bclt`` (default).
    """
    from jug.residuals.diagnostic_conventions import resolve_ne_sw_cm3
    from jug.residuals.engine_conventions import resolve_engine_profile

    td = jug_result["term_diagnostics"]
    profile = resolve_engine_profile(params, jug_result.get("compatibility", "tempo2"))
    ne_sw = resolve_ne_sw_cm3(params, profile)
    freq_topo = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    model_static = _load_model_static_for_native_chain(params, toas, jug_result)

    if _native_chain_mode() == TEMPO2_NATIVE_GRAPH_FULL:
        return compute_tempo2_native_terms_jax(
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

    mode = _native_chain_mode()
    if mode == TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR:
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
        correction_tt_tb_sec=np.asarray(td["correction_tt_tb_sec"], dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        use_native_ecliptic=bool(params.get("_ecliptic_coords", False)),
    )
    return terms


def build_native_delta_pack(setup: "GeneralFitSetup") -> NativeDeltaPack | None:
    """Build JAX-static cache for full-chain native residual deltas."""
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "model_mjd": static.get("model_mjd", static["term_diagnostics"].get("sat_mjd")),
        "ssb_obs_pos_ls": static.get("ssb_obs_pos_ls"),
        "obs_sun_pos_ls": static.get("obs_sun_pos_ls"),
        "obs_planet_pos_ls": static.get("obs_planet_pos_ls"),
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    units = parse_timescale(params)
    tropo = model_static.tropo_packed
    jump = getattr(setup, "jump_phase", None)
    tzr = getattr(setup, "tzr_phase", None)
    return NativeDeltaPack(
        sat_mjd=jnp.asarray(td["sat_mjd"], dtype=jnp.float64),
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(static["dt_sec"], dtype=jnp.float64),
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
        ne_sw=float(model_static.ne_sw),
        correct_troposphere=bool(model_static.correct_troposphere),
        obs_site_latitude_rad=(
            float(tropo.latitude_rad) if tropo is not None else 0.0
        ),
        obs_site_longitude_rad=(
            float(tropo.longitude_rad) if tropo is not None else 0.0
        ),
        obs_site_height_m=float(tropo.height_m) if tropo is not None else 0.0,
        obs_site_pressure_mbar=(
            float(tropo.pressure_mbar) if tropo is not None else 101.325
        ),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
    )


def build_native_frozen_delta_pack(
    setup: "GeneralFitSetup",
) -> NativeFrozenDeltaPack | None:
    """Build host-frozen cache for tempo2-native residual deltas (default path)."""
    from jug.residuals.tempo2_native.model_jax import tempo2_einstein_rate_host

    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    frozen = host_frozen_vectors_from_tempo2_obs_state(td)
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt_pre = np.asarray(
        jax.device_get(
            compute_tempo2_get_correction_tt_jax(
                jnp.asarray(sat, dtype=jnp.float64),
                chain_mjd_tables=tuple(
                    jnp.asarray(t, dtype=jnp.float64) for t in model_static.chain_mjd_tables
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
    site_mjd = sat + tt_pre / SECS_PER_DAY
    einstein = tempo2_einstein_rate_host(site_mjd, params)
    tropo = np.asarray(td.get("tropo_delay_sec", 0.0), dtype=np.float64)
    if tropo.ndim == 0:
        tropo = np.full(len(sat), float(tropo), dtype=np.float64)
    units = parse_timescale(params)
    jump = getattr(setup, "jump_phase", None)
    tzr = getattr(setup, "tzr_phase", None)
    planet_obs_ls = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in frozen["planet_obs_ls"].items()
    }
    return NativeFrozenDeltaPack(
        sat_mjd=jnp.asarray(sat, dtype=jnp.float64),
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(static["dt_sec"], dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(frozen["earth_ssb_km"], dtype=jnp.float64),
        observatory_earth_km=jnp.asarray(frozen["observatory_earth_km"], dtype=jnp.float64),
        site_vel_km_s=jnp.asarray(frozen["site_vel_km_s"], dtype=jnp.float64),
        ssb_obs_ls=jnp.asarray(frozen["ssb_obs_ls"], dtype=jnp.float64),
        obs_sun_ls=jnp.asarray(frozen["obs_sun_ls"], dtype=jnp.float64),
        obs_jupiter_ls=jnp.asarray(frozen["obs_jupiter_ls"], dtype=jnp.float64),
        planet_obs_ls=planet_obs_ls,
        correction_tt_sec=jnp.asarray(tt_pre, dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        einstein_rate=jnp.asarray(einstein, dtype=jnp.float64),
        tropo_sec=jnp.asarray(tropo, dtype=jnp.float64),
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
    )


def _resolve_dt_ssb_ref_sec(
    td: dict,
    *,
    params: dict,
    sat_mjd: np.ndarray,
    freq_mhz: np.ndarray,
    dt_emission_sec: np.ndarray,
    tropo_sec: np.ndarray,
    frozen: dict,
    model_static: Tempo2ModelStatic,
    ne_sw: float,
    use_native_ecliptic: bool,
) -> np.ndarray:
    """Resolve reference BCLT ``dt_ssb`` for fixed-state nonlinear mode."""
    native_terms = td.get("tempo2_native_terms") or {}
    for candidate in (
        td.get("bclt_dt_ssb_sec"),
        td.get("dt_ssb_sec"),
        native_terms.get("dt_ssb_sec"),
    ):
        if candidate is not None:
            return np.asarray(candidate, dtype=np.float64)
    ref_terms, _ = run_tempo2_toa_model_with_fixed_ifte_geometry(
        params=params,
        sat_mjd=np.asarray(sat_mjd, dtype=np.float64),
        freq_mhz=np.asarray(freq_mhz, dtype=np.float64),
        dt_emission_sec=np.asarray(dt_emission_sec, dtype=np.float64),
        tropo_sec=np.asarray(tropo_sec, dtype=np.float64),
        ssb_obs_ls=frozen["ssb_obs_ls"],
        obs_sun_ls=frozen["obs_sun_ls"],
        obs_jupiter_ls=frozen["obs_jupiter_ls"],
        obs_planets_ls=frozen["planet_obs_ls"],
        earth_ssb_km=frozen["earth_ssb_km"],
        observatory_earth_km=frozen["observatory_earth_km"][:, :3],
        site_vel_km_s=frozen["site_vel_km_s"],
        correction_tt_tb_sec=np.asarray(td["correction_tt_tb_sec"], dtype=np.float64),
        model_static=model_static,
        ne_sw=float(ne_sw),
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        use_native_ecliptic=use_native_ecliptic,
    )
    return np.asarray(ref_terms.dt_ssb_sec, dtype=np.float64)


def _build_fixed_state_pack_from_host(
    *,
    jug_result: dict,
    params: dict,
    toas: list[Any],
    model_static: Tempo2ModelStatic,
    frozen: dict,
    tropo: np.ndarray,
    dt_ssb_ref_sec: np.ndarray,
) -> NativeFixedStateNonlinearDeltaPack:
    """Assemble a fixed-state pack from host residual-cache inputs."""
    from jug.residuals.tempo2_native.model_jax import tempo2_einstein_rate_host

    td = jug_result["term_diagnostics"]
    sat = np.asarray(td["sat_mjd"], dtype=np.float64)
    tt_pre = np.asarray(
        jax.device_get(
            compute_tempo2_get_correction_tt_jax(
                jnp.asarray(sat, dtype=jnp.float64),
                chain_mjd_tables=tuple(
                    jnp.asarray(t, dtype=jnp.float64) for t in model_static.chain_mjd_tables
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
    site_mjd = sat + tt_pre / SECS_PER_DAY
    einstein = tempo2_einstein_rate_host(site_mjd, params)
    units = parse_timescale(params)
    jump = model_static.jump_phase
    tzr = model_static.tzr_phase
    planet_obs_ls = {
        k: jnp.asarray(v, dtype=jnp.float64) for k, v in frozen["planet_obs_ls"].items()
    }
    return NativeFixedStateNonlinearDeltaPack(
        sat_mjd=jnp.asarray(sat, dtype=jnp.float64),
        freq_mhz=jnp.asarray([t.freq_mhz for t in toas], dtype=jnp.float64),
        dt_emission_sec=jnp.asarray(jug_result["dt_sec"], dtype=jnp.float64),
        earth_ssb_km=jnp.asarray(frozen["earth_ssb_km"], dtype=jnp.float64),
        observatory_earth_km=jnp.asarray(frozen["observatory_earth_km"], dtype=jnp.float64),
        site_vel_km_s=jnp.asarray(frozen["site_vel_km_s"], dtype=jnp.float64),
        ssb_obs_ls=jnp.asarray(frozen["ssb_obs_ls"], dtype=jnp.float64),
        obs_sun_ls=jnp.asarray(frozen["obs_sun_ls"], dtype=jnp.float64),
        obs_jupiter_ls=jnp.asarray(frozen["obs_jupiter_ls"], dtype=jnp.float64),
        planet_obs_ls=planet_obs_ls,
        correction_tt_sec=jnp.asarray(tt_pre, dtype=jnp.float64),
        correction_tt_tb_sec=jnp.asarray(td["correction_tt_tb_sec"], dtype=jnp.float64),
        einstein_rate=jnp.asarray(einstein, dtype=jnp.float64),
        tropo_sec=jnp.asarray(tropo, dtype=jnp.float64),
        dt_ssb_ref_sec=jnp.asarray(dt_ssb_ref_sec, dtype=np.float64),
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
        dm_epoch=float(params.get("DMEPOCH", params["PEPOCH"])),
        dm_coeffs_ref=_dm_coeffs_from_params(params),
        posepoch_mjd=float(params.get("POSEPOCH", params["PEPOCH"])),
        shk_posepoch=float(params.get("POSEPOCH", params["PEPOCH"])),
        pmrv_rad_century=float(pmrv_rad_per_century(float(params.get("PMRV", 0.0)))),
        dilate_freq=bool(tempo2_dilate_freq_enabled(params)),
        si_units=bool(is_tempo2_si_units(units)),
        units_tdb=units == "TDB",
        planet_shapiro_enabled=bool(model_static.planet_shapiro_enabled),
        track_val=int(model_static.track_val),
        subtract_mean=True,
        dshk=float(params.get("DSHK", 0.0)) if "DSHK" in params else 0.0,
        jump_phase=None if jump is None else jnp.asarray(jump, dtype=jnp.float64),
        tzr_phase=None if tzr is None else jnp.asarray(tzr, dtype=jnp.float64),
        pulse_numbers=(
            None
            if model_static.pulse_numbers is None
            else jnp.asarray(model_static.pulse_numbers, dtype=jnp.int64)
        ),
        pn_add=(
            None
            if model_static.pn_add is None
            else jnp.asarray(model_static.pn_add, dtype=jnp.int64)
        ),
    )


def _native_fixed_state_terms_from_pack(
    params: dict,
    pack: NativeFixedStateNonlinearDeltaPack,
) -> tuple[Tempo2NativeTerms, jnp.ndarray]:
    pos, vel, acc = build_tempo2_pulsar_vectors(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms, pepoch = spin_params_to_jax(params)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    terms, residual_sec = compute_tempo2_toa_model_fixed_state_nonlinear_jax(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=jnp.asarray(pos, dtype=jnp.float64),
        vel_pulsar=jnp.asarray(vel, dtype=jnp.float64),
        acc_pulsar=jnp.asarray(acc, dtype=jnp.float64),
        tropo_sec=pack.tropo_sec,
        dt_emission_sec=pack.dt_emission_sec,
        earth_ssb_km=pack.earth_ssb_km,
        observatory_earth_km=pack.observatory_earth_km,
        site_vel_km_s=pack.site_vel_km_s,
        ssb_obs_ls=pack.ssb_obs_ls,
        obs_sun_ls=pack.obs_sun_ls,
        obs_jupiter_ls=pack.obs_jupiter_ls,
        planet_obs_ls=pack.planet_obs_ls,
        correction_tt_sec_pre=pack.correction_tt_sec,
        correction_tt_tb_sec_pre=pack.correction_tt_tb_sec,
        einstein_rate=pack.einstein_rate,
        dt_ssb_ref_sec=pack.dt_ssb_ref_sec,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        ne_sw=pack.ne_sw,
        posepoch_mjd=pack.posepoch_mjd,
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
    )
    return terms, residual_sec


def build_native_fixed_state_nonlinear_delta_pack(
    setup: "GeneralFitSetup",
) -> NativeFixedStateNonlinearDeltaPack | None:
    """Build fixed-state nonlinear cache for tempo2-native residual deltas."""
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    params = setup.params
    jug_result = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "compatibility": setup.compatibility,
    }
    pulse_numbers, pn_add = track2_pulse_arrays_from_toas(toas, params)
    model_static = _load_model_static_for_native_chain(
        params,
        toas,
        jug_result,
        pulse_numbers=pulse_numbers,
        pn_add=pn_add,
        jump_phase=getattr(setup, "jump_phase", None),
        tzr_phase=getattr(setup, "tzr_phase", None),
        track_val=int(params.get("TRACK", -2)) if params.get("TRACK") is not None else -2,
    )
    td = static["term_diagnostics"]
    frozen = host_frozen_vectors_from_tempo2_obs_state(td)
    tropo = np.asarray(td.get("tropo_delay_sec", 0.0), dtype=np.float64)
    if tropo.ndim == 0:
        tropo = np.full(len(td["sat_mjd"]), float(tropo), dtype=np.float64)
    dt_ssb_ref = _resolve_dt_ssb_ref_sec(
        td,
        params=params,
        sat_mjd=np.asarray(td["sat_mjd"], dtype=np.float64),
        freq_mhz=np.array([t.freq_mhz for t in toas], dtype=np.float64),
        dt_emission_sec=np.asarray(static["dt_sec"], dtype=np.float64),
        tropo_sec=tropo,
        frozen=frozen,
        model_static=model_static,
        ne_sw=float(model_static.ne_sw),
        use_native_ecliptic=bool(model_static.use_native_ecliptic),
    )
    return _build_fixed_state_pack_from_host(
        jug_result=jug_result,
        params=params,
        toas=toas,
        model_static=model_static,
        frozen=frozen,
        tropo=tropo,
        dt_ssb_ref_sec=dt_ssb_ref,
    )


def build_native_delta_pack_for_setup(
    setup: "GeneralFitSetup",
) -> NativeDeltaPack | NativeFrozenDeltaPack | NativeFixedStateNonlinearDeltaPack | None:
    """Select native delta pack by ``JUG_TEMPO2_NATIVE_GRAPH_MODE``."""
    mode = _native_chain_mode()
    if mode == TEMPO2_NATIVE_GRAPH_FULL:
        return build_native_delta_pack(setup)
    if mode == TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR:
        return build_native_fixed_state_nonlinear_delta_pack(setup)
    if mode == TEMPO2_NATIVE_GRAPH_STAGED_BCLT:
        return build_native_frozen_delta_pack(setup)
    raise ValueError(f"Unknown tempo2 native graph mode: {mode!r}")


def compute_native_fixed_state_nonlinear_residual_sec_jax(
    params: dict,
    pack: NativeFixedStateNonlinearDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the fixed-state nonlinear tail."""
    pos, vel, acc = pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms = _spin_f_terms_jax(params)
    pepoch = jnp.asarray(_param_scalar_jax(params, "PEPOCH"), dtype=jnp.float64)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    _, residual_sec = compute_tempo2_toa_model_fixed_state_nonlinear_jax(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=pos,
        vel_pulsar=vel,
        acc_pulsar=acc,
        tropo_sec=pack.tropo_sec,
        dt_emission_sec=pack.dt_emission_sec,
        earth_ssb_km=pack.earth_ssb_km,
        observatory_earth_km=pack.observatory_earth_km,
        site_vel_km_s=pack.site_vel_km_s,
        ssb_obs_ls=pack.ssb_obs_ls,
        obs_sun_ls=pack.obs_sun_ls,
        obs_jupiter_ls=pack.obs_jupiter_ls,
        planet_obs_ls=pack.planet_obs_ls,
        correction_tt_sec_pre=pack.correction_tt_sec,
        correction_tt_tb_sec_pre=pack.correction_tt_tb_sec,
        einstein_rate=pack.einstein_rate,
        dt_ssb_ref_sec=pack.dt_ssb_ref_sec,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        ne_sw=pack.ne_sw,
        posepoch_mjd=pack.posepoch_mjd,
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
    )
    return residual_sec


def compute_native_fixed_state_nonlinear_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeFixedStateNonlinearDeltaPack,
) -> jnp.ndarray:
    """Fixed-state nonlinear residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    res_ref = compute_native_fixed_state_nonlinear_residual_sec_jax(params_ref, pack)
    res_pert = compute_native_fixed_state_nonlinear_residual_sec_jax(params_pert, pack)
    delta = res_pert - res_ref
    if pack.subtract_mean:
        delta = delta - jnp.mean(delta)
    return delta


def compute_native_frozen_residual_sec_jax(
    params: dict,
    pack: NativeFrozenDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the host-frozen staging tail."""
    pos, vel, acc = pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms = _spin_f_terms_jax(params)
    pepoch = jnp.asarray(_param_scalar_jax(params, "PEPOCH"), dtype=jnp.float64)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    _, residual_sec = compute_tempo2_toa_model_staging_with_host_inputs_jax(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=pos,
        vel_pulsar=vel,
        acc_pulsar=acc,
        tropo_sec=pack.tropo_sec,
        dt_emission_sec=pack.dt_emission_sec,
        earth_ssb_km=pack.earth_ssb_km,
        observatory_earth_km=pack.observatory_earth_km,
        site_vel_km_s=pack.site_vel_km_s,
        ssb_obs_ls=pack.ssb_obs_ls,
        obs_sun_ls=pack.obs_sun_ls,
        obs_jupiter_ls=pack.obs_jupiter_ls,
        planet_obs_ls=pack.planet_obs_ls,
        correction_tt_sec_pre=pack.correction_tt_sec,
        correction_tt_tb_sec_pre=pack.correction_tt_tb_sec,
        einstein_rate=pack.einstein_rate,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        ne_sw=pack.ne_sw,
        posepoch_mjd=pack.posepoch_mjd,
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
    )
    return residual_sec


def compute_native_frozen_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeFrozenDeltaPack,
) -> jnp.ndarray:
    """Host-frozen residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    res_ref = compute_native_frozen_residual_sec_jax(params_ref, pack)
    res_pert = compute_native_frozen_residual_sec_jax(params_pert, pack)
    delta = res_pert - res_ref
    if pack.subtract_mean:
        delta = delta - jnp.mean(delta)
    return delta


def compute_native_full_chain_residual_sec_jax(
    params: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Recompute tempo2-native residuals through ``compute_tempo2_toa_model_jax``."""
    pos, vel, acc = pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms = _spin_f_terms_jax(params)
    pepoch = jnp.asarray(_param_scalar_jax(params, "PEPOCH"), dtype=jnp.float64)
    dm_vals = compute_dm_vals_jax(
        pack.sat_mjd, dm_epoch=pack.dm_epoch, dm_coeffs=_dm_coeffs_jax(params)
    )
    _, residual_sec = compute_tempo2_toa_model_jax(
        sat_mjd=pack.sat_mjd,
        freq_mhz=pack.freq_mhz,
        params_f_terms=f_terms,
        params_pepoch=pepoch,
        pos_pulsar=pos,
        vel_pulsar=vel,
        acc_pulsar=acc,
        obs_itrf_km=pack.obs_itrf_km,
        spk_packed=pack.spk_packed,
        eop_packed=pack.eop_packed,
        dm_vals=dm_vals,
        dm_epoch=pack.dm_epoch,
        dm_coeffs=pack.dm_coeffs_ref,
        dt_emission_sec=pack.dt_emission_sec,
        chain_mjd_tables=pack.chain_mjd_tables,
        chain_offset_tables=pack.chain_offset_tables,
        bipm_mjd=pack.bipm_mjd,
        bipm_offset=pack.bipm_offset,
        ifte_records=pack.ifte_records,
        ifte_start_jd=pack.ifte_start_jd,
        ifte_end_jd=pack.ifte_end_jd,
        ifte_step_jd=pack.ifte_step_jd,
        ifte_coef_offset=pack.ifte_coef_offset,
        ifte_ncf=pack.ifte_ncf,
        ifte_na=pack.ifte_na,
        ne_sw=pack.ne_sw,
        obs_site_latitude_rad=pack.obs_site_latitude_rad,
        obs_site_longitude_rad=pack.obs_site_longitude_rad,
        obs_site_height_m=pack.obs_site_height_m,
        obs_site_pressure_mbar=pack.obs_site_pressure_mbar,
        posepoch_mjd=pack.posepoch_mjd,
        parallax_mas=jnp.asarray(_param_scalar_jax(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar_jax(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar_jax(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
        correct_troposphere=pack.correct_troposphere,
    )
    return residual_sec


def compute_native_full_chain_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
) -> jnp.ndarray:
    """Full native-chain residual delta: ``res(θ+Δθ) − res(θ)`` with mean on delta."""
    res_ref = compute_native_full_chain_residual_sec_jax(params_ref, pack)
    res_pert = compute_native_full_chain_residual_sec_jax(params_pert, pack)
    delta = res_pert - res_ref
    if pack.subtract_mean:
        delta = delta - jnp.mean(delta)
    return delta


def compute_native_eval_residuals_jax(
    *,
    params: dict,
    toas: list[Any],
    jug_result: dict,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    mean_mode: str = "unweighted",
    track_val: int = -2,
    weights=None,
    addsat_sec=None,
) -> tuple[jnp.ndarray, jnp.ndarray, Tempo2NativeTerms]:
    """Production residuals: unified in-graph delay chain + spin/track."""
    native = prepare_native_chain_from_simple_result(jug_result, params, toas)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    pn_j = None if pulse_numbers is None else jnp.asarray(pulse_numbers, dtype=jnp.int64)
    pn_add_j = None if pn_add is None else jnp.asarray(pn_add, dtype=jnp.int64)
    addsat_j = None if addsat_sec is None else jnp.asarray(addsat_sec, dtype=jnp.float64)
    emission_p5_j = None
    emission_nph_j = None
    if addsat_j is not None and bool(np.any(np.asarray(addsat_sec) != 0.0)):
        from jug.residuals.tempo2_spin import compute_emission_taylor_phase5_nphase

        dt_host = np.asarray(jug_result.get("dt_sec"), dtype=np.float64)
        p5_host, nph_host = compute_emission_taylor_phase5_nphase(
            dt_host,
            params,
            jump_phase=jump_phase,
            tzr_phase=tzr_phase,
        )
        emission_p5_j = jnp.asarray(p5_host, dtype=jnp.float64)
        emission_nph_j = jnp.asarray(nph_host, dtype=jnp.float64)
    if weights is None:
        weights = jnp.ones(native.sat_mjd.shape[0], dtype=jnp.float64)
    return compute_tempo2_native_residuals_jax(
        native_terms=native,
        params=params,
        weights=jnp.asarray(weights, dtype=jnp.float64),
        pulse_numbers=pn_j,
        pn_add=pn_add_j,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
        subtract_mean=subtract_mean,
        mean_mode=mean_mode,
        track_val=track_val,
        addsat_sec=addsat_j,
        emission_phase5=emission_p5_j,
        emission_nphase=emission_nph_j,
    )
