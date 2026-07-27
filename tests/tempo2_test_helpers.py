"""Shared helpers for tempo2-native chain dev_oracle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
from jug.utils.constants import SECS_PER_DAY


def load_wsrt167_fixture():
    from tempo2_fixtures import get_tempo2_fixture

    return get_tempo2_fixture("wsrt167")


def session_cached_data_from_jug(
    jug_result: dict,
    toas: list,
) -> dict:
    """Build ``_build_general_fit_setup_from_cache`` payload from one host residual run."""
    toas_mjd = np.array([t.mjd_int + t.mjd_frac for t in toas], dtype=np.float64)
    return {
        "dt_sec": jug_result["dt_sec"],
        "dt_sec_ld": jug_result.get("dt_sec_ld"),
        "tdb_mjd": jug_result["tdb_mjd"],
        "freq_bary_mhz": jug_result["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": np.array([t.error_us for t in toas], dtype=np.float64),
        "toa_flags": [t.flags for t in toas],
        "prebinary_delay_sec": jug_result.get("prebinary_delay_sec"),
        "roemer_shapiro_sec": jug_result.get("roemer_shapiro_sec"),
        "ssb_obs_pos_ls": jug_result.get("ssb_obs_pos_ls"),
        "obs_sun_pos_ls": jug_result.get("obs_sun_pos_ls"),
        "obs_planet_pos_ls": jug_result.get("obs_planet_pos_ls"),
        "sw_geometry_pc": jug_result.get("sw_geometry_pc"),
        "jump_phase": jug_result.get("jump_phase"),
        "tzr_phase": jug_result.get("tzr_phase"),
        "term_diagnostics": jug_result.get("term_diagnostics"),
        "model_mjd": jug_result.get("model_mjd"),
        "toas": toas,
        "tempo2_native": jug_result.get("tempo2_native"),
        "tempo2_jug_options": jug_result.get("tempo2_jug_options"),
    }


def build_fit_setup_from_jug_cache(
    *,
    params: dict,
    session_cached_data: dict,
    fit_params: list[str],
    compatibility: str = "tempo2",
    tempo2_native: str | None = None,
    tempo2_jug_options: dict | None = None,
):
    """Fast ``GeneralFitSetup`` from amortized host residuals (no par/tim re-read)."""
    from jug.fitting.optimized_fitter import _build_general_fit_setup_from_cache

    return _build_general_fit_setup_from_cache(
        session_cached_data,
        params,
        list(fit_params),
        compatibility=compatibility,
        tempo2_native=tempo2_native,
        tempo2_jug_options=tempo2_jug_options,
    )


def compute_native_terms_for_fixture(
    fixture: dict,
    *,
    tempo2_native: str | None = None,
    tempo2_jug_options: dict | None = None,
) -> Any:
    """Build native JAX terms from a tempo2 fixture par/tim pair."""
    from jug.utils.jax_setup import ensure_jax_x64

    ensure_jax_x64()
    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(
        par_path,
        tim_path,
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=True,
        tempo2_native=tempo2_native,
        tempo2_jug_options=tempo2_jug_options,
    )
    if tempo2_native is not None:
        jug["tempo2_native"] = tempo2_native
    if tempo2_jug_options is not None:
        jug["tempo2_jug_options"] = tempo2_jug_options
    return prepare_tempo2_chain_from_simple_result(jug, params, toas)


def build_fit_setup_for_fixture(
    fixture: dict,
    fit_params: list[str],
    *,
    tempo2_native: str | None = "staged_bclt",
    tempo2_jug_options: dict | None = None,
):
    """Build ``GeneralFitSetup`` from one host residual pass on any fixture."""
    par_path = Path(fixture["par_path"])
    tim_path = Path(fixture["tim_path"])
    params = parse_par_file(par_path)
    toas = parse_tim_file_mjds(tim_path)
    jug = compute_residuals_simple(
        par_path,
        tim_path,
        verbose=False,
        compatibility="tempo2",
        tempo2_native=tempo2_native,
        tempo2_jug_options=tempo2_jug_options,
    )
    cache = session_cached_data_from_jug(jug, toas)
    if tempo2_native is not None:
        cache["tempo2_native"] = tempo2_native
    return build_fit_setup_from_jug_cache(
        params=params,
        session_cached_data=cache,
        fit_params=list(fit_params),
        tempo2_native=tempo2_native,
        tempo2_jug_options=tempo2_jug_options,
    )


def compute_native_terms_model_epoch(fixture: dict) -> Any:
    """Alias for the unified JAX path."""
    return compute_native_terms_for_fixture(fixture)


def native_batcorr_days(native_terms) -> np.ndarray:
    import jax

    bat = jax.device_get(native_terms.bat_corr_day + native_terms.bat_corr_day_residual)
    return np.asarray(bat, dtype=np.float64)


def delta_ns(a, b, *, is_mjd: bool = False) -> np.ndarray:
    x = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if is_mjd:
        x = x * SECS_PER_DAY
    return x * 1e9


def rms_ns(a, b, *, is_mjd: bool = False) -> float:
    """RMS difference in nanoseconds."""
    return float(np.sqrt(np.mean(delta_ns(a, b, is_mjd=is_mjd) ** 2)))


def rms_cm(a_ls, b_ls) -> float:
    """RMS vector difference in centimetres (inputs in light-seconds)."""
    from jug.utils.constants import C_KM_S

    diff = np.asarray(a_ls, dtype=np.float64) - np.asarray(b_ls, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))) * C_KM_S * 100)


def residual_jacobian_fit_from_setup(
    setup,
    fit_params,
    *,
    delay_model: str = "native",
):
    """J_fit of the selected residual graph (test helper).

    Mirrors ``_simplified_residual_jacobian_oracle`` but honors ``delay_model``
    so tempo2 native-graph tests can request ``delay_model="native"``.
    """
    import jax.numpy as jnp
    import numpy as np

    from jug.fitting.jax_residual_delta import _prepare_residual_delta_jax
    from jug.utils.units import native_derivative_to_fit_column

    fit_params = tuple(str(name).upper() for name in fit_params)
    _, _, jac_fn = _prepare_residual_delta_jax(
        setup=setup, fit_params=fit_params, delay_model=delay_model
    )
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    jac_native = np.asarray(jac_fn(zero), dtype=np.float64)
    if not fit_params:
        return np.empty((len(np.asarray(setup.tdb_mjd)), 0), dtype=np.float64)
    return np.column_stack(
        [
            np.asarray(
                native_derivative_to_fit_column(param, jac_native[:, col]),
                dtype=np.float64,
            )
            for col, param in enumerate(fit_params)
        ]
    )

