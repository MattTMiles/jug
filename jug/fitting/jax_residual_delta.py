"""JAX residual deltas and autodiff timing design matrices.

This module owns the differentiable residual path used by
``design_matrix_method="autodiff"``.  The design matrix is the Jacobian of the
same nonlinear residual-delta function used for JAX-native timing likelihoods;
there are no finite-difference perturbations and no hand-written derivative
columns in this path.

**Analytic design matrices** (default WLS) use PINT-style simplified tangents via
``designmatrix_assembly.py``. The test oracle
``compute_simplified_autodiff_designmatrix_from_setup`` differentiates the same
Taylor ``compute_total_delay_change`` + ``_phase_residual_delta_jax`` path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jug.fitting.forward_delay import compute_total_delay_change
from jug.utils.units import native_derivative_to_fit_column

if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import GeneralFitSetup

ECLIPTIC_FIT_TO_INTERNAL = {
    "ELONG": "_ecliptic_lon_deg",
    "LAMBDA": "_ecliptic_lon_deg",
    "ELAT": "_ecliptic_lat_deg",
    "BETA": "_ecliptic_lat_deg",
    "PMELONG": "_ecliptic_pm_lon",
    "PMLAMBDA": "_ecliptic_pm_lon",
    "PMELAT": "_ecliptic_pm_lat",
    "PMBETA": "_ecliptic_pm_lat",
}

_ECLIPTIC_INTERNAL_TO_ELONG_PUBLIC = {
    "_ecliptic_lon_deg": "ELONG",
    "_ecliptic_lat_deg": "ELAT",
    "_ecliptic_pm_lon": "PMELONG",
    "_ecliptic_pm_lat": "PMELAT",
}

_ECLIPTIC_INTERNAL_TO_LAMBDA_PUBLIC = {
    "_ecliptic_lon_deg": "LAMBDA",
    "_ecliptic_lat_deg": "BETA",
    "_ecliptic_pm_lon": "PMLAMBDA",
    "_ecliptic_pm_lat": "PMBETA",
}


def _phase_residual_delta_jax(
    dt_base,
    delay_change,
    ref_f_coeffs,
    f_coeffs,
    weights,
    *,
    mean_mode: str,
    f0,
):
    """Precision-safe JAX residual delta from spin and delay changes."""
    x = jnp.asarray(dt_base, dtype=jnp.float64)
    d = jnp.asarray(delay_change, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)

    n_coeffs = max(len(ref_f_coeffs), len(f_coeffs))
    spin_phase_delta = jnp.zeros_like(x)
    for i in range(n_coeffs):
        ref_coeff = (
            jnp.asarray(ref_f_coeffs[i], dtype=jnp.float64)
            if i < len(ref_f_coeffs)
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        coeff = (
            jnp.asarray(f_coeffs[i], dtype=jnp.float64)
            if i < len(f_coeffs)
            else jnp.asarray(0.0, dtype=jnp.float64)
        )
        spin_phase_delta = spin_phase_delta + (
            (coeff - ref_coeff) * (x ** (i + 1)) / float(math.factorial(i + 1))
        )

    delay_phase_delta = jnp.zeros_like(x)
    for j in range(1, len(f_coeffs) + 1):
        g_j = jnp.zeros_like(x)
        for m in range(0, len(f_coeffs) - (j - 1)):
            coeff = jnp.asarray(f_coeffs[m + j - 1], dtype=jnp.float64)
            g_j = g_j + coeff * (x**m) / float(math.factorial(m))
        delay_phase_delta = delay_phase_delta + (
            ((-d) ** j) / float(math.factorial(j)) * g_j
        )

    residual_delta = (spin_phase_delta + delay_phase_delta) / jnp.asarray(
        f0, dtype=jnp.float64
    )

    if mean_mode == "unweighted":
        residual_delta = residual_delta - jnp.mean(residual_delta)
    else:
        residual_delta = residual_delta - jnp.sum(residual_delta * weights) / jnp.sum(
            weights
        )
    return residual_delta


def _reference_param_value(params: Mapping[str, object], param: str) -> float:
    """Return a fit parameter value in native numeric storage units."""
    param_upper = param.upper()
    if param_upper in ECLIPTIC_FIT_TO_INTERNAL:
        internal_key = ECLIPTIC_FIT_TO_INTERNAL[param_upper]
        if internal_key in params:
            return float(params[internal_key])
        public_fallback = {
            "_ecliptic_lon_deg": ("ELONG", "LAMBDA"),
            "_ecliptic_lat_deg": ("ELAT", "BETA"),
            "_ecliptic_pm_lon": ("PMELONG", "PMLAMBDA"),
            "_ecliptic_pm_lat": ("PMELAT", "PMBETA"),
        }[internal_key]
        for candidate in public_fallback:
            if candidate in params:
                return float(params[candidate])
    key = param_upper if param_upper in params else param
    if key not in params:
        for candidate in (param, param_upper):
            if candidate in params:
                key = candidate
                break
        else:
            return 0.0
    value = params[key]
    if param_upper == "RAJ" and isinstance(value, str):
        from jug.io.par_reader import parse_ra

        return float(parse_ra(value))
    if param_upper == "DECJ" and isinstance(value, str):
        from jug.io.par_reader import parse_dec

        return float(parse_dec(value))
    return float(value)


def _normalize_ref_params(params: Mapping[str, object]) -> dict[str, object]:
    """Return params with string RAJ/DECJ converted to radians."""
    normalized = dict(params)
    for key in ("RAJ", "DECJ"):
        if key in normalized and isinstance(normalized[key], str):
            normalized[key] = _reference_param_value(normalized, key)
    return normalized


def _ecliptic_public_key(internal_key: str, native_family: str) -> str:
    if native_family == "lambda":
        return _ECLIPTIC_INTERNAL_TO_LAMBDA_PUBLIC[internal_key]
    return _ECLIPTIC_INTERNAL_TO_ELONG_PUBLIC[internal_key]


def _ecliptic_session_metadata(ref_params: Mapping[str, object]) -> tuple[bool, float, tuple[float, float, float, float] | None, str]:
    """Static ecliptic session flags captured before JIT compilation."""
    from jug.io.astrometry_state import native_ecliptic_family
    from jug.io.par_reader import OBLIQUITY_ARCSEC

    if not ref_params.get("_ecliptic_coords"):
        return False, 0.0, None, "elong"

    ecl_frame = str(ref_params.get("_ecliptic_frame", ref_params.get("ECL", "IERS2010"))).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC["IERS2010"])
    obl_rad = float(obl_arcsec * np.pi / (180.0 * 3600.0))
    init = (
        float(ref_params.get("_ecliptic_lon_deg", ref_params.get("ELONG", ref_params.get("LAMBDA", 0.0)))),
        float(ref_params.get("_ecliptic_lat_deg", ref_params.get("ELAT", ref_params.get("BETA", 0.0)))),
        float(ref_params.get("_ecliptic_pm_lon", ref_params.get("PMELONG", ref_params.get("PMLAMBDA", 0.0)))),
        float(ref_params.get("_ecliptic_pm_lat", ref_params.get("PMELAT", ref_params.get("PMBETA", 0.0)))),
    )
    native_family = native_ecliptic_family(ref_params) or "elong"
    return True, obl_rad, init, native_family


def _build_params_from_delta(
    ref_params: dict[str, object],
    fit_params: Sequence[str],
    ref_theta: np.ndarray,
    delta_theta,
    *,
    ecliptic_coords: bool = False,
    obl_rad: float = 0.0,
    ecliptic_init: tuple[float, float, float, float] | None = None,
    native_family: str = "elong",
):
    from jug.fitting.derivatives_astrometry import ecliptic_deg_to_equatorial_rad

    params = dict(ref_params)
    delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
    ref_theta_j = jnp.asarray(ref_theta, dtype=jnp.float64)

    lon_deg = lat_deg = pm_lon = pm_lat = None
    if ecliptic_coords and ecliptic_init is not None:
        lon_deg, lat_deg, pm_lon, pm_lat = (
            jnp.asarray(value, dtype=jnp.float64) for value in ecliptic_init
        )

    for idx, name in enumerate(fit_params):
        param_upper = str(name).upper()
        new_val = ref_theta_j[idx] + delta_theta[idx]
        if ecliptic_coords and param_upper in ECLIPTIC_FIT_TO_INTERNAL:
            internal_key = ECLIPTIC_FIT_TO_INTERNAL[param_upper]
            public_key = _ecliptic_public_key(internal_key, native_family)
            params[internal_key] = new_val
            params[public_key] = new_val
            if internal_key == "_ecliptic_lon_deg":
                lon_deg = new_val
            elif internal_key == "_ecliptic_lat_deg":
                lat_deg = new_val
            elif internal_key == "_ecliptic_pm_lon":
                pm_lon = new_val
            elif internal_key == "_ecliptic_pm_lat":
                pm_lat = new_val
        else:
            params[param_upper] = new_val
            if param_upper == "RAJ":
                params["_raj_rad"] = new_val
            elif param_upper == "DECJ":
                params["_decj_rad"] = new_val

    if ecliptic_coords and lon_deg is not None:
        ra_rad, dec_rad, pmra, pmdec = ecliptic_deg_to_equatorial_rad(
            lon_deg,
            lat_deg,
            pm_lon,
            pm_lat,
            jnp.asarray(obl_rad, dtype=jnp.float64),
            xp=jnp,
        )
        params["_raj_rad"] = ra_rad
        params["_decj_rad"] = dec_rad
        has_pm = jnp.not_equal(pm_lon, 0.0) | jnp.not_equal(pm_lat, 0.0)
        ref_pmra = jnp.asarray(float(ref_params.get("PMRA", 0.0)), dtype=jnp.float64)
        ref_pmdec = jnp.asarray(float(ref_params.get("PMDEC", 0.0)), dtype=jnp.float64)
        params["PMRA"] = jnp.where(has_pm, pmra, ref_pmra)
        params["PMDEC"] = jnp.where(has_pm, pmdec, ref_pmdec)

    return params


def _param_scalar(params: dict, name: str, default: float = 0.0):
    key = name.upper()
    if key in params:
        return params[key]
    return default


def _spin_terms_from_params(params: dict) -> list:
    terms = []
    for i in range(10):
        key = f"F{i}"
        if key in params:
            terms.append(_param_scalar(params, key))
        elif i == 0:
            terms.append(_param_scalar(params, "F0", 1.0))
        else:
            break
    return terms


def _compute_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    setup: "GeneralFitSetup",
    *,
    ref_f_terms: Sequence[float],
    phase_mean_mode: str,
    binary_plan=None,
):
    """Residual delta (perturbed - reference) through JUG's PINT-family JAX model."""
    dt_base_np = (
        setup.dt_sec_ld
        if setup.dt_sec_ld is not None
        else np.array(setup.dt_sec_cached, dtype=np.float64)
    )
    dt_base = jnp.asarray(np.asarray(dt_base_np, dtype=np.float64), dtype=jnp.float64)
    weights = jnp.asarray(setup.weights, dtype=jnp.float64)
    delay_change = compute_total_delay_change(
        params_pert,
        setup,
        xp=jnp,
        binary_plan=binary_plan,
    )

    f_terms = _spin_terms_from_params(params_pert)
    return _phase_residual_delta_jax(
        dt_base,
        delay_change,
        ref_f_terms,
        f_terms,
        weights,
        mean_mode=phase_mean_mode,
        f0=_param_scalar(params_pert, "F0", f_terms[0]),
    )


def _residual_delta_core_for_setup(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object],
    ref_theta: np.ndarray,
    phase_mean_mode: str,
    ref_f_terms: tuple[float, ...],
    binary_plan,
    ecliptic_coords: bool,
    obl_rad: float,
    ecliptic_init: dict,
    native_family: str,
):
    """Un-jitted residual-delta closure shared by residual eval and jacfwd."""

    def core(delta_theta):
        zero = jnp.zeros_like(delta_theta)
        params_ref = _build_params_from_delta(
            ref_params,
            fit_params,
            ref_theta,
            zero,
            ecliptic_coords=ecliptic_coords,
            obl_rad=obl_rad,
            ecliptic_init=ecliptic_init,
            native_family=native_family,
        )
        params_pert = _build_params_from_delta(
            ref_params,
            fit_params,
            ref_theta,
            delta_theta,
            ecliptic_coords=ecliptic_coords,
            obl_rad=obl_rad,
            ecliptic_init=ecliptic_init,
            native_family=native_family,
        )
        return _compute_residual_delta_jax(
            params_ref,
            params_pert,
            setup,
            ref_f_terms=ref_f_terms,
            phase_mean_mode=phase_mean_mode,
            binary_plan=binary_plan,
        )

    return core


def _residual_delta_jax_cache_key(
    setup: "GeneralFitSetup",
    *,
    fit_params: tuple[str, ...],
    ref_theta: np.ndarray,
    ref_f_terms: tuple[float, ...],
    phase_mean_mode: str,
) -> tuple:
    """Hashable key for session-scoped residual/Jacobian JIT bundles."""
    return (
        fit_params,
        tuple(float(x) for x in ref_theta),
        ref_f_terms,
        phase_mean_mode,
        str(setup.compatibility),
    )


def _build_residual_delta_jax_bundle(
    *,
    setup: "GeneralFitSetup",
    fit_params: tuple[str, ...],
    ref_params: Mapping[str, object],
    ref_theta: np.ndarray,
    phase_mean_mode: str,
):
    """Build shared residual core and jitted residual / Jacobian evaluators."""
    from jug.fitting.binary_delay_plan import resolve_binary_structure

    ref_f_terms = tuple(float(x) for x in _spin_terms_from_params(ref_params))
    binary_plan = resolve_binary_structure(
        ref_params, fit_params, obs_pos_ls=getattr(setup, "ssb_obs_pos_ls", None)
    )
    ecliptic_coords, obl_rad, ecliptic_init, native_family = _ecliptic_session_metadata(
        ref_params
    )

    core = _residual_delta_core_for_setup(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
        ref_f_terms=ref_f_terms,
        binary_plan=binary_plan,
        ecliptic_coords=ecliptic_coords,
        obl_rad=obl_rad,
        ecliptic_init=ecliptic_init,
        native_family=native_family,
    )
    return core, jax.jit(core), jax.jit(jax.jacfwd(core))


def _prepare_residual_delta_jax(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object] | None = None,
    ref_theta: np.ndarray | None = None,
    phase_mean_mode: str | None = None,
):
    """Build or reuse session-cached residual core and JIT evaluators."""
    from jug.fitting.forward_delay import _assert_no_epoch_fit_params

    fit_params = tuple(str(name).upper() for name in fit_params)
    _assert_no_epoch_fit_params(fit_params)
    ref_params = _normalize_ref_params(ref_params or setup.params)
    if ref_theta is None:
        ref_theta = np.array(
            [_reference_param_value(ref_params, name) for name in fit_params],
            dtype=np.float64,
        )
    else:
        ref_theta = np.asarray(ref_theta, dtype=np.float64).reshape(-1)
    if ref_theta.shape != (len(fit_params),):
        raise ValueError("ref_theta shape mismatch with fit_params.")

    ref_f_terms = tuple(float(x) for x in _spin_terms_from_params(ref_params))
    phase_mean_mode = phase_mean_mode or "weighted"
    cache_key = _residual_delta_jax_cache_key(
        setup,
        fit_params=fit_params,
        ref_theta=ref_theta,
        ref_f_terms=ref_f_terms,
        phase_mean_mode=phase_mean_mode,
    )
    cache = setup.residual_delta_jax_cache
    if cache is None:
        cache = {}
        setup.residual_delta_jax_cache = cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    bundle = _build_residual_delta_jax_bundle(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
    )
    cache[cache_key] = bundle
    return bundle


def make_residual_delta_jax_fn(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object] | None = None,
    ref_theta: np.ndarray | None = None,
    phase_mean_mode: str | None = None,
):
    """Return ``f(delta_theta) -> residual_delta`` for a frozen fit setup."""
    _, residual_fn, _ = _prepare_residual_delta_jax(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
    )
    return residual_fn


def compute_autodiff_designmatrix_from_setup(
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    *,
    include_offset_column: bool = False,
) -> np.ndarray:
    """Build JUG's public design matrix as ``-jacfwd(residual_delta)(0)``."""
    fit_params = tuple(str(name).upper() for name in fit_params)
    _, _, jac_fn = _prepare_residual_delta_jax(setup=setup, fit_params=fit_params)
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    jac_native = np.asarray(jac_fn(zero), dtype=np.float64)

    cols = []
    for col, param in enumerate(fit_params):
        public_native_col = -jac_native[:, col]
        cols.append(
            np.asarray(
                native_derivative_to_fit_column(param, public_native_col),
                dtype=np.float64,
            )
        )
    n_toa = len(np.asarray(setup.tdb_mjd))
    if include_offset_column:
        offset = np.full((n_toa,), -1.0, dtype=np.float64)
        return np.column_stack([offset] + cols) if cols else offset.reshape(-1, 1)
    return np.column_stack(cols) if cols else np.empty((n_toa, 0), dtype=np.float64)


def compute_simplified_autodiff_designmatrix_from_setup(
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    *,
    include_offset_column: bool = False,
) -> np.ndarray:
    """Jacobian of the PINT-style Taylor residual delta (test oracle for analytic columns)."""
    return compute_autodiff_designmatrix_from_setup(
        setup,
        fit_params,
        include_offset_column=include_offset_column,
    )