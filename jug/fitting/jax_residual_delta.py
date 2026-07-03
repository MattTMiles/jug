"""JAX residual deltas and autodiff timing design matrices.

This module owns the differentiable residual path used by
``design_matrix_method="autodiff"``.  The design matrix is the Jacobian of the
same nonlinear residual-delta function used for JAX-native timing likelihoods;
there are no finite-difference perturbations and no hand-written derivative
columns in this path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jug.utils.units import native_derivative_to_fit_column

if TYPE_CHECKING:
    from jug.fitting.optimized_fitter import GeneralFitSetup


def _phase_mean_mode(compatibility: str) -> str:
    mode = str(compatibility).lower()
    if mode in ("tempo2", "tempo2-compatible", "tempo2_compatible"):
        return "unweighted"
    return "weighted"


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
    """Precision-safe JAX residual delta from spin and delay changes.

    JAX has no longdouble, but JUG's host residual path needs longdouble for the
    absolute spin phase.  This function only forms small differences relative to
    the reference state:

    * spin changes are ``(F_k - F_k_ref) * x**(k+1) / (k+1)!``;
    * delay changes use the exact Taylor difference ``phase(x - d) - phase(x)``
      with the current spin coefficients.

    The reference pulse numbers and TZR phase cancel in this local residual
    delta as long as the perturbation stays within the same phase connection.
    """
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


def _build_params_from_delta(
    ref_params: dict[str, object],
    fit_params: Sequence[str],
    ref_theta: np.ndarray,
    delta_theta,
):
    params = dict(ref_params)
    delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
    ref_theta_j = jnp.asarray(ref_theta, dtype=jnp.float64)
    for idx, name in enumerate(fit_params):
        params[str(name).upper()] = ref_theta_j[idx] + delta_theta[idx]
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
    params: dict,
    setup: "GeneralFitSetup",
    *,
    ref_f_terms: Sequence[float],
    phase_mean_mode: str,
    binary_plan=None,
):
    """Residual delta (perturbed - reference) through JUG's JAX forward model."""
    from jug.fitting.forward_delay import compute_total_delay_change

    dt_base_np = (
        setup.dt_sec_ld
        if setup.dt_sec_ld is not None
        else np.array(setup.dt_sec_cached, dtype=np.float64)
    )
    dt_base = jnp.asarray(np.asarray(dt_base_np, dtype=np.float64), dtype=jnp.float64)
    weights = jnp.asarray(setup.weights, dtype=jnp.float64)
    delay_change = compute_total_delay_change(
        params,
        setup,
        xp=jnp,
        binary_plan=binary_plan,
    )

    f_terms = _spin_terms_from_params(params)
    return _phase_residual_delta_jax(
        dt_base,
        delay_change,
        ref_f_terms,
        f_terms,
        weights,
        mean_mode=phase_mean_mode,
        f0=_param_scalar(params, "F0", f_terms[0]),
    )


def make_residual_delta_jax_fn(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object] | None = None,
    ref_theta: np.ndarray | None = None,
    phase_mean_mode: str | None = None,
):
    """Return ``f(delta_theta) -> residual_delta`` for a frozen fit setup."""
    from jug.fitting.binary_delay_plan import resolve_binary_structure
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
    phase_mean_mode = phase_mean_mode or _phase_mean_mode(setup.compatibility)
    binary_plan = resolve_binary_structure(
        ref_params, fit_params, obs_pos_ls=getattr(setup, "ssb_obs_pos_ls", None)
    )

    @jax.jit
    def _fn(delta_theta):
        params = _build_params_from_delta(ref_params, fit_params, ref_theta, delta_theta)
        return _compute_residual_delta_jax(
            params,
            setup,
            ref_f_terms=ref_f_terms,
            phase_mean_mode=phase_mean_mode,
            binary_plan=binary_plan,
        )

    return _fn


def compute_autodiff_designmatrix_from_setup(
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
) -> np.ndarray:
    """Build JUG's public design matrix as ``-jacfwd(residual_delta)(0)``."""
    fit_params = tuple(str(name).upper() for name in fit_params)
    residual_fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    jac_native = np.asarray(jax.jacfwd(residual_fn)(zero), dtype=np.float64)

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
    return np.column_stack(cols) if cols else np.empty((n_toa, 0), dtype=np.float64)

