"""Frozen residual model exported from a JUG session.

``export_frozen_residual_model`` snapshots a populated ``TimingSession`` into a
convention-frozen ``FrozenResidualModel``: the jitted residual-delta closure of
the session-selected graph, its jitted Jacobian, the reference point, and the
contribution-local linear-residual transform. Stored arrays are read-only
copies (the dataclass itself is shallow-frozen). It stores no design matrix;
the canonical fitter basis is a different object (see
feature_designmatrix_naming_conventions.md).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import jax.numpy as jnp
import numpy as np

from jug.fitting.jax_residual_delta import (
    _normalize_ref_params,
    _phase_mean_mode,
    _prepare_residual_delta_jax,
    _reference_param_value,
)
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_cache,
    _compute_full_model_residuals,
    _interim_row_tokens,
)
from jug.model.parameter_spec import canonicalize_param_name


class NativeChainStatus(NamedTuple):
    native_chain_static: bool
    tempo2_obs_state: bool


@dataclass(frozen=True)
class FrozenResidualModel:
    """Convention-frozen residual model for one JUG session graph."""

    fit_params: tuple[str, ...]  # canonical requested names
    param_mapping: tuple[tuple[str, str], ...]  # canonical -> engine names
    reference_theta_native: np.ndarray  # native units; read-only copy
    reference_residuals_sec: np.ndarray  # diagnostic metadata (below)
    subtract_tzr: bool  # diagnostic metadata (below)
    compatibility: str
    mean_mode: Literal["weighted", "unweighted"]
    row_tokens: tuple[str, ...]  # session row order (D6)
    _mean_weights: np.ndarray | None  # normalized; None = unweighted
    _residual_delta_jax_fn: Any  # jitted f(delta_native) -> dr
    _residual_jacobian_native_fn: Any  # jitted jacfwd of the same core
    _native_chain_status: NativeChainStatus

    # ``reference_residuals_sec`` and ``subtract_tzr`` are recorded metadata,
    # not consumed by any method here: they let diagnostics and
    # ``compute_full_model_residuals_tempo2_native`` reconstruct absolute
    # residuals (r_abs = reference + delta) and record which residual
    # convention the reference was computed under.

    def residual_delta_jax(self, delta_native):
        return self._residual_delta_jax_fn(delta_native)

    def residual_jacobian_native(self) -> np.ndarray:
        """J = d(residual_delta)/d(delta_native) at the frozen reference."""
        zeros = jnp.zeros((len(self.fit_params),), dtype=jnp.float64)
        return np.asarray(self._residual_jacobian_native_fn(zeros), dtype=np.float64)

    def transform_linear_residual(self, raw_residual_delta):
        """Apply this contribution's centering to one raw linear residual vector.

        Linear; maps zeros to zeros; tracer-safe (jnp); 1-D vectors only —
        apply column-by-column (or vmap) for matrices. NumPy callers wrap the
        result in ``np.asarray``. No whitening, TZR handling, or reordering.
        """
        v = jnp.asarray(raw_residual_delta, dtype=jnp.float64)
        if v.ndim != 1:
            raise ValueError(
                "transform_linear_residual expects a 1-D residual vector; "
                "apply column-wise (or vmap) for matrices."
            )
        if self._mean_weights is None:
            return v - jnp.mean(v)
        w = jnp.asarray(self._mean_weights, dtype=jnp.float64)
        return v - jnp.sum(v * w)

    def verify_native_chain(self) -> None:
        """Raise unless the tempo2-native payload backing this graph is present.

        No-op for PINT compatibility. Replaces downstream inspection of
        ``state.setup.native_chain_static``.
        """
        if not str(self.compatibility).lower().startswith("tempo2"):
            return
        if not self._native_chain_status.native_chain_static:
            raise RuntimeError(
                "tempo2 FrozenResidualModel has no native_chain_static; "
                "re-export from a session whose residual cache includes "
                "term_diagnostics (call compute_residuals first)."
            )
        if not self._native_chain_status.tempo2_obs_state:
            raise RuntimeError(
                "tempo2 FrozenResidualModel payload is missing "
                "term_diagnostics['tempo2_obs_state']."
            )


def export_frozen_residual_model(
    session,
    *,
    fit_params: Sequence[str],
    subtract_tzr: bool = True,
    param_mapping: Mapping[str, str] | None = None,
) -> FrozenResidualModel:
    """Export a convention-frozen residual model from a populated TimingSession."""
    fit_params = tuple(str(name) for name in fit_params)
    if not fit_params:
        raise ValueError("fit_params must be non-empty.")

    compatibility = getattr(session, "compatibility", "pint")
    mapping = dict(param_mapping or {})

    cached = session._cached_result_by_mode.get(subtract_tzr)
    needs_native_payload = str(compatibility).lower().startswith("tempo2")
    if cached is None or "dt_sec" not in cached:
        session.compute_residuals(subtract_tzr=subtract_tzr, force_recompute=False)
        cached = session._cached_result_by_mode.get(subtract_tzr)
    elif needs_native_payload and cached.get("term_diagnostics") is None:
        session.compute_residuals(subtract_tzr=subtract_tzr, force_recompute=True)
        cached = session._cached_result_by_mode.get(subtract_tzr)
    if cached is None or "dt_sec" not in cached:
        raise RuntimeError(
            "TimingSession cache is unavailable; call compute_residuals() first."
        )

    toas_mjd = np.array([toa.mjd_int + toa.mjd_frac for toa in session.toas_data])
    errors_us = np.array([toa.error_us for toa in session.toas_data])
    toa_flags = [toa.flags for toa in session.toas_data]
    session_cached_data = {
        "dt_sec": cached["dt_sec"],
        "dt_sec_ld": cached.get("dt_sec_ld"),
        "tdb_mjd": cached["tdb_mjd"],
        "model_mjd": cached.get("model_mjd"),
        "bbat_mjd": cached.get("bbat_mjd"),
        "engine_conventions": cached.get("engine_conventions"),
        "diagnostic_conventions": cached.get("diagnostic_conventions"),
        "freq_bary_mhz": cached["freq_bary_mhz"],
        "toas_mjd": toas_mjd,
        "errors_us": errors_us,
        "toa_flags": toa_flags,
        "roemer_shapiro_sec": cached.get("roemer_shapiro_sec"),
        "prebinary_delay_sec": cached.get("prebinary_delay_sec"),
        "ssb_obs_pos_ls": cached.get("ssb_obs_pos_ls"),
        "earth_ssb_ls": cached.get("earth_ssb_ls"),
        "observatory_earth_ls": cached.get("observatory_earth_ls"),
        "sw_geometry_pc": cached.get("sw_geometry_pc"),
        "jump_phase": cached.get("jump_phase"),
        "tzr_phase": cached.get("tzr_phase"),
        "term_diagnostics": cached.get("term_diagnostics"),
        "toas": session.toas_data,
        "tempo2_native": getattr(session, "tempo2_native", None),
        "tempo2_jug_options": getattr(session, "tempo2_jug_options", None),
    }

    runtime_fit_params = tuple(
        canonicalize_param_name(mapping.get(name, name)) for name in fit_params
    )

    setup = _build_general_fit_setup_from_cache(
        session_cached_data,
        session.params,
        list(runtime_fit_params),
        compatibility=compatibility,
        tempo2_native=getattr(session, "tempo2_native", None),
        tempo2_jug_options=getattr(session, "tempo2_jug_options", None),
    )

    ref_params = _normalize_ref_params(session.params)
    ref_theta = np.array(
        [
            _reference_param_value(ref_params, mapping.get(name, name))
            for name in fit_params
        ],
        dtype=np.float64,
    )
    reference_residuals_sec, _, _, _ = _compute_full_model_residuals(ref_params, setup)
    reference_theta_native = np.array(ref_theta, dtype=np.float64, copy=True)
    reference_theta_native.setflags(write=False)
    reference_residuals_sec = np.array(
        reference_residuals_sec, dtype=np.float64, copy=True
    )
    reference_residuals_sec.setflags(write=False)

    _, residual_fn, jac_fn = _prepare_residual_delta_jax(
        setup=setup,
        fit_params=tuple(runtime_fit_params),
        ref_params=ref_params,
        ref_theta=ref_theta,
    )

    mean_mode = _phase_mean_mode(setup.compatibility)  # "weighted" | "unweighted"
    if mean_mode == "weighted":
        w = np.asarray(setup.weights, dtype=np.float64)
        mean_weights = w / w.sum()
        mean_weights.setflags(write=False)
    else:
        mean_weights = None

    static = getattr(setup, "native_chain_static", None)
    td = (getattr(static, "term_diagnostics", None) or {}) if static is not None else {}
    native_chain_status = NativeChainStatus(
        native_chain_static=static is not None,
        tempo2_obs_state="tempo2_obs_state" in td,
    )

    row_tokens = _interim_row_tokens(
        [t.mjd_int for t in session.toas_data],
        [t.mjd_frac for t in session.toas_data],
    )

    return FrozenResidualModel(
        fit_params=fit_params,
        param_mapping=tuple(sorted(mapping.items())),
        reference_theta_native=reference_theta_native,
        reference_residuals_sec=reference_residuals_sec,
        subtract_tzr=subtract_tzr,
        compatibility=str(compatibility),
        mean_mode=mean_mode,  # type: ignore[arg-type]
        row_tokens=row_tokens,
        _mean_weights=mean_weights,
        _residual_delta_jax_fn=residual_fn,
        _residual_jacobian_native_fn=jac_fn,
        _native_chain_status=native_chain_status,
    )
