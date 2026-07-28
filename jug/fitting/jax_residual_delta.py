"""JAX residual deltas for nonlinear timing likelihoods.

``make_residual_delta_jax_fn`` returns the gauge-free residual-delta function
for the session-selected graph (simplified PINT-style Taylor, or the native
tempo2 graph named by ``setup.tempo2_native``). Residual Jacobians are its
``jacfwd``; see ``jug.fitting.residual_model``. The analytic fitter basis
lives in ``designmatrix_assembly`` and is a different object (see
feature_phase_gauge.md / feature_designmatrix_naming_conventions.md).

For ``compatibility="tempo2"``, the native residual path recomputes
``residual_sec(θ+Δθ) − residual_sec(θ)`` through the tempo2-native JAX graph
selected by ``setup.tempo2_native`` (default ``fixed_state_stripped``). Set
``tempo2_native="full"`` only to differentiate through the unified in-graph
model; expect multi-minute JIT compile on first call.

**Host vs fit model split:** production host residuals (``compute_residuals_simple``)
use Taylor emission spin for TRACK −2 / absent TRACK; this module uses native
``phase5@bbat`` in JAX. See ``jug.residuals.tempo2.host`` routing contract.

**``nonlinear_params`` (caller-declared linearization):** ``None`` keeps the
native residual_delta graph above. ``"binary"`` / ``"binary+"`` select the
hybrid formula Δr = J_lin @ δ_lin + phase(Δbinary at frozen t_pre) with
gauge-free analytic bake-in J = -M (native-delta units). ``"binary"`` freezes
all astrometry including PX inside the binary/Kopeikin call; ``"binary+"``
keeps PX live in the plan (PM/sky remain frozen). When the residual-delta
axis list has no binary names, both hybrid modes degenerate to pure
``J @ δ`` — live Kopeikin-PX for ``"binary+"`` therefore requires at least
one binary axis in that list (typical PTA DDK sampling). JUG validates and
executes; it does not auto-select a mode from the δ-axis list. No open
parameter lists. See ``feature_hybrid_linear_binary.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jug.fitting.forward_delay import compute_side_delay_change, compute_total_delay_change
from jug.residuals.engine_conventions import normalize_compatibility_mode
from jug.residuals.gauge import ReferenceGauge, _gauge_offset_values, reconstruct_absolute_residuals
from jug.residuals.tempo2.common import NativeDeltaPack
from jug.residuals.tempo2.delta_pack import build_delta_pack_for_setup
from jug.residuals.tempo2.terms import compute_bbat_delay_change_sec_jax
from jug.utils.constants import SECS_PER_DAY
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


def _phase_mean_mode(compatibility: str) -> str:
    from jug.residuals.engine_conventions import normalize_compatibility_mode

    if normalize_compatibility_mode(compatibility) == "tempo2":
        return "unweighted"
    return "weighted"


def _host_reference_gauge(
    *,
    compatibility: str,
    params: Mapping[str, object],
    model_mjd,
    weights,
    subtract_tzr: bool = True,
    tzr_apply_mode: str | None = None,
    tzr_offset_sec: float | None = None,
) -> ReferenceGauge:
    """Build the gauge descriptor matching the host residual vector.

    When tempo2 ``REFPHS TZR`` selects ``post_wrap``, the host applies a
    constant (spin-dependent) offset; the descriptor records
    ``mode="constant"`` so absolute-residual reconstruction refuses rather
    than silently mean-projecting a constant-gauged reference.
    """
    from jug.residuals.tzr_geometry import resolve_tempo2_tzr_apply_mode

    if tzr_apply_mode is None:
        if normalize_compatibility_mode(str(compatibility)) == "tempo2":
            tzr_apply_mode = resolve_tempo2_tzr_apply_mode(
                dict(params),
                np.asarray(model_mjd, dtype=np.float64),
                subtract_tzr=subtract_tzr,
            )
        else:
            tzr_apply_mode = "pre_wrap" if subtract_tzr else "none"

    if tzr_apply_mode == "post_wrap":
        # Offset is provenance for the descriptor; reconstruction refuses for
        # constant gauges regardless of the numeric value (theta-dependent TZR).
        offset = 0.0 if tzr_offset_sec is None else float(tzr_offset_sec)
        return ReferenceGauge(mode="constant", offset_sec=offset)

    mean_mode = _phase_mean_mode(compatibility)
    if mean_mode == "weighted":
        w = np.asarray(weights, dtype=np.float64)
        return ReferenceGauge(mode="mean", weights=(w / w.sum()))
    return ReferenceGauge(mode="mean", weights=None)


def _phase_residual_delta_jax(
    dt_base,
    delay_change,
    ref_f_coeffs,
    f_coeffs,
    weights,
    *,
    mean_mode: str,
    f0,
    spin_term_deltas: Sequence | None = None,
):
    """Precision-safe JAX residual delta from spin and delay changes.

    JAX has no longdouble, but JUG's host residual path needs longdouble for the
    absolute spin phase.  This function only forms small differences relative to
    the reference state:

    * spin changes are ``δF_k * x**(k+1) / (k+1)!`` (prefer exact
      ``spin_term_deltas`` over ``(F_k - F_k_ref)`` to avoid float64
      ``(θ+δ)-θ`` cancellation);
    * delay changes use the exact Taylor difference ``phase(x - d) - phase(x)``
      with the current spin coefficients.

    The reference pulse numbers and TZR phase cancel in this local residual
    delta as long as the perturbation stays within the same phase connection.
    """
    x = jnp.asarray(dt_base, dtype=jnp.float64)
    d = jnp.asarray(delay_change, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)

    n_coeffs = max(len(ref_f_coeffs), len(f_coeffs))
    if spin_term_deltas is not None:
        n_coeffs = max(n_coeffs, len(spin_term_deltas))
    spin_phase_delta = jnp.zeros_like(x)
    for i in range(n_coeffs):
        if spin_term_deltas is not None and i < len(spin_term_deltas):
            dF = jnp.asarray(spin_term_deltas[i], dtype=jnp.float64)
        else:
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
            dF = coeff - ref_coeff
        spin_phase_delta = spin_phase_delta + (
            dF * (x ** (i + 1)) / float(math.factorial(i + 1))
        )

    delay_phase_delta = jnp.zeros_like(x)
    for j in range(1, len(f_coeffs) + 1):
        g_j = jnp.zeros_like(x)
        for m in range(0, len(f_coeffs) - (j - 1)):
            coeff = jnp.asarray(f_coeffs[m + j - 1], dtype=jnp.float64)
            g_j = g_j + coeff * (x**m) / float(math.factorial(m))
        delay_phase_delta = delay_phase_delta + (((-d) ** j) / float(math.factorial(j)) * g_j)

    residual_delta = (spin_phase_delta + delay_phase_delta) / jnp.asarray(f0, dtype=jnp.float64)

    # Tracer-safe kernel only — never construct ReferenceGauge in-trace.
    residual_delta = residual_delta - _gauge_offset_values(
        residual_delta,
        mode="none" if mean_mode == "none" else "mean",
        weights=weights if mean_mode == "weighted" else None,
        xp=jnp,
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


def _ecliptic_session_metadata(
    ref_params: Mapping[str, object],
) -> tuple[bool, float, tuple[float, float, float, float] | None, str]:
    """Static ecliptic session flags captured before JIT compilation."""
    from jug.io.astrometry_state import native_ecliptic_family
    from jug.io.par_reader import OBLIQUITY_ARCSEC

    if not ref_params.get("_ecliptic_coords"):
        return False, 0.0, None, "elong"

    ecl_frame = str(ref_params.get("_ecliptic_frame", ref_params.get("ECL", "IERS2010"))).upper()
    obl_arcsec = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC["IERS2010"])
    obl_rad = float(obl_arcsec * np.pi / (180.0 * 3600.0))
    init = (
        float(
            ref_params.get(
                "_ecliptic_lon_deg", ref_params.get("ELONG", ref_params.get("LAMBDA", 0.0))
            )
        ),
        float(
            ref_params.get("_ecliptic_lat_deg", ref_params.get("ELAT", ref_params.get("BETA", 0.0)))
        ),
        float(
            ref_params.get(
                "_ecliptic_pm_lon", ref_params.get("PMELONG", ref_params.get("PMLAMBDA", 0.0))
            )
        ),
        float(
            ref_params.get(
                "_ecliptic_pm_lat", ref_params.get("PMELAT", ref_params.get("PMBETA", 0.0))
            )
        ),
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
        # Match NumPy reconvert_ecliptic_to_equatorial: only refresh PMRA/PMDEC
        # when ecliptic proper motion is nonzero; otherwise keep ref values.
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


def _binary_delay_change_jax(params: dict, setup: "GeneralFitSetup", *, binary_plan):
    """Traceable binary-delay change, matching the shared Taylor delay path."""
    if not setup.binary_params or setup.initial_binary_delay is None:
        return None
    if setup.prebinary_delay_sec is None:
        raise ValueError("Binary delay-change requires prebinary_delay_sec in setup.")
    plan = binary_plan
    if plan is None:
        from jug.fitting.binary_delay_plan import resolve_binary_structure

        plan = resolve_binary_structure(
            setup.params, setup.fit_param_list, obs_pos_ls=setup.ssb_obs_pos_ls
        )
    tdb_mjd = jnp.asarray(setup.tdb_mjd, dtype=jnp.float64)
    toas_prebinary = tdb_mjd - (
        jnp.asarray(setup.prebinary_delay_sec, dtype=jnp.float64) / SECS_PER_DAY
    )
    new_binary = jnp.asarray(
        plan.evaluate(toas_prebinary, params, setup.ssb_obs_pos_ls, jnp),
        dtype=jnp.float64,
    )
    return new_binary - jnp.asarray(setup.initial_binary_delay, dtype=jnp.float64)


_HYBRID_BINARY_CACHE_MSG = (
    "hybrid nonlinear_params requires prebinary_delay_sec and "
    "initial_binary_delay on setup (call compute_residuals before "
    "building the residual-delta closure)."
)


def _validate_hybrid_binary_setup(
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    binary_indices: tuple[int, ...],
) -> None:
    """Raise if hybrid binary axes lack cache / setup coverage (§4.3)."""
    if not binary_indices:
        return
    if setup.prebinary_delay_sec is None:
        raise ValueError(_HYBRID_BINARY_CACHE_MSG)
    if setup.initial_binary_delay is None:
        raise ValueError(_HYBRID_BINARY_CACHE_MSG)
    setup_bin = {str(p).upper() for p in setup.binary_params}
    delta_bin = {str(fit_params[i]).upper() for i in binary_indices}
    if not delta_bin <= setup_bin:
        raise ValueError(
            "setup.binary_params must cover all binary delta_params "
            f"({sorted(delta_bin)} vs setup {sorted(setup_bin)})"
        )

_FROZEN_ASTROMETRY_KEYS: tuple[str, ...] = (
    "RAJ",
    "DECJ",
    "PMRA",
    "PMDEC",
    "PX",
    "_raj_rad",
    "_decj_rad",
    "ELONG",
    "ELAT",
    "PMELONG",
    "PMELAT",
    "LAMBDA",
    "BETA",
    "PMLAMBDA",
    "PMBETA",
    "_ecliptic_lon_deg",
    "_ecliptic_lat_deg",
    "_ecliptic_pm_lon",
    "_ecliptic_pm_lat",
)

_LIVE_PX_KEYS: frozenset[str] = frozenset({"PX"})


def _bake_residual_jacobian_native(
    setup: "GeneralFitSetup",
    delta_params: Sequence[str],
) -> np.ndarray:
    """Gauge-free J in native-delta units, session row order (§2.2)."""
    from jug.fitting.optimized_fitter import _compute_designmatrix_from_setup
    from jug.utils.units import native_to_fit_value

    names = tuple(str(name).upper() for name in delta_params)
    M_fit = np.asarray(
        _compute_designmatrix_from_setup(setup, list(names)),
        dtype=np.float64,
    )  # raw fitter M: uncentered, unweighted, fit units (HR1)
    J_native = np.empty_like(M_fit)
    for col, name in enumerate(names):
        # Δr ≈ -M_fit @ δ_fit, δ_native = δ_fit / factor
        # ⇒ J_native = -M_fit * factor with factor = native_to_fit_value(name, 1.0)
        factor = float(native_to_fit_value(name, 1.0))
        J_native[:, col] = -M_fit[:, col] * factor
    return J_native


def _params_with_frozen_astrometry(
    params_pert: Mapping[str, object],
    ref_params: Mapping[str, object],
    setup: "GeneralFitSetup",
    *,
    live_px: bool,
) -> dict[str, object]:
    """Hybrid binary call: freeze astrometry; optionally keep live PX."""
    out = dict(params_pert)
    keys = set(_FROZEN_ASTROMETRY_KEYS)
    keys.update(str(n).upper() for n in setup.astrometry_params)
    if live_px:
        keys -= _LIVE_PX_KEYS
    for key in keys:
        if key in ref_params:
            out[key] = ref_params[key]
    return out


def _hybrid_delta_partition(delta_params: Sequence[str]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition δ axes into binary vs linear indices (§2.1)."""
    from jug.model.parameter_spec import get_binary_params_from_list

    names = tuple(str(p).upper() for p in delta_params)
    binary_names = set(get_binary_params_from_list(list(names)))
    I_bin = tuple(i for i, p in enumerate(names) if p in binary_names)
    I_lin = tuple(i for i in range(len(names)) if i not in set(I_bin))
    return I_bin, I_lin


def _compute_residual_delta_jax_hybrid(
    params_pert: dict,
    setup: "GeneralFitSetup",
    *,
    binary_plan,
    ref_f_terms: Sequence[float],
    residual_jacobian: jnp.ndarray,
    linear_indices: tuple[int, ...],
    binary_indices: tuple[int, ...],
    delta_theta: jnp.ndarray,
    ref_params: Mapping[str, object],
    live_px: bool,  # False → "binary"; True → "binary+"
):
    """Hybrid: gauge-free J_lin @ δ_lin + frozen-prebinary binary block."""
    delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
    J = jnp.asarray(residual_jacobian, dtype=jnp.float64)

    if linear_indices:
        idx = jnp.asarray(linear_indices, dtype=jnp.int32)
        residual = J[:, idx] @ delta_theta[idx]
    else:
        residual = jnp.zeros(J.shape[0], dtype=jnp.float64)

    if binary_indices:
        params_for_binary = _params_with_frozen_astrometry(
            params_pert, ref_params, setup, live_px=live_px
        )
        binary_delay_change = _binary_delay_change_jax(
            params_for_binary, setup, binary_plan=binary_plan
        )
        if binary_delay_change is None:
            raise ValueError(
                "hybrid binary block returned None despite non-empty I_bin; "
                "check setup.initial_binary_delay and cache."
            )
        dt_base = (
            setup.dt_sec_ld
            if setup.dt_sec_ld is not None
            else setup.dt_sec_cached
        )
        f0_ref = jnp.asarray(ref_f_terms[0], dtype=jnp.float64)
        residual = residual + _phase_residual_delta_jax(
            np.asarray(dt_base, dtype=np.float64),
            binary_delay_change,
            ref_f_terms,
            ref_f_terms,  # frozen spin
            jnp.asarray(setup.weights, dtype=jnp.float64),
            mean_mode="none",  # HR3
            f0=f0_ref,
        )
    return residual


def _residual_delta_core_for_setup_hybrid(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object],
    ref_theta: np.ndarray,
    ref_f_terms: tuple[float, ...],
    binary_plan,
    ecliptic_coords: bool,
    obl_rad: float,
    ecliptic_init: dict,
    native_family: str,
    residual_jacobian: np.ndarray,
    linear_indices: tuple[int, ...],
    binary_indices: tuple[int, ...],
    live_px: bool,
):
    """Un-jitted hybrid residual-delta closure."""
    J_const = np.asarray(residual_jacobian, dtype=np.float64)

    def core(delta_theta):
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
        return _compute_residual_delta_jax_hybrid(
            params_pert,
            setup,
            binary_plan=binary_plan,
            ref_f_terms=ref_f_terms,
            residual_jacobian=J_const,
            linear_indices=linear_indices,
            binary_indices=binary_indices,
            delta_theta=delta_theta,
            ref_params=ref_params,
            live_px=live_px,
        )

    return core


def _exact_spin_term_deltas(
    fit_params: Sequence[str],
    delta_theta,
    n_terms: int,
) -> list:
    """Exact δF_k from the δ vector (avoids float64 (θ+δ)-θ cancellation)."""
    delta_theta = jnp.asarray(delta_theta, dtype=jnp.float64).reshape(-1)
    deltas: list = [jnp.asarray(0.0, dtype=jnp.float64) for _ in range(n_terms)]
    for idx, name in enumerate(fit_params):
        param_upper = str(name).upper()
        if param_upper.startswith("F") and param_upper[1:].isdigit():
            order = int(param_upper[1:])
            if 0 <= order < n_terms:
                deltas[order] = delta_theta[idx]
    return deltas


def _compute_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    setup: "GeneralFitSetup",
    *,
    native_pack: NativeDeltaPack | None,
    ref_f_terms: Sequence[float],
    phase_mean_mode: str,
    binary_plan=None,
    delay_model: str = "native",
    fit_params: Sequence[str] | None = None,
    delta_theta=None,
):
    """Residual delta (perturbed - reference) through JUG's JAX forward model."""
    use_native_tempo2 = (
        delay_model != "simplified"
        and normalize_compatibility_mode(str(getattr(setup, "compatibility", ""))) == "tempo2"
    )
    f_terms = _spin_terms_from_params(params_pert)
    spin_deltas = None
    if fit_params is not None and delta_theta is not None:
        n_terms = max(len(ref_f_terms), len(f_terms))
        spin_deltas = _exact_spin_term_deltas(fit_params, delta_theta, n_terms)
        # Freeze F0 divisor at the reference for spin-linear consistency with
        # analytic J = -M (bake / hybrid linear block).
        f0_div = jnp.asarray(ref_f_terms[0], dtype=jnp.float64)
    else:
        f0_div = _param_scalar(params_pert, "F0", f_terms[0])

    if use_native_tempo2:
        if native_pack is None:
            static = getattr(setup, "native_chain_static", None)
            if static is None:
                raise ValueError(
                    "tempo2 native residual_delta requires native_chain_static on "
                    "GeneralFitSetup. Rebuild from a residual cache that includes "
                    "term_diagnostics (e.g. call compute_residuals before "
                    "export_frozen_residual_model). "
                    "Set tempo2_native to fixed_state_stripped (default), "
                    "fixed_state_bclt, staged_bclt, or full."
                )
            raise ValueError(
                "tempo2 native residual_delta could not build a native delta pack "
                "(missing term_diagnostics['tempo2_obs_state'] or TOA list on setup)."
            )
        native_delay_change = compute_bbat_delay_change_sec_jax(
            params_ref, params_pert, native_pack
        )
        binary_delay_change = _binary_delay_change_jax(params_pert, setup, binary_plan=binary_plan)
        side_delay_change = compute_side_delay_change(params_pert, setup, xp=jnp)
        total_delay_change = native_delay_change + side_delay_change
        if binary_delay_change is not None:
            total_delay_change = total_delay_change + binary_delay_change
        return _phase_residual_delta_jax(
            np.asarray(setup.dt_sec_cached, dtype=np.float64),
            total_delay_change,
            ref_f_terms,
            f_terms,
            jnp.asarray(setup.weights, dtype=jnp.float64),
            mean_mode=phase_mean_mode,
            f0=f0_div,
            spin_term_deltas=spin_deltas,
        )

    del native_pack

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

    return _phase_residual_delta_jax(
        dt_base,
        delay_change,
        ref_f_terms,
        f_terms,
        weights,
        mean_mode=phase_mean_mode,
        f0=f0_div,
        spin_term_deltas=spin_deltas,
    )


def _residual_delta_core_for_setup(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object],
    ref_theta: np.ndarray,
    phase_mean_mode: str,
    native_pack: NativeDeltaPack | None,
    ref_f_terms: tuple[float, ...],
    binary_plan,
    ecliptic_coords: bool,
    obl_rad: float,
    ecliptic_init: dict,
    native_family: str,
    delay_model: str = "native",
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
            native_pack=native_pack,
            ref_f_terms=ref_f_terms,
            phase_mean_mode=phase_mean_mode,
            binary_plan=binary_plan,
            delay_model=delay_model,
            fit_params=fit_params,
            delta_theta=delta_theta,
        )

    return core


def _residual_delta_jax_cache_key(
    setup: "GeneralFitSetup",
    *,
    fit_params: tuple[str, ...],
    ref_theta: np.ndarray,
    ref_f_terms: tuple[float, ...],
    phase_mean_mode: str,
    delay_model: str = "native",
    nonlinear_params: str | None = None,
    residual_jacobian: np.ndarray | None = None,
) -> tuple:
    """Hashable key for session-scoped residual/Jacobian JIT bundles."""
    from jug.fitting.nonlinear_params import (
        is_hybrid_nonlinear_params,
        validate_nonlinear_params,
    )

    mode = validate_nonlinear_params(nonlinear_params)
    J_shape = None
    J_digest = None
    if is_hybrid_nonlinear_params(mode):
        if residual_jacobian is None:
            raise ValueError(
                "hybrid residual_delta cache key requires a baked residual_jacobian"
            )
        J_arr = np.ascontiguousarray(residual_jacobian, dtype=np.float64)
        J_shape = J_arr.shape
        J_digest = hash(J_arr.tobytes())  # host-side only
    return (
        fit_params,
        tuple(float(x) for x in ref_theta),
        ref_f_terms,
        phase_mean_mode,
        delay_model,
        str(getattr(setup, "tempo2_native", None)),
        str(setup.compatibility),
        id(getattr(setup, "native_chain_static", None)),
        mode,
        J_shape,
        J_digest,
    )


def _build_residual_delta_jax_bundle(
    *,
    setup: "GeneralFitSetup",
    fit_params: tuple[str, ...],
    ref_params: Mapping[str, object],
    ref_theta: np.ndarray,
    phase_mean_mode: str,
    delay_model: str = "native",
    nonlinear_params: str | None = None,
    residual_jacobian: np.ndarray | None = None,
):
    """Build shared residual core and jitted residual / Jacobian evaluators."""
    from jug.fitting.binary_delay_plan import resolve_binary_structure
    from jug.fitting.nonlinear_params import (
        NONLINEAR_PARAMS_BINARY_PLUS,
        is_hybrid_nonlinear_params,
        validate_nonlinear_params,
    )

    mode = validate_nonlinear_params(nonlinear_params)
    ref_f_terms = tuple(float(x) for x in _spin_terms_from_params(ref_params))
    binary_plan = resolve_binary_structure(
        ref_params, fit_params, obs_pos_ls=getattr(setup, "ssb_obs_pos_ls", None)
    )
    ecliptic_coords, obl_rad, ecliptic_init, native_family = _ecliptic_session_metadata(ref_params)

    if is_hybrid_nonlinear_params(mode):
        I_bin, I_lin = _hybrid_delta_partition(fit_params)
        _validate_hybrid_binary_setup(setup, fit_params, I_bin)
        if residual_jacobian is None:
            residual_jacobian = _bake_residual_jacobian_native(setup, fit_params)
        else:
            residual_jacobian = np.asarray(residual_jacobian, dtype=np.float64)
        live_px = mode == NONLINEAR_PARAMS_BINARY_PLUS
        core = _residual_delta_core_for_setup_hybrid(
            setup=setup,
            fit_params=fit_params,
            ref_params=ref_params,
            ref_theta=ref_theta,
            ref_f_terms=ref_f_terms,
            binary_plan=binary_plan,
            ecliptic_coords=ecliptic_coords,
            obl_rad=obl_rad,
            ecliptic_init=ecliptic_init,
            native_family=native_family,
            residual_jacobian=residual_jacobian,
            linear_indices=I_lin,
            binary_indices=I_bin,
            live_px=live_px,
        )
        return core, jax.jit(core), jax.jit(jax.jacfwd(core))

    from jug.residuals.engine_conventions import normalize_compatibility_mode

    native_pack = None
    if (
        delay_model != "simplified"
        and normalize_compatibility_mode(str(setup.compatibility)) == "tempo2"
    ):
        native_pack = build_delta_pack_for_setup(setup)

    core = _residual_delta_core_for_setup(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
        native_pack=native_pack,
        ref_f_terms=ref_f_terms,
        binary_plan=binary_plan,
        ecliptic_coords=ecliptic_coords,
        obl_rad=obl_rad,
        ecliptic_init=ecliptic_init,
        native_family=native_family,
        delay_model=delay_model,
    )
    return core, jax.jit(core), jax.jit(jax.jacfwd(core))


def _prepare_residual_delta_jax(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object] | None = None,
    ref_theta: np.ndarray | None = None,
    phase_mean_mode: str | None = None,
    delay_model: str = "native",
    nonlinear_params: str | None = None,
    residual_jacobian: np.ndarray | None = None,
):
    """Build or reuse session-cached residual core and JIT evaluators.

    Defaults to a gauge-free graph (``phase_mean_mode="none"``). Host reporting
    gauges are opt-in via an explicit ``phase_mean_mode``; ``_phase_mean_mode``
    is not consulted here.
    """
    from jug.fitting.forward_delay import _assert_no_epoch_fit_params
    from jug.fitting.nonlinear_params import (
        is_hybrid_nonlinear_params,
        validate_nonlinear_params,
    )

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

    # Omitted / explicit None both mean "use setup.nonlinear_params". To force
    # native while setup still says hybrid, mutate setup.nonlinear_params first.
    if nonlinear_params is None:
        mode = validate_nonlinear_params(getattr(setup, "nonlinear_params", None))
    else:
        mode = validate_nonlinear_params(nonlinear_params)
    setup.nonlinear_params = mode

    ref_f_terms = tuple(float(x) for x in _spin_terms_from_params(ref_params))
    if phase_mean_mode is None:
        phase_mean_mode = "none"
    if is_hybrid_nonlinear_params(mode):
        if phase_mean_mode != "none":
            raise ValueError(
                "hybrid nonlinear_params forces phase_mean_mode='none' "
                f"(got {phase_mean_mode!r})"
            )
        I_bin, _I_lin = _hybrid_delta_partition(fit_params)
        # Validate binary cache before design-matrix bake so missing
        # prebinary yields _HYBRID_BINARY_CACHE_MSG, not a bake-path error.
        _validate_hybrid_binary_setup(setup, fit_params, I_bin)
        if residual_jacobian is None:
            residual_jacobian = _bake_residual_jacobian_native(setup, fit_params)
        else:
            residual_jacobian = np.asarray(residual_jacobian, dtype=np.float64)

    cache_key = _residual_delta_jax_cache_key(
        setup,
        fit_params=fit_params,
        ref_theta=ref_theta,
        ref_f_terms=ref_f_terms,
        phase_mean_mode=phase_mean_mode,
        delay_model=delay_model,
        nonlinear_params=mode,
        residual_jacobian=residual_jacobian,
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
        delay_model=delay_model,
        nonlinear_params=mode,
        residual_jacobian=residual_jacobian,
    )
    cache[cache_key] = bundle
    return bundle


def make_residual_delta_jax_fn(
    *,
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
    ref_params: Mapping[str, object] | None = None,
    ref_theta: np.ndarray | None = None,
    phase_mean_mode: str = "none",
    nonlinear_params: str | None = None,
    residual_jacobian: np.ndarray | None = None,
):
    """Return the gauge-free residual-delta function for a frozen fit setup.

    Defaults to ``phase_mean_mode="none"``: a public residual_delta constructor
    must not silently apply a reporting gauge. Pass an explicit mode only for
    host/reporting parity callers that need a centered graph.

    ``nonlinear_params`` selects native (``None``) or hybrid (``"binary"`` /
    ``"binary+"``) linearization. When omitted, ``setup.nonlinear_params`` is
    used. Hybrid forces ``phase_mean_mode="none"``.
    """
    _, residual_fn, _ = _prepare_residual_delta_jax(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
        phase_mean_mode=phase_mean_mode,
        nonlinear_params=nonlinear_params,
        residual_jacobian=residual_jacobian,
    )
    return residual_fn


def compute_full_model_residuals_tempo2_native(
    eval_params: Mapping[str, object],
    setup: "GeneralFitSetup",
    *,
    ref_params: Mapping[str, object],
    ref_residuals_sec: np.ndarray,
    subtract_tzr: bool = True,
    tzr_apply_mode: str | None = None,
    tzr_offset_sec: float | None = None,
) -> np.ndarray:
    """Absolute residuals at eval_params via cached host ref + native tempo2 delta."""
    if normalize_compatibility_mode(str(getattr(setup, "compatibility", ""))) != "tempo2":
        raise ValueError("tempo2 native full-model residuals require compatibility='tempo2'")
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        raise ValueError(
            "tempo2 native full-model residuals require native_chain_static on setup "
            "(populate term_diagnostics via compute_residuals first)."
        )

    fit_params = [str(p).upper() for p in (setup.fit_param_list or ())]
    if not fit_params:
        return np.asarray(ref_residuals_sec, dtype=np.float64)

    model_mjd = getattr(static, "model_mjd", None)
    if model_mjd is None:
        model_mjd = getattr(setup, "toas_mjd", np.asarray(ref_residuals_sec) * 0.0)
    reference_gauge = _host_reference_gauge(
        compatibility=str(setup.compatibility),
        params=ref_params,
        model_mjd=model_mjd,
        weights=setup.weights,
        subtract_tzr=subtract_tzr,
        tzr_apply_mode=tzr_apply_mode,
        tzr_offset_sec=tzr_offset_sec,
    )

    ref_params_norm = _normalize_ref_params(ref_params)
    ref_theta = np.array(
        [_reference_param_value(ref_params_norm, name) for name in fit_params],
        dtype=np.float64,
    )
    eval_theta = np.array(
        [_reference_param_value(eval_params, name) for name in fit_params],
        dtype=np.float64,
    )
    delta_theta = eval_theta - ref_theta

    residual_fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params_norm,
        ref_theta=ref_theta,
        phase_mean_mode="none",
    )
    delta_sec = np.asarray(residual_fn(delta_theta), dtype=np.float64)
    return np.asarray(
        reconstruct_absolute_residuals(
            np.asarray(ref_residuals_sec, dtype=np.float64),
            delta_sec,
            reference_gauge,
        ),
        dtype=np.float64,
    )


def _simplified_residual_jacobian_oracle(
    setup: "GeneralFitSetup",
    fit_params: Sequence[str],
) -> np.ndarray:
    """J_fit of the simplified gauge-free residual graph (test oracle only).

    Returns the residual Jacobian (positive sign, fit-unit columns) of the
    PINT-style Taylor graph, ignoring ``compatibility``/``tempo2_native``.
    Exists to validate analytic derivative blocks and graph traceability;
    never imported by production code. Tests compare it as
    J_fit ≈ -M_analytic for certified parameters (no centering transform).
    """
    fit_params = tuple(str(name).upper() for name in fit_params)
    _, _, jac_fn = _prepare_residual_delta_jax(
        setup=setup,
        fit_params=fit_params,
        delay_model="simplified",
        phase_mean_mode="none",
    )
    zero = jnp.zeros((len(fit_params),), dtype=jnp.float64)
    jac_native = np.asarray(jac_fn(zero), dtype=np.float64)
    return (
        np.column_stack(
            [
                np.asarray(
                    native_derivative_to_fit_column(param, jac_native[:, col]),
                    dtype=np.float64,
                )
                for col, param in enumerate(fit_params)
            ]
        )
        if fit_params
        else np.empty((len(np.asarray(setup.tdb_mjd)), 0))
    )
