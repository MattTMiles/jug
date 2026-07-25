"""Static binary-model dispatch resolved once from concrete reference params."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from jug.utils.constants import SECS_PER_DAY, T_SUN

from jug.delays.binary_bt import bt_binary_delay_from_tt0
from jug.fitting.binary_t2_dispatch import _is_ell1_parameterization
from jug.fitting.derivatives_binary import _compute_ell1_binary_delay_jit, _extract_ell1_params
from jug.fitting.derivatives_dd import (
    _as_f64,
    _compute_dd_binary_delay_jit,
    _compute_kopeikin_corrections_traceable,
    _extract_dd_params,
    resolve_kopeikin_flags,
)
from jug.io.par_reader import get_longdouble
from jug.utils.orbit_reduction import reduce_binary_time_sec


def _pval(params, key, default):
    v = params.get(key, None)
    return default if v is None else v


def _first_live_value(params, live_keys, keys, default):
    """Return first live traced value among aliases, else reference default."""
    for key in keys:
        if key in live_keys and params.get(key, None) is not None:
            return _pval(params, key, default)
    return default


_DD_ARG_KEYS = {
    "a1": ("A1",),
    "pb": ("PB",),
    "t0": ("T0",),
    "ecc": ("ECC", "E"),
    "om_deg": ("OM",),
    "omdot": ("OMDOT",),
    "pbdot": ("PBDOT",),
    "gamma": ("GAMMA",),
    "sini": ("SINI",),
    "m2": ("M2",),
    "xdot": ("XDOT", "A1DOT"),
    "edot": ("EDOT",),
    "h3": ("H3",),
    "stig": ("STIG", "STIGMA"),
}
_DD_ARG_ORDER = (
    "a1",
    "pb",
    "t0",
    "ecc",
    "om_deg",
    "omdot",
    "pbdot",
    "gamma",
    "sini",
    "m2",
    "xdot",
    "edot",
)

_ELL1_ARG_KEYS = {
    "a1": ("A1",),
    "pb": ("PB",),
    "tasc": ("TASC",),
    "eps1": ("EPS1",),
    "eps2": ("EPS2",),
    "pbdot": ("PBDOT",),
    "a1dot": ("A1DOT", "XDOT"),
    "sini": ("SINI",),
    "m2": ("M2",),
    "gamma": ("GAMMA",),
    "h3": ("H3",),
    "h4": ("H4",),
    "stig": ("STIG", "STIGMA"),
    "eps1dot": ("EPS1DOT",),
    "eps2dot": ("EPS2DOT",),
}
_ELL1_ARG_ORDER = (
    "a1",
    "pb",
    "tasc",
    "eps1",
    "eps2",
    "pbdot",
    "a1dot",
    "sini",
    "m2",
    "gamma",
    "h3",
    "h4",
    "stig",
    "eps1dot",
    "eps2dot",
)

_ELL1_MODELS = frozenset({"ELL1", "ELL1H", "ELL1K"})
_BT_MODELS = frozenset({"BT", "BTX"})
_DD_MODELS = frozenset({"DD", "DDS", "DDH", "DDGR"})
_ORTHOMETRIC_KEYS = frozenset({"H3", "H4", "STIG", "STIGMA"})

_BT_ARG_KEYS = {k: _DD_ARG_KEYS[k] for k in (
    "a1", "pb", "t0", "ecc", "om_deg", "omdot", "pbdot", "gamma", "xdot", "edot",
)}
_BT_ARG_ORDER = tuple(_BT_ARG_KEYS.keys())


def _bt_tt0_arrays(toas_prebinary, params, live_keys, t0, pb):
    """(tt0_sec, tt0_red_sec) for BT; longdouble reduction when T0 is not live."""
    t0_j = jnp.asarray(t0, dtype=jnp.float64)
    t = jnp.asarray(toas_prebinary, dtype=jnp.float64)
    tt0_sec = (t - t0_j) * SECS_PER_DAY
    tt0_red = None
    if "T0" not in live_keys:
        t0_ld = get_longdouble(params, "T0", default=t0)
        toas_ld = np.asarray(toas_prebinary, dtype=np.longdouble)
        tt0_ld = (toas_ld - np.longdouble(t0_ld)) * np.longdouble(SECS_PER_DAY)
        tt0_sec = jnp.asarray(tt0_ld, dtype=np.float64)
        tt0_red = jnp.asarray(
            reduce_binary_time_sec(tt0_ld, pb_days=float(pb)), dtype=jnp.float64
        )
    return tt0_sec, tt0_red


@dataclass(frozen=True)
class BinaryDelayPlan:
    family: str
    ref_scalars: dict
    live_keys: frozenset
    structural_args: frozenset
    fb_ref: tuple
    kopeikin: Optional[Any] = None
    pb_ld: Any = None
    fb0_ld: Any = None
    nharm: float = 4.0
    shapiro_param: str = "m2_sini"  # "m2_sini" | "h3_stig"
    has_shapiro: bool = False

    def _arg(self, arg, params, arg_keys):
        if arg not in self.structural_args:
            for key in arg_keys[arg]:
                if key in self.live_keys and params.get(key, None) is not None:
                    return _pval(params, key, self.ref_scalars[arg])
        return self.ref_scalars[arg]

    def _fb_array(self, params):
        if not self.fb_ref:
            return jnp.asarray([], dtype=jnp.float64)
        vals = []
        for i, ref in enumerate(self.fb_ref):
            key = f"FB{i}"
            if key in self.live_keys and params.get(key, None) is not None:
                vals.append(_pval(params, key, ref))
            else:
                vals.append(ref)
        return jnp.asarray(vals, dtype=jnp.float64)

    def evaluate(self, toas_prebinary, params, obs_pos_ls, xp):
        t = jnp.asarray(toas_prebinary, dtype=jnp.float64)
        if self.family == "ELL1":
            args = [self._arg(a, params, _ELL1_ARG_KEYS) for a in _ELL1_ARG_ORDER]
            (
                a1,
                pb,
                tasc,
                eps1,
                eps2,
                pbdot,
                a1dot,
                sini,
                m2,
                gamma,
                h3,
                h4,
                stig,
                eps1dot,
                eps2dot,
            ) = args
            fb = self._fb_array(params)
            tasc_j = jnp.asarray(tasc, dtype=jnp.float64)
            ttasc_sec = (t - tasc_j) * SECS_PER_DAY
            # Longdouble orbit-count reduction is applied in the NumPy
            # compute_ell1_binary_delay path; autodiff uses float64 here.
            return _compute_ell1_binary_delay_jit(
                ttasc_sec,
                _as_f64(a1),
                _as_f64(pb),
                _as_f64(eps1),
                _as_f64(eps2),
                _as_f64(pbdot),
                _as_f64(a1dot),
                _as_f64(sini),
                _as_f64(m2),
                _as_f64(gamma),
                _as_f64(h3),
                _as_f64(h4),
                _as_f64(stig),
                fb,
                _as_f64(eps1dot),
                _as_f64(eps2dot),
                nharm=_as_f64(self.nharm),
            )

        if self.family == "BT":
            args = [self._arg(a, params, _BT_ARG_KEYS) for a in _BT_ARG_ORDER]
            (a1, pb, t0, ecc, om_deg, omdot, pbdot, gamma, xdot, edot) = args
            tt0_sec, tt0_red = _bt_tt0_arrays(
                toas_prebinary, params, self.live_keys, t0, pb
            )
            return bt_binary_delay_from_tt0(
                tt0_sec,
                _as_f64(pb),
                _as_f64(a1),
                _as_f64(ecc),
                _as_f64(om_deg),
                _as_f64(gamma),
                _as_f64(pbdot),
                _as_f64(omdot),
                _as_f64(xdot),
                _as_f64(edot),
                tt0_red_sec=tt0_red,
            )

        args = [self._arg(a, params, _DD_ARG_KEYS) for a in _DD_ARG_ORDER]
        (a1, pb, t0, ecc, om_deg, omdot, pbdot, gamma, sini, m2, xdot, edot) = args
        if self.shapiro_param == "h3_stig":
            h3 = _as_f64(self._arg("h3", params, _DD_ARG_KEYS))
            stig = _as_f64(self._arg("stig", params, _DD_ARG_KEYS))
            sini = 2.0 * stig / (1.0 + stig * stig)
            m2 = h3 / (stig * stig * stig * T_SUN)
        if self.family == "DDK":
            kop = self.kopeikin
            kin = _pval(params, "KIN", kop.kin_deg_ref)
            kom = _pval(params, "KOM", kop.kom_deg_ref)
            px = _pval(params, "PX", kop.px_mas_ref)
            pmra = (
                _first_live_value(params, self.live_keys, kop.pmra_keys, kop.pmra_ref)
                * kop.pm_factor
            )
            pmdec = (
                _first_live_value(params, self.live_keys, kop.pmdec_keys, kop.pmdec_ref)
                * kop.pm_factor
            )
            d_a1, d_om, sini_eff = _compute_kopeikin_corrections_traceable(
                t,
                _as_f64(a1),
                _as_f64(t0),
                _as_f64(kin),
                _as_f64(kom),
                _as_f64(px),
                _as_f64(pmra),
                _as_f64(pmdec),
                obs_pos_ls,
                kop,
            )
            a1 = a1 + d_a1
            om_deg = om_deg + d_om
            sini = sini_eff
        t0_j = jnp.asarray(t0, dtype=jnp.float64)
        tt0_sec = (t - t0_j) * SECS_PER_DAY
        return _compute_dd_binary_delay_jit(
            tt0_sec,
            _as_f64(a1),
            _as_f64(pb),
            _as_f64(ecc),
            _as_f64(om_deg),
            _as_f64(omdot),
            _as_f64(pbdot),
            _as_f64(gamma),
            _as_f64(sini),
            _as_f64(m2),
            _as_f64(xdot),
            _as_f64(edot),
            has_shapiro=self.has_shapiro,
        )


def resolve_binary_structure(ref_params, fit_params, *, obs_pos_ls=None):
    """Build a BinaryDelayPlan from concrete reference params."""
    model = str(ref_params.get("BINARY", "")).upper().strip()
    if not model:
        return None
    live = frozenset(str(p).upper() for p in (fit_params or []))

    if model in _ELL1_MODELS:
        family = "ELL1"
    elif model == "DDK":
        family = "DDK"
    elif model == "T2":
        if _is_ell1_parameterization(ref_params):
            family = "ELL1"
        elif "KIN" in ref_params or "KOM" in ref_params:
            family = "DDK"
        else:
            family = "DD"
    elif model in _BT_MODELS:
        family = "BT"
    elif model in _DD_MODELS:
        family = "DD"
    else:
        known = sorted(set(_ELL1_MODELS) | set(_BT_MODELS) | set(_DD_MODELS) | {"DDK", "T2"})
        raise NotImplementedError(
            f"Binary model {model!r} is not supported by traceable dispatch. "
            f"Known models: {known}."
        )

    if family == "ELL1":
        p = _extract_ell1_params(ref_params)
        ref_scalars = {k: float(p[k]) for k in _ELL1_ARG_ORDER}
        fb_ref = tuple(p["fb_coeffs"])
        structural = frozenset({"pb"}) if fb_ref else frozenset()
        return BinaryDelayPlan(
            family,
            ref_scalars,
            live,
            structural,
            fb_ref,
            pb_ld=p["pb_ld"],
            fb0_ld=p.get("fb0_ld"),
            nharm=float(p.get("nharm", 4.0)),
        )

    if family == "BT":
        p = _extract_dd_params(ref_params)
        ref_scalars = {k: float(p[k]) for k in _BT_ARG_ORDER}
        structural = (
            frozenset({"pb"})
            if ("PB" not in ref_params and "FB0" in ref_params)
            else frozenset()
        )
        return BinaryDelayPlan(family, ref_scalars, live, structural, (), None)

    # DD-family fall-through (DD / DDS / DDH / DDGR / DDK / T2→DD/DDK)
    p = _extract_dd_params(ref_params)
    ref_scalars = {k: float(p[k]) for k in _DD_ARG_ORDER}
    ref_scalars["h3"] = float(get_longdouble(ref_params, "H3", default=0.0))
    ref_scalars["h4"] = float(get_longdouble(ref_params, "H4", default=0.0))
    ref_scalars["stig"] = float(
        get_longdouble(
            ref_params,
            "STIG",
            default=get_longdouble(ref_params, "STIGMA", default=0.0),
        )
    )

    if "SHAPMAX" in live:
        raise NotImplementedError(
            "Fitting SHAPMAX (DDS) is not supported via autodiff; fit SINI, "
            "or extend the DDS parameterization in the binary plan."
        )

    shapiro_param = "m2_sini"
    ortho_live = live & _ORTHOMETRIC_KEYS
    h3_ref = ref_scalars["h3"]
    h4_ref = ref_scalars["h4"]
    stig_ref = ref_scalars["stig"]
    ortho_active = bool(ortho_live) or any(
        v != 0.0 for v in (h3_ref, h4_ref, stig_ref)
    )
    if ortho_active:
        # Every active orthometric reference must be finite, regardless of
        # which parameterization is (or is not) selected below — a NaN/inf
        # standalone H3 must not fall through the dispatch silently.
        for _name, _val in (("H3", h3_ref), ("H4", h4_ref), ("STIG", stig_ref)):
            if not np.isfinite(_val):
                raise ValueError(f"Reference {_name}={_val!r} must be finite.")
    if family == "DDK" and ortho_active:
        raise NotImplementedError(
            "Orthometric Shapiro parameters (H3/H4/STIG) are not supported for "
            "DDK/Kopeikin binaries: DDK derives the inclination from KIN. "
            "Fit KIN and M2 instead."
        )
    if ortho_active and family != "DDK":
        if (live & {"M2", "SINI"}) and ortho_live:
            raise ValueError(
                "DD Shapiro fit mixes M2/SINI and orthometric parameters; "
                "choose exactly one parameterization."
            )
        if "H4" in live:
            raise NotImplementedError(
                "Fitting H4 on a DD-family binary via autodiff is not supported: "
                "live H3/H4 requires a coupled same-sign chart (sigma = H4/H3 is "
                "singular at H3 = 0). Use H3/STIG, or fit M2/SINI."
            )
        stig_configured = (stig_ref != 0.0) or bool({"STIG", "STIGMA"} & live)
        h4_configured = h4_ref != 0.0
        if stig_configured:
            shapiro_param = "h3_stig"
            if h4_configured:
                warnings.warn(
                    "Both STIG and H4 present; using H3/STIG and ignoring H4.",
                    UserWarning,
                )
            if stig_ref <= 0.0:
                raise ValueError(
                    f"Reference STIG={stig_ref!r} must be > 0."
                )
            # h3_ref: finiteness already guaranteed above; any signed value,
            # including 0, is allowed.
        elif h4_configured:
            # Reference-only H3/H4: converted ONCE to M2/SINI reference values
            # by _extract_dd_params; the plan stays "m2_sini". H3 cannot
            # be live in this configuration (no STIG to anchor sigma).
            if "H3" in live:
                raise NotImplementedError(
                    "Fitting H3 with an H3/H4 reference (no STIG) is not "
                    "supported via autodiff; provide STIG/STIGMA, or fit "
                    "M2/SINI."
                )
            if h3_ref == 0.0:
                raise ValueError(
                    f"H3/H4 reference requires nonzero H3; got H3={h3_ref!r}."
                )
            sigma = h4_ref / h3_ref
            if sigma <= 0.0:
                raise ValueError(
                    f"H4/H3={sigma!r} must be > 0 "
                    "(H3 and H4 must carry the same sign)."
                )
        elif ortho_live:
            raise ValueError(
                "Orthometric Shapiro fit requires STIG/STIGMA alongside H3; "
                f"got H3={h3_ref!r}, H4={h4_ref!r}, STIG={stig_ref!r}."
            )

    has_shapiro = (
        shapiro_param != "m2_sini"
        or (ref_scalars["sini"] > 0.0 and ref_scalars["m2"] != 0.0)
        or bool(live & {"M2", "SINI"})
    )

    structural = (
        frozenset({"pb"})
        if ("PB" not in ref_params and "FB0" in ref_params)
        else frozenset()
    )
    kop = resolve_kopeikin_flags(ref_params) if family == "DDK" else None
    return BinaryDelayPlan(
        family,
        ref_scalars,
        live,
        structural,
        (),
        kop,
        shapiro_param=shapiro_param,
        has_shapiro=has_shapiro,
    )


@dataclass(frozen=True)
class BinaryChartFacts:
    """Facts about a resolved binary relevant to eccentricity-vector
    (Laplace-Lagrange) coordinate reparameterizations.

    A common low-eccentricity reparameterization replaces the Kepler triple
    ``(ECC, OM, T0)`` with the Cartesian triple
    ``(EPS1 = e·sinω, EPS2 = e·cosω, TASC = T0 - PB·ω/2π)``. Deciding whether
    that change of variables is applicable and well-behaved for a given binary
    needs three model facts, which JUG (having resolved the binary model) is the
    natural owner of:

    convention_family: the RESOLVED Kepler-parameter family —
        ``'dd'`` (``ECC/OM/T0``: DD/DDS/DDH/DDGR/BT/BTX/DDK, and T2 reduced to
        any of these), ``'ell1'`` (already ``EPS1/EPS2/TASC``), or ``'other'``.
    epoch_shift_exact: whether ``(OM + 360°, T0 + PB)`` is an exact orbit
        identity — ``False`` when any secular rate is active for this binary
        (an explicit nonzero or fitted OMDOT/PBDOT/EDOT/A1DOT/XDOT, or a
        GR-derived family such as DDGR whose post-Keplerian rates are computed
        internally rather than exposed as parameters).
    secular_terms: canonical names of the active secular rates (informational).
    """

    convention_family: str
    epoch_shift_exact: bool
    secular_terms: tuple


# Binary families whose secular rates are GR-derived from the model name alone
# (invisible to a name search over explicit params). resolve_binary_structure
# collapses T2/DDGR into a resolved family, so the ORIGINAL BINARY value is read
# to recover this.
_GR_DERIVED_MODELS = frozenset({"DDGR"})
# ref_scalars arg key -> canonical secular-rate name (DD/BT/DDK plans).
_SECULAR_ARG_TO_NAME = {
    "omdot": "OMDOT",
    "pbdot": "PBDOT",
    "edot": "EDOT",
    "xdot": "A1DOT",
}
# live (fit) fitpar name -> canonical secular-rate name.
_SECULAR_LIVE_TO_NAME = {
    "OMDOT": "OMDOT",
    "PBDOT": "PBDOT",
    "EDOT": "EDOT",
    "A1DOT": "A1DOT",
    "XDOT": "A1DOT",
}


def binary_chart_facts(ref_params, fit_params=None):
    """Return the resolved-binary facts needed to reason about an
    eccentricity-vector (Laplace-Lagrange) coordinate reparameterization.

    Returns a :class:`BinaryChartFacts`, or ``None`` when the parameters carry
    no binary. JUG resolves ``T2 -> DD/ELL1/DDK`` and knows which families are
    Kepler-parameterized.
    """
    plan = resolve_binary_structure(ref_params, fit_params)
    if plan is None:
        return None

    if plan.family == "ELL1":
        convention_family = "ell1"
    elif plan.family in ("DD", "DDK", "BT"):
        convention_family = "dd"
    else:
        convention_family = "other"

    original_model = str(ref_params.get("BINARY", "")).upper().strip()
    secular = set()
    if original_model in _GR_DERIVED_MODELS:
        # GR-derived secular rates are always "active" for epoch-shift purposes.
        secular.update(("OMDOT", "PBDOT"))
    for arg, name in _SECULAR_ARG_TO_NAME.items():
        if arg in plan.ref_scalars and float(plan.ref_scalars[arg]) != 0.0:
            secular.add(name)
    for live_name, canon in _SECULAR_LIVE_TO_NAME.items():
        if live_name in plan.live_keys:
            secular.add(canon)

    return BinaryChartFacts(
        convention_family=convention_family,
        epoch_shift_exact=not bool(secular),
        secular_terms=tuple(sorted(secular)),
    )
