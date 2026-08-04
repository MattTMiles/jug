"""Static binary-model dispatch resolved once from concrete reference params."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from jug.utils.constants import SECS_PER_DAY

from jug.delays.binary_bt import bt_binary_delay_from_tt0
from jug.fitting.binary_t2_dispatch import _is_ell1_parameterization
from jug.fitting.derivatives_binary import _compute_ell1_binary_delay_jit, _extract_ell1_params
from jug.fitting.derivatives_dd import (
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


def _as_f64(x):
    """Cast a concrete or traced scalar to float64 for JAX kernels."""
    if isinstance(x, (np.longdouble, np.float128)):
        return jnp.float64(float(x))
    return jnp.asarray(x, dtype=jnp.float64)


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
    "ecc": ("ECC",),
    "om_deg": ("OM",),
    "omdot": ("OMDOT",),
    "pbdot": ("PBDOT",),
    "gamma": ("GAMMA",),
    "sini": ("SINI",),
    "m2": ("M2",),
    "xdot": ("XDOT", "A1DOT"),
    "edot": ("EDOT",),
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
        # T0 is in HIGH_PRECISION_PARAMS, so the fitter hands it over as
        # np.longdouble and JAX cannot promote float128. Cast once, up here:
        # the DDK/Kopeikin branch below needs the float64 value too. b109577
        # coerced the other binary scalars but this call site was missed.
        t0_j = jnp.asarray(np.float64(t0), dtype=jnp.float64)
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
            # Every scalar entering the JAX Kopeikin kernel must be float64:
            # callers (e.g. the PINT-matching harness) can hand these over as
            # np.longdouble, which JAX refuses to promote.
            d_a1, d_om, sini_eff = _compute_kopeikin_corrections_traceable(
                t,
                _as_f64(a1),
                t0_j,
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

    if live & _ORTHOMETRIC_KEYS:
        raise NotImplementedError(
            "Fitting orthometric Shapiro parameters (H3/H4/STIG) is not supported "
            "for DD-family binaries via autodiff. Use ELL1H or fit M2/SINI."
        )
    p = _extract_dd_params(ref_params)
    ref_scalars = {k: float(p[k]) for k in _DD_ARG_ORDER}
    structural = (
        frozenset({"pb"})
        if ("PB" not in ref_params and "FB0" in ref_params)
        else frozenset()
    )
    kop = resolve_kopeikin_flags(ref_params) if family == "DDK" else None
    return BinaryDelayPlan(family, ref_scalars, live, structural, (), kop)
