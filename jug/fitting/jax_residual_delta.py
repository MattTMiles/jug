"""JAX residual deltas and autodiff timing design matrices.

This module owns the differentiable residual path used by
``design_matrix_method="autodiff"``.  The design matrix is the Jacobian of the
same nonlinear residual-delta function used for JAX-native timing likelihoods;
there are no finite-difference perturbations and no hand-written derivative
columns in this path.

When ``use_jax_tempo2_native_chain=True``, tempo2-native fits recompute
``residual_sec(θ+Δθ) − residual_sec(θ)`` through ``compute_tempo2_toa_model_jax``
(clocks, ephemeris, BCLT / Roemer, DM, formBats, Shklovskii, binary closure,
spin / TRACK−2).  The Taylor ``dt_base + delay_change`` fallback is not used.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

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


def _spin_f_terms_jax(params: dict) -> jnp.ndarray:
    """Collect spin coefficients as a JAX vector (supports traced fit values)."""
    terms = []
    for i in range(10):
        key = f"F{i}"
        if key in params:
            terms.append(jnp.asarray(_param_scalar(params, key), dtype=jnp.float64))
        elif i == 0:
            terms.append(jnp.asarray(_param_scalar(params, "F0", 1.0), dtype=jnp.float64))
        else:
            break
    return jnp.stack(terms)


def _raj_decj_rad_jax(params: dict) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Equatorial sky position in radians from a (possibly traced) params dict."""
    if "_raj_rad" in params:
        alpha = jnp.asarray(params["_raj_rad"], dtype=jnp.float64)
    else:
        alpha = jnp.asarray(params["RAJ"], dtype=jnp.float64)
    if "_decj_rad" in params:
        delta = jnp.asarray(params["_decj_rad"], dtype=jnp.float64)
    else:
        delta = jnp.asarray(params["DECJ"], dtype=jnp.float64)
    return alpha, delta


def _pulsar_vectors_from_params_jax(
    params: dict,
    *,
    use_native_ecliptic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """JAX port of ``build_tempo2_pulsar_vectors`` for autodiff residual deltas."""
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
        pmra = jnp.asarray(_param_scalar(params, "PMRA"), dtype=jnp.float64)
        pmdec = jnp.asarray(_param_scalar(params, "PMDEC"), dtype=jnp.float64)
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
    pmra2 = jnp.asarray(_param_scalar(params, "PMRA2"), dtype=jnp.float64)
    pmdec2 = jnp.asarray(_param_scalar(params, "PMDEC2"), dtype=jnp.float64)
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
        coeffs.append(jnp.asarray(_param_scalar(params, key), dtype=jnp.float64))
        k += 1
    if not coeffs:
        coeffs = [jnp.asarray(0.0, dtype=jnp.float64)]
    return tuple(coeffs)


@dataclass(frozen=True)
class NativeDeltaPack:
    """Prepacked static inputs for tempo2-native full-chain residual deltas."""

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


def _load_native_delta_pack(setup: "GeneralFitSetup") -> NativeDeltaPack | None:
    """Build JAX-static cache for full-chain native residual deltas."""
    static = getattr(setup, "native_chain_static", None)
    if static is None:
        return None
    toas = static.get("toas")
    if not toas:
        return None

    from jug.delays.tempo2_geometry import pmrv_rad_per_century, tempo2_dilate_freq_enabled
    from jug.residuals.tempo2_native.chain_jax import _load_model_static_for_native_chain
    from jug.residuals.tempo2_native.model_jax import (
        _dm_coeffs_from_params,
        _eop_to_jax,
        _spk_to_jax,
    )
    from jug.utils.timescales import is_tempo2_si_units, parse_timescale

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
    model_static = _load_model_static_for_native_chain(params, toas, jug_result)
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


def _native_tempo2_residual_sec_jax(params: dict, pack: NativeDeltaPack) -> jnp.ndarray:
    """Recompute tempo2-native residuals through the unified JAX TOA model."""
    from jug.residuals.tempo2_native.model_jax import (
        compute_dm_vals_jax,
        compute_tempo2_toa_model_jax,
    )

    pos, vel, acc = _pulsar_vectors_from_params_jax(
        params, use_native_ecliptic=pack.use_native_ecliptic
    )
    f_terms = _spin_f_terms_jax(params)
    pepoch = jnp.asarray(_param_scalar(params, "PEPOCH"), dtype=jnp.float64)
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
        parallax_mas=jnp.asarray(_param_scalar(params, "PX"), dtype=jnp.float64),
        pmrv_rad_century=pack.pmrv_rad_century,
        dilate_freq=pack.dilate_freq,
        si_units=pack.si_units,
        units_tdb=pack.units_tdb,
        planet_shapiro_enabled=pack.planet_shapiro_enabled,
        track_val=pack.track_val,
        subtract_mean=False,
        dshk=pack.dshk,
        pmra=jnp.asarray(_param_scalar(params, "PMRA"), dtype=jnp.float64),
        pmdec=jnp.asarray(_param_scalar(params, "PMDEC"), dtype=jnp.float64),
        shk_posepoch=pack.shk_posepoch,
        jump_phase=pack.jump_phase,
        tzr_phase=pack.tzr_phase,
        pulse_numbers=pack.pulse_numbers,
        pn_add=pack.pn_add,
        correct_troposphere=pack.correct_troposphere,
    )
    return residual_sec


def _native_tempo2_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    pack: NativeDeltaPack,
):
    """Full native-chain residual delta: ``res(θ+Δθ) - res(θ)``.

    Both residuals are recomputed through ``compute_tempo2_toa_model_jax``
    (clocks, ephemeris geometry, BCLT / Roemer, DM, formBats, Shklovskii,
    binary closure, and spin / TRACK−2).
    """
    res_ref = _native_tempo2_residual_sec_jax(params_ref, pack)
    res_pert = _native_tempo2_residual_sec_jax(params_pert, pack)
    delta = res_pert - res_ref
    if pack.subtract_mean:
        delta = delta - jnp.mean(delta)
    return delta


def _compute_residual_delta_jax(
    params_ref: dict,
    params_pert: dict,
    setup: "GeneralFitSetup",
    *,
    native_pack: NativeDeltaPack | None,
    ref_f_terms: Sequence[float],
    phase_mean_mode: str,
    binary_plan=None,
):
    """Residual delta (perturbed - reference) through JUG's JAX forward model."""
    if (
        str(getattr(setup, "compatibility", "")).lower().startswith("tempo2")
        and getattr(setup, "use_jax_tempo2_native_chain", False)
    ):
        if native_pack is None:
            raise ValueError(
                "tempo2 native residual_delta requires native_chain_static on "
                "GeneralFitSetup; rebuild with USE_JAX_TEMPO2_NATIVE_CHAIN enabled"
            )
        return _native_tempo2_residual_delta_jax(params_ref, params_pert, native_pack)

    del native_pack
    from jug.fitting.forward_delay import compute_total_delay_change

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
    ecliptic_coords, obl_rad, ecliptic_init, native_family = _ecliptic_session_metadata(
        ref_params
    )
    native_pack = None
    if (
        str(setup.compatibility).lower().startswith("tempo2")
        and getattr(setup, "use_jax_tempo2_native_chain", False)
    ):
        native_pack = _load_native_delta_pack(setup)

    @jax.jit
    def _fn(delta_theta):
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

