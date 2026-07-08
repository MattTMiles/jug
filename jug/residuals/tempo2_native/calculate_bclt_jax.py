"""Tempo2 ``calculate_bclt.C`` iterative Roemer epoch (host + JAX helpers)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_geometry import (
    GM_C3,
    GMJ_C3,
    GMS_C3,
    GMN_C3,
    GMU_C3,
    GMV_C3,
    build_tempo2_pulsar_vectors,
    compute_tempo2_bclt_roemer_ls,
    compute_tempo2_dm_delays_sec,
    compute_tempo2_shapiro_sec,
    planet_shapiro_sec,
    pmrv_rad_per_century,
    psr_pos_at_delt,
    tempo2_dilate_freq_enabled,
)
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY

_PLANET_SHAP = (
    ("venus", GMV_C3),
    ("jupiter", GMJ_C3),
    ("saturn", GMS_C3),
    ("uranus", GMU_C3),
    ("neptune", GMN_C3),
)


@dataclass
class BcltTermsHost:
    """Host-side BCLT delay terms per TOA (NumPy reference path)."""

    roemer_sec: np.ndarray
    tdis1_sec: np.ndarray
    tdis2_sec: np.ndarray
    shapiro_sun_sec: np.ndarray
    shapiro_jupiter_sec: np.ndarray
    shapiro_planets_sec: np.ndarray
    dt_ssb_sec: np.ndarray
    bclt_iterations: np.ndarray
    converged: np.ndarray


class BcltTerms(NamedTuple):
    """Converged BCLT delay terms per TOA (JAX pytree)."""

    roemer_sec: jnp.ndarray
    tdis1_sec: jnp.ndarray
    tdis2_sec: jnp.ndarray
    shapiro_sun_sec: jnp.ndarray
    shapiro_jupiter_sec: jnp.ndarray
    shapiro_planets_sec: jnp.ndarray
    dt_ssb_sec: jnp.ndarray
    bclt_iterations: jnp.ndarray
    converged: jnp.ndarray


def bclt_delt_centuries(
    sat_mjd,
    posepoch_mjd,
    correction_tt_sec,
    correction_tt_tb_sec,
    dt_ssb_sec,
):
    """Tempo2 calculate_bclt.C L131-L132 epoch, not IFTE model_mjd."""
    clock_day = (correction_tt_sec + correction_tt_tb_sec + dt_ssb_sec) / SECS_PER_DAY
    return (sat_mjd - posepoch_mjd + clock_day) / 36525.0


def update_dt_ssb(
    roemer_sec,
    tdis1_sec,
    tdis2_sec,
    shapiro_sun_sec,
    shapiro_jupiter_sec,
    planet_shapiro: float,
):
    """Tempo2 calculate_bclt.C L160-L161 update."""
    dispersive = tdis1_sec + tdis2_sec
    shapiro_update = shapiro_sun_sec + planet_shapiro * shapiro_jupiter_sec
    return roemer_sec - dispersive - shapiro_update


def _dm_val_at_sat(sat_mjd: float, params: dict) -> float:
    dm_epoch = float(params.get("DMEPOCH", params["PEPOCH"]))
    dt_years = (sat_mjd - dm_epoch) / 365.25
    coeffs = []
    k = 0
    while True:
        key = "DM" if k == 0 else f"DM{k}"
        if key not in params:
            break
        coeffs.append(float(params[key]))
        k += 1
    if not coeffs:
        coeffs = [0.0]
    return sum(coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(coeffs)))


def compute_bclt_terms_numpy(
    *,
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    correction_tt_tb_sec: np.ndarray,
    observatory_earth_km: np.ndarray,
    params: dict,
    use_native_ecliptic: bool,
    planet_shapiro_enabled: bool,
    ssb_obs_ls_fixed: np.ndarray,
    obs_sun_ls_fixed: np.ndarray,
    obs_planets_ls_fixed: dict[str, np.ndarray] | None,
    freq_mhz: np.ndarray,
    earth_ssb_vel_km_s: np.ndarray,
    ne_sw: float = 0.0,
    einstein_rate: np.ndarray | None = None,
    site_vel_km_s: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1.0e-10,
) -> BcltTermsHost:
    """Host-side iterative BCLT loop with tempo2-fixed IFTE geometry.

    Mirrors ``calculate_bclt.C``: ``rca`` / ``sun_ssb`` / ``earth_ssb`` are fixed
    from the IFTE epoch; only ``delt`` and in-loop ``dm_delays`` vary per iteration.
    """
    n = len(sat_mjd)
    sat = np.asarray(sat_mjd, dtype=np.float64)
    tt = np.asarray(correction_tt_sec, dtype=np.float64)
    tt_tb = np.asarray(correction_tt_tb_sec, dtype=np.float64)
    ssb_obs_ls = np.asarray(ssb_obs_ls_fixed, dtype=np.float64)
    obs_sun_ls = np.asarray(obs_sun_ls_fixed, dtype=np.float64)
    earth_vel = np.asarray(earth_ssb_vel_km_s, dtype=np.float64)
    site_vel = (
        np.zeros_like(earth_vel)
        if site_vel_km_s is None
        else np.asarray(site_vel_km_s, dtype=np.float64)
    )
    freq = np.asarray(freq_mhz, dtype=np.float64)
    planets = obs_planets_ls_fixed or {}
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    parallax_mas = float(params.get("PX", 0.0))
    pmrv = pmrv_rad_per_century(float(params.get("PMRV", 0.0))) if "PMRV" in params else 0.0
    planet_shapiro = 1.0 if str(params.get("PLANET_SHAPIRO", "1")).upper() in ("1", "Y", "T", "TRUE") else 0.0
    dilate_freq = tempo2_dilate_freq_enabled(params)
    if einstein_rate is None and dilate_freq:
        from jug.delays.barycentric import compute_einstein_rate
        from jug.utils.timescales import is_tempo2_si_units, parse_timescale

        mjd_tt = sat + tt / SECS_PER_DAY
        units = parse_timescale(params)
        scale = "TCB" if is_tempo2_si_units(units) else "TDB"
        einstein = np.asarray(compute_einstein_rate(mjd_tt, units=scale), dtype=np.float64)
    elif einstein_rate is None:
        einstein = np.ones(n, dtype=np.float64)
    else:
        einstein = np.asarray(einstein_rate, dtype=np.float64)

    pos_pulsar, vel_pulsar, acc_pulsar = build_tempo2_pulsar_vectors(
        params,
        use_native_ecliptic=use_native_ecliptic,
    )

    roemer = np.zeros(n, dtype=np.float64)
    tdis1 = np.zeros(n, dtype=np.float64)
    tdis2 = np.zeros(n, dtype=np.float64)
    shap_sun = np.zeros(n, dtype=np.float64)
    shap_jup = np.zeros(n, dtype=np.float64)
    shap_planets = np.zeros(n, dtype=np.float64)
    dt_ssb = np.zeros(n, dtype=np.float64)
    iterations = np.zeros(n, dtype=np.int32)
    converged = np.zeros(n, dtype=bool)

    for i in range(n):
        dm_val = _dm_val_at_sat(float(sat[i]), params)
        dt_old = np.inf
        dt = 0.0
        it = 0
        roemer_i = 0.0
        tdis1_i = 0.0
        tdis2_i = 0.0
        shap_sun_i = 0.0
        shap_jup_i = 0.0
        shap_planets_i = 0.0
        while abs(dt - dt_old) > tol and it < max_iter:
            dt_old = dt
            delt = bclt_delt_centuries(sat[i], posepoch, tt[i], tt_tb[i], dt)
            psr_pos = psr_pos_at_delt(pos_pulsar, vel_pulsar, delt)
            roemer_i = compute_tempo2_bclt_roemer_ls(
                ssb_obs_ls[i],
                pos_pulsar,
                vel_pulsar,
                acc_pulsar,
                delt_centuries=delt,
                parallax_mas=parallax_mas,
                pmrv_rad_century=pmrv,
            )
            rsa_sun = -obs_sun_ls[i]
            shap_sun_i = compute_tempo2_shapiro_sec(
                rsa_sun,
                psr_pos,
                4.925490947e-6,
            )[0]
            shap_jup_i = 0.0
            if planet_shapiro_enabled and "jupiter" in planets:
                shap_jup_i = compute_tempo2_shapiro_sec(
                    planets["jupiter"][i],
                    psr_pos,
                    GMJ_C3_JAX,
                )[0]
            tdis1_i, tdis2_i = compute_tempo2_dm_delays_sec(
                sat_mjd=float(sat[i]),
                freq_mhz=float(freq[i]),
                psr_pos=psr_pos,
                obs_to_sun_ls=obs_sun_ls[i],
                earth_ssb_vel_km_s=earth_vel[i],
                dm_val=dm_val,
                ne_sw=float(ne_sw),
                dilate_freq=dilate_freq,
                einstein_rate=float(einstein[i]),
                site_vel_km_s=site_vel[i],
            )
            dt = update_dt_ssb(
                roemer_i,
                tdis1_i,
                tdis2_i,
                shap_sun_i,
                shap_jup_i,
                planet_shapiro,
            )
            it += 1

        delt_final = bclt_delt_centuries(sat[i], posepoch, tt[i], tt_tb[i], dt)
        psr_pos_final = psr_pos_at_delt(pos_pulsar, vel_pulsar, delt_final)
        shap_sun_i = float(
            compute_tempo2_shapiro_sec(-obs_sun_ls[i], psr_pos_final, GM_C3_JAX)[0]
        )
        shap_planets_i = 0.0
        if planet_shapiro_enabled:
            for name, gm in _PLANET_SHAP:
                if name not in planets:
                    continue
                rsa = np.asarray(planets[name][i], dtype=np.float64)
                if np.linalg.norm(rsa) <= 1e-20:
                    continue
                shap_planets_i += float(
                    compute_tempo2_shapiro_sec(rsa, psr_pos_final, gm)[0]
                )
        if "jupiter" in planets:
            shap_jup_i = float(
                compute_tempo2_shapiro_sec(
                    np.asarray(planets["jupiter"][i], dtype=np.float64),
                    psr_pos_final,
                    GMJ_C3_JAX,
                )[0]
            )

        roemer[i] = roemer_i
        tdis1[i] = tdis1_i
        tdis2[i] = tdis2_i
        shap_sun[i] = shap_sun_i
        shap_jup[i] = shap_jup_i
        shap_planets[i] = shap_planets_i
        dt_ssb[i] = dt
        iterations[i] = it
        converged[i] = abs(dt - dt_old) <= tol

    return BcltTermsHost(
        roemer_sec=roemer,
        tdis1_sec=tdis1,
        tdis2_sec=tdis2,
        shapiro_sun_sec=shap_sun,
        shapiro_jupiter_sec=shap_jup,
        shapiro_planets_sec=shap_planets,
        dt_ssb_sec=dt_ssb,
        bclt_iterations=iterations,
        converged=converged,
    )


def _dm_vals_numpy(sat_mjd: np.ndarray, params: dict) -> np.ndarray:
    n = len(sat_mjd)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        out[i] = _dm_val_at_sat(float(sat_mjd[i]), params)
    return out


def bclt_terms_to_jax(terms: BcltTermsHost) -> dict[str, jnp.ndarray]:
    """Convert host BCLT terms to JAX arrays."""
    return {
        name: jnp.asarray(getattr(terms, name))
        for name in terms.__dataclass_fields__
    }


# --- JAX production BCLT (vmap + fixed-length scan) ---

# Fixed iteration count for reverse-mode AD (NUTS/HMC). Host NumPy path keeps dynamic
# while-loop convergence; JAX always runs exactly this many steps.
DEFAULT_BCLT_JAX_FIXED_ITER = 12


def bclt_jax_fixed_iter_count(max_iter: int | None = None) -> int:
    """Return the fixed BCLT scan length for the JAX path."""
    if max_iter is not None:
        return int(max_iter)
    env = os.environ.get("JUG_TEMPO2_BCLT_FIXED_ITER", "").strip()
    if env:
        return int(env)
    return DEFAULT_BCLT_JAX_FIXED_ITER

GM_C3_JAX = GM_C3
GMJ_C3_JAX = GMJ_C3
GMS_C3_JAX = GMS_C3
GMU_C3_JAX = GMU_C3
GMN_C3_JAX = GMN_C3
GMV_C3_JAX = GMV_C3
_PLANET_SHAP_JAX = _PLANET_SHAP
PX_CONV_JAX = 1.74532925199432958e-2 / 3600.0e3
AULTSC_JAX = 499.00478364
K_DM_SEC_JAX = K_DM_SEC
_SAFE_NORM2_JAX = jnp.asarray(1.0e-60, dtype=jnp.float64)
_SOLAR_WIND_COS_EPS_JAX = jnp.asarray(1.0e-12, dtype=jnp.float64)


def _safe_norm_jax(x):
    """Norm with a finite derivative at the zero vector."""
    return jnp.sqrt(jnp.maximum(jnp.dot(x, x), _SAFE_NORM2_JAX))


def _solar_wind_angular_factor_jax(ctheta):
    """Return acos(c) / sqrt(1 - c**2) without singular AD tangents."""
    c_safe = jnp.clip(
        ctheta,
        -1.0 + _SOLAR_WIND_COS_EPS_JAX,
        1.0 - _SOLAR_WIND_COS_EPS_JAX,
    )
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - c_safe * c_safe, _SAFE_NORM2_JAX))
    return jnp.arccos(c_safe) / sin_theta


def _bclt_delt_jax(sat, posepoch, tt, tt_tb, dt):
    clock_day = (tt + tt_tb + dt) / SECS_PER_DAY
    return (sat - posepoch + clock_day) / 36525.0


def _psr_pos_jax(pos, vel, delt):
    p = pos + delt * vel
    return p / jnp.maximum(_safe_norm_jax(p), 1e-30)


def _roemer_ls_jax(rca, pos, vel, acc, delt, parallax_mas, pmrv):
    rcos1 = jnp.dot(pos, rca)
    rr = jnp.dot(rca, rca)
    pmtrans_rcos2 = jnp.dot(vel, rca)
    pmtrans = _safe_norm_jax(vel)
    dt_pm = delt * pmtrans_rcos2
    dt_pmtt = -0.5 * pmtrans * pmtrans * delt * delt * rcos1
    dt_acctrans = 0.5 * delt * delt * jnp.dot(acc, rca)
    dt_px = jnp.where(
        parallax_mas != 0.0,
        -0.5 * parallax_mas * PX_CONV_JAX * (rr - rcos1 * rcos1) / AULTSC_JAX,
        0.0,
    )
    dt_pmtr = -delt * delt * pmrv * pmtrans_rcos2
    return rcos1 + dt_pm + dt_pmtt + dt_px + dt_pmtr + dt_acctrans


def _shapiro_jax(rsa, psr_pos, gm_c3):
    r = _safe_norm_jax(rsa)
    ctheta = jnp.dot(psr_pos, rsa) / jnp.maximum(r, 1e-30)
    return -2.0 * gm_c3 * jnp.log(jnp.maximum(r / AULTSC_JAX * (1.0 + ctheta), 1e-30))


def _dm_jax(
    freq_mhz,
    dm_val,
    psr_pos,
    obs_sun_ls,
    earth_vel,
    site_vel,
    einstein,
    dilate_freq,
    ne_sw,
):
    rsa = -obs_sun_ls
    vobs = earth_vel / 299792.458 + site_vel / 299792.458
    r = _safe_norm_jax(rsa)
    ctheta = jnp.dot(psr_pos, rsa) / jnp.maximum(r, 1e-30)
    voverc = jnp.dot(psr_pos, vobs)
    freqf = freq_mhz * 1.0e6 * (1.0 - voverc)
    freqf = jnp.where(dilate_freq & (einstein != 0.0), freqf / einstein, freqf)
    tdis1 = jnp.where(freqf > 1.0, dm_val * K_DM_SEC_JAX / ((freqf / 1.0e6) ** 2), 0.0)
    solar_wind_angle = _solar_wind_angular_factor_jax(ctheta)
    tdis2 = jnp.where(
        (ne_sw != 0.0) & (freqf > 1.0) & (r > 0.0),
        ne_sw
        * 1.0e6
        * 1.49598e11
        * 1.49598e11
        / 299792458.0
        / 7.436e6
        * solar_wind_angle
        / r
        / freqf
        / freqf,
        0.0,
    )
    return tdis1, tdis2


def _sum_planet_shapiro_jax(
    psr_pos: jnp.ndarray,
    planet_rsa_ls: tuple[jnp.ndarray, ...],
    *,
    enabled: bool,
) -> jnp.ndarray:
    """Sum planetary Shapiro at fixed ``psrPos`` (``shapiro_delay.C`` export path)."""
    total = jnp.array(0.0, dtype=jnp.float64)
    if not enabled:
        return total
    for (_, gm), rsa in zip(_PLANET_SHAP_JAX, planet_rsa_ls):
        r = _safe_norm_jax(rsa)
        contrib = _shapiro_jax(rsa, psr_pos, gm)
        total = total + jnp.where(r > 1e-20, contrib, 0.0)
    return total


def _bclt_step_jax(
    dt: jnp.ndarray,
    *,
    sat,
    posepoch_mjd,
    tt,
    tt_tb,
    rca,
    osun,
    venus_rsa,
    jup_rsa,
    sat_rsa,
    ura_rsa,
    nep_rsa,
    freq,
    earth_vel,
    site_vel,
    dm_val,
    einstein_i,
    pos_pulsar,
    vel_pulsar,
    acc_pulsar,
    parallax_mas,
    pmrv_rad_century,
    dilate_freq,
    planet_shapiro_enabled,
    ne_sw,
):
    """One BCLT fixed-point update: ``dt -> new_dt`` plus in-loop delay terms."""
    planet_rsa = (venus_rsa, jup_rsa, sat_rsa, ura_rsa, nep_rsa)
    sun_rsa = -osun
    delt = _bclt_delt_jax(sat, posepoch_mjd, tt, tt_tb, dt)
    psr_pos = _psr_pos_jax(pos_pulsar, vel_pulsar, delt)
    roemer_ls = _roemer_ls_jax(
        rca, pos_pulsar, vel_pulsar, acc_pulsar, delt, parallax_mas, pmrv_rad_century
    )
    shap_sun = _shapiro_jax(sun_rsa, psr_pos, GM_C3_JAX)
    shap_jup = _shapiro_jax(jup_rsa, psr_pos, GMJ_C3_JAX)
    tdis1_i, tdis2_i = _dm_jax(
        freq, dm_val, psr_pos, osun, earth_vel, site_vel, einstein_i, dilate_freq, ne_sw
    )
    shap_update = shap_sun + jnp.where(
        planet_shapiro_enabled & (_safe_norm_jax(jup_rsa) > 1e-20),
        shap_jup,
        0.0,
    )
    new_dt = roemer_ls - (tdis1_i + tdis2_i) - shap_update
    return new_dt, roemer_ls, tdis1_i, tdis2_i, shap_sun, shap_jup


def compute_bclt_terms_jax(
    *,
    sat_mjd: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    correction_tt_tb_sec: jnp.ndarray,
    ssb_obs_ls: jnp.ndarray,
    obs_sun_ls: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    earth_ssb_vel_km_s: jnp.ndarray,
    site_vel_km_s: jnp.ndarray,
    dm_vals: jnp.ndarray,
    pos_pulsar: jnp.ndarray,
    vel_pulsar: jnp.ndarray,
    acc_pulsar: jnp.ndarray,
    posepoch_mjd: float,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
    ne_sw: float = 0.0,
    einstein_rate: jnp.ndarray | None = None,
    dilate_freq: bool = False,
    planet_shapiro_enabled: bool = True,
    obs_jupiter_ls: jnp.ndarray | None = None,
    planet_obs_ls: tuple[jnp.ndarray, ...] | None = None,
    max_iter: int | None = None,
    tol: float = 1.0e-10,
) -> BcltTerms:
    """JAX BCLT: ``vmap`` over TOAs with fixed-length ``lax.scan`` (reverse-mode safe)."""
    fixed_iter = bclt_jax_fixed_iter_count(max_iter)
    n = sat_mjd.shape[0]
    einstein = jnp.ones(n, dtype=jnp.float64) if einstein_rate is None else einstein_rate
    if planet_obs_ls is None:
        zeros = jnp.zeros((n, 3), dtype=jnp.float64)
        jup_fill = (
            -obs_jupiter_ls if obs_jupiter_ls is not None else zeros
        )
        planet_obs_ls = (zeros, jup_fill, zeros, zeros, zeros)
    venus_rsa, jup_rsa, sat_rsa, ura_rsa, nep_rsa = planet_obs_ls
    del obs_jupiter_ls

    def single_bclt(
        sat,
        tt,
        tt_tb,
        rca,
        osun,
        venus_rsa,
        jup_rsa,
        sat_rsa,
        ura_rsa,
        nep_rsa,
        freq,
        earth_vel,
        site_vel,
        dm_val,
        einstein_i,
    ):
        planet_rsa = (venus_rsa, jup_rsa, sat_rsa, ura_rsa, nep_rsa)

        def scan_body(carry, _):
            dt_old, dt, it, first_converged, has_converged, roemer, tdis1, tdis2, shap_sun, shap_jup = (
                carry
            )
            new_dt, roemer, tdis1, tdis2, shap_sun, shap_jup = _bclt_step_jax(
                dt,
                sat=sat,
                posepoch_mjd=posepoch_mjd,
                tt=tt,
                tt_tb=tt_tb,
                rca=rca,
                osun=osun,
                venus_rsa=venus_rsa,
                jup_rsa=jup_rsa,
                sat_rsa=sat_rsa,
                ura_rsa=ura_rsa,
                nep_rsa=nep_rsa,
                freq=freq,
                earth_vel=earth_vel,
                site_vel=site_vel,
                dm_val=dm_val,
                einstein_i=einstein_i,
                pos_pulsar=pos_pulsar,
                vel_pulsar=vel_pulsar,
                acc_pulsar=acc_pulsar,
                parallax_mas=parallax_mas,
                pmrv_rad_century=pmrv_rad_century,
                dilate_freq=dilate_freq,
                planet_shapiro_enabled=planet_shapiro_enabled,
                ne_sw=ne_sw,
            )
            err = jnp.abs(new_dt - dt)
            converged_now = err <= tol
            first_converged = jnp.where(
                (~has_converged) & converged_now,
                it + 1,
                first_converged,
            )
            has_converged = has_converged | converged_now
            new_carry = (
                dt,
                new_dt,
                it + 1,
                first_converged,
                has_converged,
                roemer,
                tdis1,
                tdis2,
                shap_sun,
                shap_jup,
            )
            return new_carry, None

        init_carry = (
            jnp.array(jnp.inf, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
        )
        final, _ = jax.lax.scan(scan_body, init_carry, None, length=fixed_iter)
        dt_old, dt, it, first_converged, has_converged, roemer, tdis1, tdis2, _shap_sun, shap_jup = (
            final
        )
        delt_final = _bclt_delt_jax(sat, posepoch_mjd, tt, tt_tb, dt)
        psr_pos_final = _psr_pos_jax(pos_pulsar, vel_pulsar, delt_final)
        shap_sun_final = _shapiro_jax(-osun, psr_pos_final, GM_C3_JAX)
        shap_jup_final = _shapiro_jax(jup_rsa, psr_pos_final, GMJ_C3_JAX)
        shap_planets_final = _sum_planet_shapiro_jax(
            psr_pos_final,
            planet_rsa,
            enabled=planet_shapiro_enabled,
        )
        iters = jnp.where(first_converged > 0, first_converged, jnp.int32(fixed_iter))
        converged = has_converged | (jnp.abs(dt - dt_old) <= tol)
        return (
            roemer,
            tdis1,
            tdis2,
            shap_sun_final,
            shap_jup_final,
            shap_planets_final,
            dt,
            iters,
            converged,
        )

    outs = jax.vmap(single_bclt)(
        sat_mjd,
        correction_tt_sec,
        correction_tt_tb_sec,
        ssb_obs_ls,
        obs_sun_ls,
        venus_rsa,
        jup_rsa,
        sat_rsa,
        ura_rsa,
        nep_rsa,
        freq_mhz,
        earth_ssb_vel_km_s,
        site_vel_km_s,
        dm_vals,
        einstein,
    )
    roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets, dt_ssb, iters, converged = outs
    return BcltTerms(
        roemer_sec=roemer,
        tdis1_sec=tdis1,
        tdis2_sec=tdis2,
        shapiro_sun_sec=shap_sun,
        shapiro_jupiter_sec=shap_jup,
        shapiro_planets_sec=shap_planets,
        dt_ssb_sec=dt_ssb,
        bclt_iterations=iters.astype(jnp.int32),
        converged=converged,
    )


def compute_bclt_terms_fixed_state_jax(
    *,
    sat_mjd: jnp.ndarray,
    correction_tt_sec: jnp.ndarray,
    correction_tt_tb_sec: jnp.ndarray,
    dt_ssb_ref_sec: jnp.ndarray,
    ssb_obs_ls: jnp.ndarray,
    obs_sun_ls: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    earth_ssb_vel_km_s: jnp.ndarray,
    site_vel_km_s: jnp.ndarray,
    dm_vals: jnp.ndarray,
    pos_pulsar: jnp.ndarray,
    vel_pulsar: jnp.ndarray,
    acc_pulsar: jnp.ndarray,
    posepoch_mjd: float,
    parallax_mas: float = 0.0,
    pmrv_rad_century: float = 0.0,
    ne_sw: float = 0.0,
    einstein_rate: jnp.ndarray | None = None,
    dilate_freq: bool = False,
    planet_shapiro_enabled: bool = True,
    obs_jupiter_ls: jnp.ndarray | None = None,
    planet_obs_ls: tuple[jnp.ndarray, ...] | None = None,
) -> BcltTerms:
    """One-pass nonlinear tempo2 BCLT terms at frozen reference dt_ssb."""
    n = sat_mjd.shape[0]
    einstein = jnp.ones(n, dtype=jnp.float64) if einstein_rate is None else einstein_rate
    if planet_obs_ls is None:
        zeros = jnp.zeros((n, 3), dtype=jnp.float64)
        jup_fill = -obs_jupiter_ls if obs_jupiter_ls is not None else zeros
        planet_obs_ls = (zeros, jup_fill, zeros, zeros, zeros)
    venus_rsa, jup_rsa, sat_rsa, ura_rsa, nep_rsa = planet_obs_ls

    def single(
        sat,
        tt,
        tt_tb,
        dt_ref,
        rca,
        osun,
        venus,
        jup,
        saturn,
        ura,
        nep,
        freq,
        earth_vel,
        site_vel,
        dm_val,
        einstein_i,
    ):
        planet_rsa = (venus, jup, saturn, ura, nep)
        delt = _bclt_delt_jax(sat, posepoch_mjd, tt, tt_tb, dt_ref)
        psr_pos = _psr_pos_jax(pos_pulsar, vel_pulsar, delt)
        roemer = _roemer_ls_jax(
            rca, pos_pulsar, vel_pulsar, acc_pulsar, delt, parallax_mas, pmrv_rad_century
        )
        shap_sun = _shapiro_jax(-osun, psr_pos, GM_C3_JAX)
        shap_jup = _shapiro_jax(jup, psr_pos, GMJ_C3_JAX)
        shap_planets = _sum_planet_shapiro_jax(
            psr_pos, planet_rsa, enabled=planet_shapiro_enabled
        )
        tdis1, tdis2 = _dm_jax(
            freq, dm_val, psr_pos, osun, earth_vel, site_vel, einstein_i, dilate_freq, ne_sw
        )
        dt_ssb = update_dt_ssb(
            roemer,
            tdis1,
            tdis2,
            shap_sun,
            shap_jup,
            jnp.where(planet_shapiro_enabled, 1.0, 0.0),
        )
        return roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets, dt_ssb

    roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets, dt_ssb = jax.vmap(single)(
        sat_mjd,
        correction_tt_sec,
        correction_tt_tb_sec,
        dt_ssb_ref_sec,
        ssb_obs_ls,
        obs_sun_ls,
        venus_rsa,
        jup_rsa,
        sat_rsa,
        ura_rsa,
        nep_rsa,
        freq_mhz,
        earth_ssb_vel_km_s,
        site_vel_km_s,
        dm_vals,
        einstein,
    )
    return BcltTerms(
        roemer_sec=roemer,
        tdis1_sec=tdis1,
        tdis2_sec=tdis2,
        shapiro_sun_sec=shap_sun,
        shapiro_jupiter_sec=shap_jup,
        shapiro_planets_sec=shap_planets,
        dt_ssb_sec=dt_ssb,
        bclt_iterations=jnp.ones(n, dtype=jnp.int32),
        converged=jnp.ones(n, dtype=bool),
    )
