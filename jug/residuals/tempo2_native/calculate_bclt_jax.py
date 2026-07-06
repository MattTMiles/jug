"""Tempo2 ``calculate_bclt.C`` iterative Roemer epoch (host + JAX helpers)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from jug.delays.tempo2_geometry import (
    build_tempo2_pulsar_vectors,
    compute_tempo2_bclt_roemer_ls,
    compute_tempo2_dm_delays_sec,
    compute_tempo2_shapiro_sec,
    planet_shapiro_sec,
    pmrv_rad_per_century,
    psr_pos_at_delt,
)
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY


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
    einstein = (
        np.ones(n, dtype=np.float64)
        if einstein_rate is None
        else np.asarray(einstein_rate, dtype=np.float64)
    )

    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    parallax_mas = float(params.get("PX", 0.0))
    pmrv = pmrv_rad_per_century(float(params.get("PMRV", 0.0)))
    planet_shapiro = 1.0 if planet_shapiro_enabled else 0.0
    dilate_freq = str(params.get("DILATEFREQ", "N")).upper() in ("Y", "YES", "TRUE", "1")

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
            if "jupiter" in planets:
                shap_jup_i = compute_tempo2_shapiro_sec(
                    -planets["jupiter"][i],
                    psr_pos,
                    4.70255e-9,
                )[0]
            shap_planets_i = planet_shapiro_sec(
                {name: arr[i : i + 1] for name, arr in planets.items()},
                psr_pos.reshape(1, 3),
                enabled=planet_shapiro_enabled,
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


# --- JAX production BCLT (vmap + while_loop) ---

GM_C3_JAX = 4.925490947e-6
GMJ_C3_JAX = 4.70255e-9
PX_CONV_JAX = 1.74532925199432958e-2 / 3600.0e3
AULTSC_JAX = 499.00478364
K_DM_SEC_JAX = K_DM_SEC


def _bclt_delt_jax(sat, posepoch, tt, tt_tb, dt):
    clock_day = (tt + tt_tb + dt) / SECS_PER_DAY
    return (sat - posepoch + clock_day) / 36525.0


def _psr_pos_jax(pos, vel, delt):
    p = pos + delt * vel
    return p / jnp.maximum(jnp.linalg.norm(p), 1e-30)


def _roemer_ls_jax(rca, pos, vel, acc, delt, parallax_mas, pmrv):
    rcos1 = jnp.dot(pos, rca)
    rr = jnp.dot(rca, rca)
    pmtrans_rcos2 = jnp.dot(vel, rca)
    pmtrans = jnp.linalg.norm(vel)
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
    r = jnp.linalg.norm(rsa)
    ctheta = jnp.dot(psr_pos, rsa) / jnp.maximum(r, 1e-30)
    return -2.0 * gm_c3 * jnp.log(jnp.maximum(r / AULTSC_JAX * (1.0 + ctheta), 1e-30))


def _dm_jax(freq_mhz, dm_val, psr_pos, obs_sun_ls, earth_vel, site_vel, einstein, dilate_freq, ne_sw):
    rsa = -obs_sun_ls
    vobs = earth_vel / 299792.458 + site_vel / 299792.458
    r = jnp.linalg.norm(rsa)
    ctheta = jnp.dot(psr_pos, rsa) / jnp.maximum(r, 1e-30)
    voverc = jnp.dot(psr_pos, vobs)
    freqf = freq_mhz * 1.0e6 * (1.0 - voverc)
    freqf = jnp.where(dilate_freq & (einstein != 0.0), freqf / einstein, freqf)
    tdis1 = jnp.where(freqf > 1.0, dm_val * K_DM_SEC_JAX / ((freqf / 1.0e6) ** 2), 0.0)
    tdis2 = jnp.where(
        (ne_sw != 0.0) & (freqf > 1.0) & (r > 0.0),
        ne_sw
        * 1.0e6
        * 1.49598e11
        * 1.49598e11
        / 299792458.0
        / 7.436e6
        * jnp.arccos(jnp.clip(ctheta, -1.0, 1.0))
        / jnp.maximum(jnp.sqrt(jnp.maximum(1.0 - ctheta * ctheta, 0.0)), 1e-30)
        / r
        / freqf
        / freqf,
        0.0,
    )
    return tdis1, tdis2


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
    max_iter: int = 100,
    tol: float = 1.0e-10,
) -> BcltTerms:
    """JAX BCLT iteration: ``vmap`` over TOAs with ``lax.while_loop``."""
    n = sat_mjd.shape[0]
    einstein = jnp.ones(n, dtype=jnp.float64) if einstein_rate is None else einstein_rate
    jup = (
        jnp.zeros((n, 3), dtype=jnp.float64)
        if obs_jupiter_ls is None
        else obs_jupiter_ls
    )

    def single_bclt(sat, tt, tt_tb, rca, osun, jup_ls, freq, earth_vel, site_vel, dm_val, einstein_i):
        def cond(state):
            dt_old, dt, it, *_ = state
            return (jnp.abs(dt - dt_old) > tol) & (it < max_iter)

        def body(state):
            dt_old, dt, it, roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets = state
            delt = _bclt_delt_jax(sat, posepoch_mjd, tt, tt_tb, dt)
            psr_pos = _psr_pos_jax(pos_pulsar, vel_pulsar, delt)
            roemer_ls = _roemer_ls_jax(
                rca, pos_pulsar, vel_pulsar, acc_pulsar, delt, parallax_mas, pmrv_rad_century
            )
            shap_sun = _shapiro_jax(-osun, psr_pos, GM_C3_JAX)
            shap_jup = _shapiro_jax(-jup_ls, psr_pos, GMJ_C3_JAX)
            tdis1_i, tdis2_i = _dm_jax(
                freq, dm_val, psr_pos, osun, earth_vel, site_vel, einstein_i, dilate_freq, ne_sw
            )
            shap_planets = jnp.where(planet_shapiro_enabled, shap_jup, 0.0)
            new_dt = roemer_ls - (tdis1_i + tdis2_i) - (shap_sun + shap_planets)
            return (dt, new_dt, it + 1, roemer_ls, tdis1_i, tdis2_i, shap_sun, shap_jup, shap_planets)

        init = (
            jnp.inf,
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0, dtype=jnp.int32),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
            jnp.array(0.0, dtype=jnp.float64),
        )
        final = jax.lax.while_loop(cond, body, init)
        dt_old, dt, it, roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets = final
        converged = jnp.abs(dt - dt_old) <= tol
        return roemer, tdis1, tdis2, shap_sun, shap_jup, shap_planets, dt, it, converged

    outs = jax.vmap(single_bclt)(
        sat_mjd,
        correction_tt_sec,
        correction_tt_tb_sec,
        ssb_obs_ls,
        obs_sun_ls,
        jup,
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
