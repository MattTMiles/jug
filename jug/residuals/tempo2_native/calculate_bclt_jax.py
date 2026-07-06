"""Tempo2 ``calculate_bclt.C`` iterative Roemer epoch (host + JAX helpers)."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
from jug.utils.constants import SECS_PER_DAY


@dataclass
class BcltTerms:
    """Converged BCLT delay terms per TOA."""

    roemer_sec: np.ndarray
    tdis1_sec: np.ndarray
    tdis2_sec: np.ndarray
    shapiro_sun_sec: np.ndarray
    shapiro_jupiter_sec: np.ndarray
    shapiro_planets_sec: np.ndarray
    dt_ssb_sec: np.ndarray
    bclt_iterations: np.ndarray
    converged: np.ndarray


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
    max_iter: int = 100,
    tol: float = 1.0e-10,
) -> BcltTerms:
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

    return BcltTerms(
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


def bclt_terms_to_jax(terms: BcltTerms) -> dict[str, jnp.ndarray]:
    """Convert host BCLT terms to JAX arrays."""
    return {
        name: jnp.asarray(getattr(terms, name))
        for name in BcltTerms.__dataclass_fields__
    }
