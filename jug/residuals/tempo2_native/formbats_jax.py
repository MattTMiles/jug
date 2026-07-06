"""Tempo2 ``formBats.C`` assembly in JAX."""

from __future__ import annotations

import jax.numpy as jnp

from jug.residuals.tempo2_native.compensated import assemble_mjd_from_day_sec, two_sum
from jug.utils.constants import C_KM_S, SECS_PER_DAY


def compute_formbats_jax(
    sat_mjd,
    correction_tt_sec,
    correction_tt_tb_sec,
    tropospheric_sec,
    roemer_sec,
    shapiro_delay_sec,
    tdis1_sec,
    tdis2_sec,
    shklovskii_sec,
):
    """Port formBats.C L67-L83 with separate tdis1/tdis2 slots."""
    correction_sec = correction_tt_sec + (
        correction_tt_tb_sec
        - tropospheric_sec
        + roemer_sec
        - shapiro_delay_sec
        - tdis1_sec
        - tdis2_sec
    )
    bat_corr_day, bat_corr_resid = two_sum(correction_sec / SECS_PER_DAY, 0.0)
    bat_mjd, bat_resid = assemble_mjd_from_day_sec(sat_mjd, correction_sec)
    bbat_mjd, bbat_resid = assemble_mjd_from_day_sec(bat_mjd, -shklovskii_sec)
    bat_corr_total_resid = bat_corr_resid + bat_resid + bbat_resid
    return bat_corr_day, bat_corr_total_resid, bat_mjd, bbat_mjd


def compute_shklovskii_sec_jax(bat_mjd, params):
    """JAX Shklovskii delay; host params dict with float values."""
    bat = jnp.asarray(bat_mjd, dtype=jnp.float64)
    if "DSHK" not in params:
        return jnp.zeros_like(bat)
    if "PMRA" not in params and "PMDEC" not in params:
        return jnp.zeros_like(bat)

    kpc2m = 3.08568025e19
    mas_yr2rad_s = 1.536281850e-16
    posepoch = float(params.get("POSEPOCH", params["PEPOCH"]))
    dshk = float(params.get("DSHK", 0.0))
    pmra = float(params.get("PMRA", 0.0))
    pmdec = float(params.get("PMDEC", 0.0))
    t0 = (bat - posepoch) * SECS_PER_DAY
    pm2 = (pmra * pmra + pmdec * pmdec) * mas_yr2rad_s * mas_yr2rad_s
    return (t0 * t0 / (2.0 * C_KM_S)) * (dshk * kpc2m) * pm2


def compute_shklovskii_sec_jax_pure(
    bat_mjd,
    pepoch_mjd,
    f_terms,
    *,
    dshk: float = 0.0,
    pmra: float = 0.0,
    pmdec: float = 0.0,
    posepoch_mjd: float | None = None,
) -> jnp.ndarray:
    """Pure JAX Shklovskii without host dict lookup."""
    if dshk == 0.0:
        return jnp.zeros_like(jnp.asarray(bat_mjd, dtype=jnp.float64))
    bat = jnp.asarray(bat_mjd, dtype=jnp.float64)
    kpc2m = 3.08568025e19
    mas_yr2rad_s = 1.536281850e-16
    pep = float(posepoch_mjd if posepoch_mjd is not None else pepoch_mjd)
    t0 = (bat - pep) * SECS_PER_DAY
    pm2 = (pmra * pmra + pmdec * pmdec) * mas_yr2rad_s * mas_yr2rad_s
    return (t0 * t0 / (2.0 * C_KM_S)) * (dshk * kpc2m) * pm2


def compute_torb_closure_jax(bbat_mjd, dt_emission_sec, pepoch_mjd):
    """Tempo2 formResiduals.C closure: deltaT = (bbat - PEPOCH)*86400 + torb."""
    return dt_emission_sec - (bbat_mjd - pepoch_mjd) * jnp.asarray(
        SECS_PER_DAY,
        dtype=jnp.float64,
    )
