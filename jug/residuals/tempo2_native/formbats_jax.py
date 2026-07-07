"""Tempo2 ``formBats.C`` assembly in JAX.

Production spin uses two-part ``(int_day, sec_in_day)`` barycentric time so
``formResiduals.C`` phase2/phase3 match tempo2 ``long double`` semantics.
Diagnostic ``bat_mjd`` / ``bbat_mjd`` views remain single float64 for oracles.
"""

from __future__ import annotations

import jax.numpy as jnp

from jug.residuals.tempo2_native.compensated import (
    add_seconds_daysec,
    assemble_mjd_from_day_sec,
    mjd_view_from_daysec,
    split_mjd_to_daysec,
    two_sum,
)
from jug.utils.constants import C_KM_S, SECS_PER_DAY


def compute_formbats_daysec(
    sat_int_day,
    sat_sec_in_day,
    correction_tt_sec,
    correction_tt_tb_sec,
    tropospheric_sec,
    roemer_sec,
    shapiro_delay_sec,
    tdis1_sec,
    tdis2_sec,
    shklovskii_sec,
):
    """Port ``formBats.C`` L67-L83 with two-part sat/bat/bbat."""
    correction_sec = correction_tt_sec + (
        correction_tt_tb_sec
        - tropospheric_sec
        + roemer_sec
        - shapiro_delay_sec
        - tdis1_sec
        - tdis2_sec
    )
    bat_int, bat_sec = add_seconds_daysec(sat_int_day, sat_sec_in_day, correction_sec)
    bbat_int, bbat_sec = add_seconds_daysec(bat_int, bat_sec, -shklovskii_sec)
    bat_corr_day, bat_corr_resid = two_sum(correction_sec / SECS_PER_DAY, 0.0)
    return (
        bat_corr_day,
        bat_corr_resid,
        bat_int,
        bat_sec,
        bbat_int,
        bbat_sec,
    )


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
    """Legacy single-MJD wrapper; prefer :func:`compute_formbats_daysec`."""
    sat_int, sat_sec = split_mjd_to_daysec(sat_mjd)
    (
        bat_corr_day,
        bat_corr_resid,
        bat_int,
        bat_sec,
        bbat_int,
        bbat_sec,
    ) = compute_formbats_daysec(
        sat_int,
        sat_sec,
        correction_tt_sec,
        correction_tt_tb_sec,
        tropospheric_sec,
        roemer_sec,
        shapiro_delay_sec,
        tdis1_sec,
        tdis2_sec,
        shklovskii_sec,
    )
    bat_mjd = mjd_view_from_daysec(bat_int, bat_sec)
    bbat_mjd = mjd_view_from_daysec(bbat_int, bbat_sec)
    return bat_corr_day, bat_corr_resid, bat_mjd, bbat_mjd


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


def compute_shklovskii_sec_jax_pure_daysec(
    bat_int,
    bat_sec,
    posepoch_int,
    posepoch_frac,
    *,
    dshk: float = 0.0,
    pmra: float = 0.0,
    pmdec: float = 0.0,
) -> jnp.ndarray:
    """Pure JAX Shklovskii from two-part BAT and POSEPOCH."""
    if dshk == 0.0:
        return jnp.zeros_like(jnp.asarray(bat_sec, dtype=jnp.float64))
    kpc2m = 3.08568025e19
    mas_yr2rad_s = 1.536281850e-16
    t0 = (bat_int - posepoch_int) * SECS_PER_DAY + bat_sec - posepoch_frac * SECS_PER_DAY
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
    pep = float(posepoch_mjd if posepoch_mjd is not None else pepoch_mjd)
    bat_int, bat_sec = split_mjd_to_daysec(bat_mjd)
    pep_int = jnp.floor(jnp.asarray(pep, dtype=jnp.float64))
    pep_frac = jnp.asarray(pep, dtype=jnp.float64) - pep_int
    return compute_shklovskii_sec_jax_pure_daysec(
        bat_int,
        bat_sec,
        pep_int,
        pep_frac,
        dshk=dshk,
        pmra=pmra,
        pmdec=pmdec,
    )


def compute_torb_closure_daysec(bbat_int, bbat_sec, dt_emission_sec, pep_int, pep_frac):
    """Tempo2 ``formResiduals.C`` closure with two-part bbat and PEPOCH."""
    ntpd = bbat_int - pep_int
    return dt_emission_sec - (
        ntpd * jnp.asarray(SECS_PER_DAY, dtype=jnp.float64)
        + bbat_sec
        - pep_frac * jnp.asarray(SECS_PER_DAY, dtype=jnp.float64)
    )


def compute_torb_closure_jax(bbat_mjd, dt_emission_sec, pepoch_mjd):
    """Legacy single-MJD torb closure; prefer :func:`compute_torb_closure_daysec`."""
    bbat_int, bbat_sec = split_mjd_to_daysec(bbat_mjd)
    pep_int = jnp.floor(jnp.asarray(pepoch_mjd, dtype=jnp.float64))
    pep_frac = jnp.asarray(pepoch_mjd, dtype=jnp.float64) - pep_int
    return compute_torb_closure_daysec(bbat_int, bbat_sec, dt_emission_sec, pep_int, pep_frac)
