"""Tempo2 ``formBats.C`` assembly in JAX."""

from __future__ import annotations

import jax.numpy as jnp

from jug.residuals.tempo2_native.compensated import assemble_mjd_from_day_sec, two_sum
from jug.utils.constants import SECS_PER_DAY


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
    import numpy as np

    bat = np.asarray(bat_mjd)
    from jug.residuals.tempo2_clock import compute_shklovskii_sec

    return jnp.asarray(compute_shklovskii_sec(bat, params), dtype=jnp.float64)


def compute_torb_closure_jax(bbat_mjd, dt_emission_sec, pepoch_mjd):
    """Tempo2 formResiduals.C closure: deltaT = (bbat - PEPOCH)*86400 + torb."""
    return dt_emission_sec - (bbat_mjd - pepoch_mjd) * jnp.asarray(
        SECS_PER_DAY,
        dtype=jnp.float64,
    )
