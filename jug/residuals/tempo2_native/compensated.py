"""Compensated float64 helpers for tempo2 MJD assembly in JAX."""

from __future__ import annotations

import jax.numpy as jnp

from jug.utils.constants import SECS_PER_DAY


def two_sum(a, b):
    """Error-free transform of a + b for JAX float64 arrays."""
    x = a + b
    eb = x - a
    ea = x - eb
    err = (a - ea) + (b - eb)
    return x, err


def kahan_sum(values, axis=0):
    """Compensated summation for small tempo2 clock/delay slots."""
    total = jnp.zeros_like(jnp.take(values, 0, axis=axis))
    comp = jnp.zeros_like(total)
    for idx in range(values.shape[axis]):
        y = jnp.take(values, idx, axis=axis) - comp
        t = total + y
        comp = (t - total) - y
        total = t
    return total


def assemble_mjd_from_day_sec(day_mjd, correction_sec):
    """Return MJD and residual day part for sat + correction_sec/86400."""
    corr_day = correction_sec / jnp.asarray(SECS_PER_DAY, dtype=jnp.float64)
    mjd_hi, mjd_lo = two_sum(day_mjd, corr_day)
    day = jnp.floor(mjd_hi)
    frac_hi, frac_lo = two_sum(mjd_hi, -day)
    residual = frac_hi + frac_lo + mjd_lo
    mjd = day + residual
    return mjd, mjd_lo
