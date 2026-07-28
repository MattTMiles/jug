"""Compensated float64 helpers for tempo2 MJD assembly in JAX."""

from __future__ import annotations

import jax.numpy as jnp

from jug.utils.constants import SECS_PER_DAY

_SPLIT = 134217729.0  # 2**27 + 1, Veltkamp split for float64


def two_sum(a, b):
    """Error-free transform of a + b for JAX float64 arrays."""
    x = a + b
    eb = x - a
    ea = x - eb
    err = (a - ea) + (b - eb)
    return x, err


def _veltkamp_split(a):
    c = _SPLIT * a
    hi = c - (c - a)
    return hi, a - hi


def two_prod(a, b):
    """Error-free transform: ``a * b == p + err`` (double-double product)."""
    p = a * b
    ah, al = _veltkamp_split(a)
    bh, bl = _veltkamp_split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def frac_of_int_times(n_int, x):
    """Fractional part of ``n_int * x`` to ~1e-16 turn precision."""
    p_hi, p_lo = two_prod(n_int, x)
    k = jnp.round(p_hi)
    return (p_hi - k) + p_lo


def add_seconds_daysec(int_day, sec_in_day, correction_sec):
    """Add seconds to a two-part MJD; normalize ``sec_in_day`` into [0, 86400)."""
    sec = sec_in_day + correction_sec
    carry = jnp.floor(sec / SECS_PER_DAY)
    return int_day + carry, sec - carry * SECS_PER_DAY


def split_mjd_to_daysec(mjd):
    """Split float64 MJD into ``(int_day, sec_in_day)``."""
    int_day = jnp.floor(mjd)
    return int_day, (mjd - int_day) * SECS_PER_DAY


def mjd_view_from_daysec(int_day, sec_in_day):
    """Diagnostic single-float64 MJD view matching tempo2 split assembly."""
    frac_day = sec_in_day / SECS_PER_DAY
    mjd_hi, mjd_lo = two_sum(int_day, frac_day)
    return mjd_hi + mjd_lo


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
