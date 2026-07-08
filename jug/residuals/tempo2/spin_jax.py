"""JAX tempo2 spin: phase5 and TRACK -2 (formResiduals.C)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jug.residuals.tempo2.compensated import frac_of_int_times
from jug.utils.constants import SECS_PER_DAY


def fortran_mod_jax(value, period):
    x = jnp.asarray(value, dtype=jnp.float64)
    p = jnp.asarray(period, dtype=jnp.float64)
    return x - jnp.trunc(x / p) * p


def fortran_nlong_jax(value):
    x = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(
        x > 0.0,
        jnp.trunc(x + 0.5),
        jnp.trunc(x - 0.5),
    ).astype(jnp.int64)


def phase3_horner_jax(delta_t_sec, f_terms):
    """Taylor tail with factorial denominators k+1, matching formResiduals.C."""
    phase = jnp.zeros_like(delta_t_sec, dtype=jnp.float64)
    arg = delta_t_sec * delta_t_sec
    for idx in range(1, len(f_terms)):
        denom = float(math.factorial(idx + 1))
        phase = phase + f_terms[idx] * arg / denom
        arg = arg * delta_t_sec
    return phase


def pepoch_parts_from_value(pepoch_mjd):
    """Split PEPOCH into integer day and fractional day (formResiduals.C)."""
    pep = jnp.asarray(pepoch_mjd, dtype=jnp.float64)
    pep_int = jnp.floor(pep)
    pep_frac = pep - pep_int
    return pep, pep_int, pep_frac


def compute_tempo2_phase5_daysec(
    bbat_int,
    bbat_sec,
    torb_sec,
    f_terms,
    pep_int,
    pep_frac,
    jump_phase=None,
    tzr_phase=None,
):
    """Full phase5 in turns (before fractional reduction), formResiduals.C L507-L536."""
    f0 = f_terms[0]
    nf0 = jnp.trunc(f0)
    ff0 = f0 - nf0
    frac_day = bbat_sec / SECS_PER_DAY
    ntpd = bbat_int - pep_int
    fct = frac_day - pep_frac
    ftpd = fct + torb_sec / SECS_PER_DAY
    phase2 = (nf0 * ftpd + ntpd * ff0 + ftpd * ff0) * SECS_PER_DAY
    delta_t = ntpd * SECS_PER_DAY + fct * SECS_PER_DAY + torb_sec
    phase3 = phase3_horner_jax(delta_t, f_terms)
    phase5 = phase2 + phase3
    if jump_phase is not None:
        phase5 = phase5 + jump_phase
    if tzr_phase is not None:
        phase5 = phase5 - tzr_phase
    return phase5


def compute_tempo2_frac_phase_daysec(
    bbat_int,
    bbat_sec,
    torb_sec,
    f_terms,
    pep_int,
    pep_frac,
    jump_phase=None,
    tzr_phase=None,
):
    """Fractional-turn phase for non-TRACK residual formation."""
    f0 = f_terms[0]
    nf0 = jnp.trunc(f0)
    ff0 = f0 - nf0
    frac_day = bbat_sec / SECS_PER_DAY
    ntpd = bbat_int - pep_int
    fct = frac_day - pep_frac
    ftpd = fct + torb_sec / SECS_PER_DAY
    m_turns = nf0 * SECS_PER_DAY
    n_turns = ntpd * SECS_PER_DAY
    t1 = frac_of_int_times(m_turns, ftpd)
    t2 = frac_of_int_times(n_turns, ff0)
    t3 = ftpd * ff0 * SECS_PER_DAY
    delta_t = n_turns + fct * SECS_PER_DAY + torb_sec
    phase3 = phase3_horner_jax(delta_t, f_terms)
    phase3_frac = phase3 - jnp.round(phase3)
    total = t1 + t2 + t3 + phase3_frac
    if jump_phase is not None:
        total = total + jump_phase
    if tzr_phase is not None:
        total = total - tzr_phase
    return total - jnp.round(total)


def compute_tempo2_phase5_jax(
    bbat_mjd,
    torb_sec,
    f_terms,
    pepoch_mjd,
    jump_phase=None,
    tzr_phase=None,
):
    """Legacy single-MJD phase5; prefer :func:`compute_tempo2_phase5_daysec`."""
    from jug.residuals.tempo2.compensated import split_mjd_to_daysec

    bbat_int, bbat_sec = split_mjd_to_daysec(bbat_mjd)
    _, pep_int, pep_frac = pepoch_parts_from_value(pepoch_mjd)
    return compute_tempo2_phase5_daysec(
        bbat_int,
        bbat_sec,
        torb_sec,
        f_terms,
        pep_int,
        pep_frac,
        jump_phase=jump_phase,
        tzr_phase=tzr_phase,
    )


def track_minus2_frac_phase_jax(
    phase5,
    bbat_int,
    f0,
    pulse_numbers,
    pn_add,
):
    """JAX port of tempo2 pnNew wrapping, preserving obsn[0] anchoring."""
    nf0 = jnp.trunc(f0).astype(jnp.int64)
    phas1 = fortran_mod_jax(phase5[0], 1.0)
    p5 = phase5 - phas1
    nphase = fortran_nlong_jax(p5)
    pn_base = pulse_numbers[0]
    bbat_int = jnp.asarray(bbat_int, dtype=jnp.int64)
    bbat0 = bbat_int[0]

    def step(carry, x):
        pn0 = carry
        p5_i, nphase_i, pn_i, pn_add_i, bbat_i = x
        ntpd_i = bbat_i - bbat0
        phaseint = nf0 * ntpd_i * 86400
        pn_new_abs = phaseint + fortran_nlong_jax(jnp.asarray([p5_i]))[0]
        pn0_new = jnp.where(pn0 == -1, pn_new_abs, pn0)
        pn_new = jnp.where(pn0 == -1, 0, pn_new_abs - pn0)
        pn_act = (pn_i - pn_base) + pn_add_i
        add_phase = pn_new - pn_act
        frac = (p5_i - nphase_i.astype(jnp.float64)) + add_phase.astype(jnp.float64)
        pulse = pn_new_abs.astype(jnp.float64) - add_phase.astype(jnp.float64)
        return pn0_new, (frac, pulse)

    _, (frac, pulse) = jax.lax.scan(
        step,
        jnp.asarray(-1, dtype=jnp.int64),
        (p5, nphase, pulse_numbers, pn_add, bbat_int),
    )
    return frac, pulse


def spin_params_to_jax(params):
    """Collect F coefficients and PEPOCH parts as JAX float64."""
    from jug.io.par_reader import get_longdouble

    f_terms = [float(get_longdouble(params, "F0"))]
    k = 1
    while f"F{k}" in params:
        f_terms.append(float(get_longdouble(params, f"F{k}", default=0.0)))
        k += 1
    pep_ld = get_longdouble(params, "PEPOCH")
    pep = float(pep_ld)
    pep_int = float(int(pep_ld))
    pep_frac = pep - pep_int
    return (
        jnp.asarray(f_terms, dtype=jnp.float64),
        jnp.asarray(pep, dtype=jnp.float64),
        jnp.asarray(pep_int, dtype=jnp.float64),
        jnp.asarray(pep_frac, dtype=jnp.float64),
    )
