"""JAX tempo2 spin: phase5 and TRACK -2 (formResiduals.C)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

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


def compute_tempo2_phase5_jax(
    bbat_mjd,
    torb_sec,
    f_terms,
    pepoch_mjd,
    jump_phase=None,
    tzr_phase=None,
):
    """JAX phase2 + phase3 port of formResiduals.C L507-L536."""
    f0 = f_terms[0]
    nf0 = jnp.trunc(f0).astype(jnp.int64)
    ff0 = f0 - nf0.astype(jnp.float64)
    c_bbat = jnp.trunc(bbat_mjd)
    c_pep = jnp.trunc(pepoch_mjd)
    ntpd = c_bbat - c_pep
    fct = (bbat_mjd - c_bbat) - (pepoch_mjd - c_pep)
    ftpd = fct + torb_sec / SECS_PER_DAY
    phase2 = (nf0.astype(jnp.float64) * ftpd + ntpd * ff0 + ftpd * ff0) * SECS_PER_DAY
    delta_t = (bbat_mjd - pepoch_mjd) * SECS_PER_DAY + torb_sec
    phase3 = phase3_horner_jax(delta_t, f_terms)
    phase5 = phase2 + phase3
    if jump_phase is not None:
        phase5 = phase5 + jump_phase
    if tzr_phase is not None:
        phase5 = phase5 - tzr_phase
    return phase5


def track_minus2_frac_phase_jax(
    phase5,
    bbat_mjd,
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
    c_bbat = jnp.trunc(bbat_mjd).astype(jnp.int64)
    c_bbat0 = c_bbat[0]

    def step(carry, x):
        pn0 = carry
        p5_i, nphase_i, pn_i, pn_add_i, c_bbat_i = x
        ntpd_i = c_bbat_i - c_bbat0
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
        (p5, nphase, pulse_numbers, pn_add, c_bbat),
    )
    return frac, pulse


def spin_params_to_jax(params):
    """Collect F coefficients and PEPOCH as JAX float64."""
    from jug.io.par_reader import get_longdouble

    f_terms = [float(get_longdouble(params, "F0"))]
    k = 1
    while f"F{k}" in params:
        f_terms.append(float(get_longdouble(params, f"F{k}", default=0.0)))
        k += 1
    pepoch = float(get_longdouble(params, "PEPOCH"))
    return jnp.asarray(f_terms, dtype=jnp.float64), jnp.asarray(pepoch, dtype=jnp.float64)
