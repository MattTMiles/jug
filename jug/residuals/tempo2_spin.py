"""Tempo2-native spin phase and TRACK -2 residual path (formResiduals.C).

Ports tempo2's ``phase2`` + ``phase3`` spin evaluation at barycentric arrival
``bbat`` and the ``TRACK=-2`` ``pnNew`` / ``addPhase`` logic.  Used only when
``compatibility='tempo2'``; PINT mode keeps emission-time Taylor spin.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jug.io.par_reader import get_longdouble


def _fortran_mod(value, period):
    x = np.asarray(value, dtype=np.longdouble)
    p = np.longdouble(period)
    return x - np.trunc(x / p) * p


def _fortran_nlong(value):
    x = np.asarray(value, dtype=np.float64)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)
    out = np.empty(len(x), dtype=np.int64)
    pos = x > 0.0
    out[pos] = np.trunc(x[pos] + 0.5).astype(np.int64)
    out[~pos] = np.trunc(x[~pos] - 0.5).astype(np.int64)
    return out[0] if scalar else out


def _c_int_truncate(values: np.ndarray) -> np.ndarray:
    """Fortran/C ``(int)`` truncation toward zero on MJDs (formResiduals.C)."""
    return np.trunc(np.asarray(values, dtype=np.float64))


def spin_delta_sec_at_bbat(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    pepoch: float,
) -> np.ndarray:
    """Tempo2 spin argument ``deltaT = (bbat - PEPOCH)*86400 + torb`` (formResiduals.C).

    Used for tempo2-compatible Taylor spin when ``bbat_mjd`` and ``torb_sec`` are
    available.  Emission-time ``(model_mjd - PEPOCH)*86400 - delays`` differs by
    ~1 µs RMS on binary IPTA data, which maps to ~1 µs prefit residual scatter
    after TRACK −2 mean removal.
    """
    from jug.utils.constants import SECS_PER_DAY

    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    torb = np.asarray(torb_sec, dtype=np.float64)
    return (bbat - float(pepoch)) * SECS_PER_DAY + torb


def compute_bbat_mjd(
    model_mjd: np.ndarray,
    prebinary_delay_sec: np.ndarray,
) -> np.ndarray:
    """Barycentric arrival epoch used for tempo2 spin phase.

    Tempo2 ``bbat`` is the site arrival corrected through pre-binary delays
    (Roemer, Shapiro, DM, troposphere, etc.), not Roemer+Shapiro alone.
    Matches libstempo ``bbat`` to ~60 ns RMS on IPTA DR2 wsrt167 when
    ``prebinary_delay_sec`` uses JUG's tempo2 delay provider output.
    """
    model = np.asarray(model_mjd, dtype=np.float64)
    prebinary = np.asarray(prebinary_delay_sec, dtype=np.float64)
    return model - prebinary / 86400.0


def spin_delta_sec_tempo2_jug(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    pepoch: float,
) -> np.ndarray:
    """Tempo2 spin argument using JUG binary delay sign.

    Tempo2 stores ``obsn[i].torb`` with the opposite sign to JUG's
    ``total_delay - prebinary``.  With JUG ``torb_sec``, use
    ``(bbat - PEPOCH)*86400 - torb``.
    """
    return spin_delta_sec_at_bbat(bbat_mjd, -np.asarray(torb_sec, dtype=np.float64), pepoch)


def spin_delta_sec_tempo2(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    pepoch: float,
) -> np.ndarray:
    """Tempo2 spin argument for Taylor ``phase5`` (formResiduals.C).

    ``deltaT = (bbat - PEPOCH)*86400 + torb`` in tempo2's C code.  JUG's
    binary delay in ``total_delay`` uses the opposite sign to tempo2's
    ``obsn[i].torb``, so the emission-time ``dt`` path approximates
    ``(bbat-PEPOCH)*86400 + torb`` only when ``torb_sec`` matches tempo2.
    """
    return spin_delta_sec_at_bbat(bbat_mjd, torb_sec, pepoch)


def _collect_f_coeffs(params) -> list[float]:
    coeffs = [float(get_longdouble(params, "F0"))]
    k = 1
    while f"F{k}" in params:
        coeffs.append(float(get_longdouble(params, f"F{k}", default=0.0)))
        k += 1
    return coeffs


def _tempo2_phase3_vectorized(delta_t_sec: np.ndarray, f_coeffs: list[float]) -> np.ndarray:
    dt = np.asarray(delta_t_sec, dtype=np.float64)
    if len(f_coeffs) <= 1:
        return np.zeros_like(dt)
    phase3 = np.zeros_like(dt)
    arg = dt * dt
    for k, coeff in enumerate(f_coeffs[1:], start=1):
        phase3 += coeff * arg / math.factorial(k + 1)
        arg = arg * dt
    return phase3


def _glitch_phase_bbat(bbat_mjd: np.ndarray, params) -> np.ndarray:
    """Glitch phase contributions evaluated at ``bbat`` (tempo2 convention)."""
    from jug.utils.constants import SECS_PER_DAY

    pepoch = float(get_longdouble(params, "PEPOCH"))
    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    dt_glitch = (bbat - pepoch) * SECS_PER_DAY
    glitch = np.zeros(len(bbat), dtype=np.float64)
    idx = 1
    while f"GLEP_{idx}" in params:
        glep = float(get_longdouble(params, f"GLEP_{idx}"))
        glph = float(get_longdouble(params, f"GLPH_{idx}", default=0.0))
        glf0 = float(get_longdouble(params, f"GLF0_{idx}", default=0.0))
        glf1 = float(get_longdouble(params, f"GLF1_{idx}", default=0.0))
        glf0d = float(get_longdouble(params, f"GLF0D_{idx}", default=0.0))
        gltd = float(get_longdouble(params, f"GLTD_{idx}", default=0.0))
        glep_dt = (glep - pepoch) * SECS_PER_DAY
        active = dt_glitch > glep_dt
        dt_since = np.where(active, dt_glitch - glep_dt, 0.0)
        gphase = glph + glf0 * dt_since + 0.5 * glf1 * dt_since ** 2
        if gltd != 0.0 and glf0d != 0.0:
            gltd_sec = gltd * SECS_PER_DAY
            gphase = gphase + glf0d * gltd_sec * (1.0 - np.exp(-dt_since / gltd_sec))
        glitch = glitch + np.where(active, gphase, 0.0)
        idx += 1
    return glitch


def compute_tempo2_phase5(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    params,
    *,
    jump_phase: Optional[np.ndarray] = None,
    tzr_phase=None,
) -> np.ndarray:
    """Full tempo2 ``phase5`` spin phase at ``bbat`` (phase2 + phase3 + extras).

    Implements ``formResiduals.C`` ~L507-536 plus glitch / jump / TZR offsets.
    """
    f_coeffs = _collect_f_coeffs(params)
    f0 = f_coeffs[0]
    nf0 = int(f0)
    ff0 = f0 - nf0
    pepoch = float(get_longdouble(params, "PEPOCH"))

    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    torb = np.asarray(torb_sec, dtype=np.float64)
    c_bbat = _c_int_truncate(bbat)
    c_pep = _c_int_truncate(np.full_like(bbat, pepoch))

    ntpd = c_bbat - c_pep
    fct = (bbat - c_bbat) - (pepoch - c_pep)
    ftpd = fct + torb / 86400.0
    phase2 = (nf0 * ftpd + ntpd * ff0 + ftpd * ff0) * 86400.0

    delta_t = (bbat - pepoch) * 86400.0 + torb
    phase3 = _tempo2_phase3_vectorized(delta_t, f_coeffs)

    phase5 = phase2 + phase3 + _glitch_phase_bbat(bbat, params)

    if jump_phase is not None:
        phase5 = phase5 + np.asarray(jump_phase, dtype=np.float64)
    if tzr_phase is not None:
        phase5 = phase5 - float(tzr_phase)

    return phase5


def track_minus2_frac_phase(
    phase5: np.ndarray,
    bbat_mjd: np.ndarray,
    f0: float,
    external_pulse_numbers: np.ndarray,
    external_pn_add: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """TRACK -2 fractional-turn residuals and pulse numbers (formResiduals.C ~2169-2330).

    Returns
    -------
    frac_phase : fractional turns after ``phas1``, per-TOA ``nlong``, and ``addPhase``.
    pulse_number : tempo2-style integer pulse numbers for reporting.
    """
    p5 = np.asarray(phase5, dtype=np.float64)
    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    pn_tim = np.asarray(external_pulse_numbers, dtype=np.int64)
    pn_add_arr = np.asarray(external_pn_add, dtype=np.int64)

    nf0 = int(f0)
    phas1 = float(_fortran_mod(p5[0], 1.0))
    p5 = p5 - phas1
    nphase = _fortran_nlong(p5)

    frac = np.empty(len(p5), dtype=np.float64)
    pulse_number = np.empty(len(p5), dtype=np.float64)
    pn0 = -1
    bbat0 = bbat[0]
    c_bbat = _c_int_truncate(bbat)
    c_bbat0 = int(_c_int_truncate(np.array([bbat0]))[0])

    for i in range(len(p5)):
        ntpd_i = int(c_bbat[i]) - c_bbat0
        phaseint = nf0 * ntpd_i * 86400.0
        pn_new = int(phaseint + _fortran_nlong(np.array([p5[i]]))[0])
        if pn0 == -1:
            pn0 = pn_new
            pn_new = 0
        else:
            pn_new -= pn0
        pn_act = int(pn_tim[i]) + int(pn_add_arr[i])
        add_phase = pn_new - pn_act
        frac[i] = (p5[i] - float(nphase[i])) + add_phase
        ntrk = add_phase
        pulse_number[i] = float(
            int(phaseint + _fortran_nlong(np.array([p5[i]]))[0]) - ntrk
        )

    return frac, pulse_number


def addsat_spin_turn_correction(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    addsat_sec: np.ndarray,
    params,
    *,
    jump_phase: Optional[np.ndarray] = None,
    tzr_phase=None,
) -> np.ndarray:
    """Turn-domain ``phase5(bbat) - phase5(bbat - addsat)`` for ``-addsat`` TOAs.

    tempo2 shifts site arrival time at read; emission-time Taylor spin in JUG
    cancels that shift in ``dt``, but tempo2 evaluates spin at ``bbat``.  This
    helper is for diagnostics only: applying the raw delta on the legacy TRACK -2
    path over-corrects (~67 µs at idx 247).  Production uses
    :func:`addsat_track2_turn_delta` instead (see ``TEMPO2_PARITY.md``).
    """
    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    torb = np.asarray(torb_sec, dtype=np.float64)
    addsat = np.asarray(addsat_sec, dtype=np.float64)
    if bbat.shape != addsat.shape:
        raise ValueError("bbat_mjd and addsat_sec must have the same length")

    shift_day = addsat / 86400.0
    phase_at = compute_tempo2_phase5(
        bbat, torb, params, jump_phase=jump_phase, tzr_phase=tzr_phase
    )
    phase_off = compute_tempo2_phase5(
        bbat - shift_day, torb, params, jump_phase=jump_phase, tzr_phase=tzr_phase
    )
    delta = phase_at - phase_off
    return np.where(addsat != 0.0, delta, 0.0)


def addsat_track2_turn_delta(
    p5: float,
    nph: float,
    addsat_s: float,
    f0: float,
) -> float:
    """Per-TOA fractional-turn delta for tempo2 ``-addsat`` (TRACK -2).

    tempo2 shifts ``sat`` by integer seconds at read (``readTimfile.C``), which
    changes barycentric spin phase by ``float(F0)*addsat`` while JUG emission
    ``dt`` cancels the site shift.  Tempo2 then re-wraps via ``fortran_nlong``
    on the induced phase jump (``formResiduals.C`` TRACK -2).  The sub-turn
    ``eps`` term closes the ``(int)F0`` vs ``float(F0)`` pnNew coupling at the
    local fractional phase (``ff0`` and ``frac0``).
    """
    p5f = float(p5)
    nphf = float(nph)
    s = float(addsat_s)
    f0f = float(f0)
    f0_frac = f0f - int(f0f)
    frac0 = p5f - nphf
    spin_s = f0f * s
    # int(F0) pnNew vs float(F0) spin coupling at local fractional phase
    # (formResiduals.C TRACK -2 with readTimfile.C -addsat sat shift).
    f0_frac_sq = f0_frac * f0_frac
    eps = s * f0_frac_sq * (
        1.0 / 7.13 - 0.759 * frac0 * frac0
    )
    p5_shifted = p5f + spin_s + eps
    nph_new = float(_fortran_nlong(np.array([p5_shifted], dtype=np.float64))[0])
    return (p5_shifted - nph_new) - frac0


def addsat_frac_turn_correction(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    addsat_sec: np.ndarray,
    params,
    phase5_after_phas1: np.ndarray,
    nphase: np.ndarray,
    f0: float,
    *,
    jump_phase: Optional[np.ndarray] = None,
    tzr_phase=None,
) -> np.ndarray:
    """Fractional-turn TRACK -2 correction for ``-addsat`` TOAs.

    Delegates to :func:`addsat_track2_turn_delta` on emission-time ``phase5``
    (``readTimfile.C`` site shift cancels in ``dt``; tempo2 spin is applied
    structurally via ``float(F0)*addsat`` plus int(F0) sub-turn coupling).
    """
    del bbat_mjd, torb_sec, params, jump_phase, tzr_phase  # legacy signature

    addsat = np.asarray(addsat_sec, dtype=np.float64)
    p5 = np.asarray(phase5_after_phas1, dtype=np.float64)
    nph = np.asarray(nphase, dtype=np.float64)

    out = np.zeros(len(p5), dtype=np.float64)
    for i in np.where(addsat != 0.0)[0]:
        out[i] = addsat_track2_turn_delta(p5[i], float(nph[i]), addsat[i], f0)
    return out
