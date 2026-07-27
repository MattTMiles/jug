"""Tempo2-native spin phase and TRACK -2 helpers (formResiduals.C).

Quarantined experimental path: ``compute_tempo2_phase5`` and
``track_minus2_frac_phase`` are **not** on the production parity route.
Production uses emission-time Taylor spin + legacy TRACK −2.

See ``jug.residuals.tempo2.graph_config`` and
``PARITY_ROADMAP.md``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from jug.io.par_reader import get_longdouble
from jug.residuals.gauge import ReferenceGauge, apply_phase_gauge


def compute_emission_taylor_phase5_nphase(
    dt_sec,
    params,
    *,
    jump_phase=None,
    tzr_phase=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Emission-time Taylor ``phase5`` / ``nphase`` for TRACK -2 ``-addsat`` coupling.

    Tempo2 applies ``-addsat`` at read time (site MJD shift) but evaluates the
    TRACK -2 integer-turn correction against emission-time spin phase, not the
    ``phase5@bbat`` used for ``pnNew`` wrapping.
    """
    from jug.utils.constants import SECS_PER_DAY

    f0 = get_longdouble(params, "F0")
    pepoch = get_longdouble(params, "PEPOCH")
    dt = np.asarray(dt_sec, dtype=np.longdouble)
    f_coeffs = [f0]
    k = 1
    while f"F{k}" in params:
        f_coeffs.append(get_longdouble(params, f"F{k}", default=0.0))
        k += 1

    phase = np.longdouble(0.0)
    for i in range(len(f_coeffs) - 1, -1, -1):
        phase = (phase + f_coeffs[i] / np.longdouble(math.factorial(i + 1))) * dt

    glitch_idx = 1
    while f"GLEP_{glitch_idx}" in params:
        glep = get_longdouble(params, f"GLEP_{glitch_idx}")
        glph = get_longdouble(params, f"GLPH_{glitch_idx}", default=0.0)
        glf0 = get_longdouble(params, f"GLF0_{glitch_idx}", default=0.0)
        glf1 = get_longdouble(params, f"GLF1_{glitch_idx}", default=0.0)
        glf0d = get_longdouble(params, f"GLF0D_{glitch_idx}", default=0.0)
        gltd = get_longdouble(params, f"GLTD_{glitch_idx}", default=0.0)
        glep_dt = (glep - pepoch) * np.longdouble(SECS_PER_DAY)
        active = dt > glep_dt
        dt_since = np.where(active, dt - glep_dt, np.longdouble(0.0))
        glitch_phase = glph + glf0 * dt_since + np.longdouble(0.5) * glf1 * dt_since**2
        if gltd != 0.0 and glf0d != 0.0:
            gltd_sec = gltd * np.longdouble(SECS_PER_DAY)
            glitch_phase = glitch_phase + glf0d * gltd_sec * (
                np.longdouble(1.0) - np.exp(-dt_since / gltd_sec)
            )
        phase = phase + np.where(active, glitch_phase, np.longdouble(0.0))
        glitch_idx += 1

    if jump_phase is not None:
        phase = phase + np.asarray(jump_phase, dtype=np.longdouble)
    if tzr_phase is not None:
        phase = phase - np.longdouble(tzr_phase)

    phase_f64 = np.asarray(phase, dtype=np.float64)
    phas1 = float(_fortran_mod(phase_f64[0], 1.0))
    phase5 = phase_f64 - phas1
    nphase = _fortran_nlong(phase5).astype(np.float64)
    return phase5, nphase


def compute_tempo2_bbat_mjd(
    model_mjd: np.ndarray,
    prebinary_delay_sec: np.ndarray,
) -> np.ndarray:
    """Tempo2 ``obsn.bbat`` from JUG delay geometry (matches libstempo/pytempo).

    ``bbat = model_mjd − prebinary_delay_sec / 86400``.  This is **not** the formBats
    diagnostic ``term_diagnostics['bbat_mjd']`` (~65 s wrong on wsrt167).
    """
    from jug.utils.constants import SECS_PER_DAY

    model = np.asarray(model_mjd, dtype=np.float64)
    prebin = np.asarray(prebinary_delay_sec, dtype=np.float64)
    return model - prebin / np.float64(SECS_PER_DAY)


def _fortran_mod(value, period):
    x = np.asarray(value, dtype=np.float64)
    p = np.float64(period)
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


def compute_tempo2_torb_sec(
    bbat_mjd: np.ndarray,
    dt_sec: np.ndarray,
    pepoch,
) -> np.ndarray:
    """Tempo2 ``obsn[i].torb`` for ``phase5`` (formResiduals.C).

    Matches ``deltaT = (bbat - PEPOCH)*86400 + torb`` with emission-time ``dt_sec``:
    ``torb = dt - (bbat - PEPOCH)*86400``.  This is **not** JUG ``total - prebinary``.
    """
    from jug.utils.constants import SECS_PER_DAY

    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    dt = np.asarray(dt_sec, dtype=np.float64)
    pep = np.float64(pepoch)
    return dt - (bbat - pep) * np.float64(SECS_PER_DAY)


def spin_delta_sec_at_bbat(
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    pepoch: float,
) -> np.ndarray:
    """Tempo2 spin argument ``deltaT = (bbat - PEPOCH)*86400 + torb`` (formResiduals.C)."""
    from jug.utils.constants import SECS_PER_DAY

    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    torb = np.asarray(torb_sec, dtype=np.float64)
    return (bbat - np.float64(pepoch)) * np.float64(SECS_PER_DAY) + torb


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
        return np.zeros_like(dt, dtype=np.float64)
    phase3 = np.zeros_like(dt, dtype=np.float64)
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
        gphase = glph + glf0 * dt_since + 0.5 * glf1 * dt_since**2
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

    .. deprecated::
        Dev/oracle wrapper only. Production tempo2 spin uses emission-time Taylor;
        JAX counterpart is ``compute_tempo2_phase5_daysec`` in ``jug.residuals.tempo2.spin_jax``.
    """
    import warnings

    warnings.warn(
        "compute_tempo2_phase5 is deprecated; use jug.residuals.tempo2.spin_jax for JAX production.",
        DeprecationWarning,
        stacklevel=2,
    )
    from jug.utils.constants import SECS_PER_DAY

    f_coeffs = _collect_f_coeffs(params)
    f0 = f_coeffs[0]
    nf0 = int(f0)
    ff0 = np.float64(f0 - nf0)
    pepoch = float(get_longdouble(params, "PEPOCH"))

    bbat = np.asarray(bbat_mjd, dtype=np.float64)
    torb = np.asarray(torb_sec, dtype=np.float64)
    c_bbat = np.trunc(bbat)
    c_pep = np.trunc(np.full_like(bbat, pepoch, dtype=np.float64))

    ntpd = c_bbat - c_pep
    fct = (bbat - c_bbat) - (pepoch - c_pep)
    ftpd = fct + torb / np.float64(SECS_PER_DAY)
    phase2 = (np.float64(nf0) * ftpd + ntpd * ff0 + ftpd * ff0) * np.float64(SECS_PER_DAY)

    delta_t = (bbat - pepoch) * np.float64(SECS_PER_DAY) + torb
    phase3 = _tempo2_phase3_vectorized(delta_t, f_coeffs)

    phase5 = phase2 + phase3 + _glitch_phase_bbat(bbat, params)

    if jump_phase is not None:
        phase5 = phase5 + np.asarray(jump_phase, dtype=np.float64)
    if tzr_phase is not None:
        phase5 = phase5 - np.float64(tzr_phase)

    return np.asarray(phase5, dtype=np.float64)


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
    # IPTA tim files store ``-pn`` as an absolute-looking offset whose *delta from
    # obsn[0]* equals tempo2 ``pnNew`` (after ``pn0`` anchoring).  Using raw ``-pn``
    # in ``pnAct`` blows up ``addPhase`` on wsrt167 (~10¹⁰ turns); see
    # ``PARITY_ROADMAP.md`` § "Phase D — TRACK −2 pnNew".
    pn_tim_base = int(pn_tim[0])

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
        pn_act = int(pn_tim[i]) - pn_tim_base + int(pn_add_arr[i])
        add_phase = pn_new - pn_act
        frac[i] = (p5[i] - float(nphase[i])) + add_phase
        ntrk = add_phase
        pulse_number[i] = float(int(phaseint + _fortran_nlong(np.array([p5[i]]))[0]) - ntrk)

    return frac, pulse_number


def form_residuals_tempo2_numpy(
    *,
    bbat_mjd: np.ndarray,
    torb_sec: np.ndarray,
    params,
    jump_phase: np.ndarray | None,
    tzr_phase=None,
    track_val: int,
    pn_tim: np.ndarray | None,
    pn_add: np.ndarray | None,
    addsat_sec: np.ndarray | None = None,
    emission_phase5: np.ndarray | None = None,
    subtract_mean: bool = True,
    mean_mode: str = "unweighted",
    first_in_range_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NumPy tempo2 ``formResiduals.C`` semantics for strict probes."""
    del first_in_range_idx  # START/FINISH masks not wired in fixture path yet
    f0 = float(get_longdouble(params, "F0"))
    if emission_phase5 is not None:
        phase5_full = np.asarray(emission_phase5, dtype=np.float64)
        if jump_phase is not None:
            phase5_full = phase5_full + np.asarray(jump_phase, dtype=np.float64)
        if tzr_phase is not None:
            phase5_full = phase5_full - np.float64(tzr_phase)
    else:
        jump_arr = None if jump_phase is None else np.asarray(jump_phase, dtype=np.float64)
        phase5_full = compute_tempo2_phase5(
            bbat_mjd,
            torb_sec,
            params,
            jump_phase=jump_arr,
            tzr_phase=tzr_phase,
        )

    if int(track_val) == -2 and pn_tim is not None and pn_add is not None:
        frac_turns, pulse_number = track_minus2_frac_phase(
            phase5_full,
            bbat_mjd,
            f0,
            np.asarray(pn_tim, dtype=np.int64),
            np.asarray(pn_add, dtype=np.int64),
        )
        if addsat_sec is not None and np.any(np.asarray(addsat_sec) != 0.0):
            pass  # -addsat applied to sat at timfile read; no phase-domain fudge
        nphase = _fortran_nlong(phase5_full - float(_fortran_mod(phase5_full[0], 1.0))).astype(
            np.float64
        )
        residual_turns = frac_turns
    else:
        phas1 = float(_fortran_mod(phase5_full[0], 1.0))
        p5 = phase5_full - phas1
        nphase = _fortran_nlong(p5).astype(np.float64)
        pulse_number = np.zeros_like(p5)
        residual_turns = p5 - nphase
        if addsat_sec is not None and np.any(np.asarray(addsat_sec) != 0.0):
            pass  # -addsat applied to sat at timfile read; no phase-domain fudge

    residual_sec = residual_turns / f0

    if not subtract_mean or mean_mode == "none":
        gauge = ReferenceGauge(mode="none")
    elif mean_mode != "unweighted":
        # This host path has no TOA weights; only unweighted mean is supported.
        raise ValueError(
            "tempo2_spin residual path supports only mean_mode='unweighted' "
            f"(got {mean_mode!r}); weighted gauges belong on paths that carry "
            "TOA weights"
        )
    else:
        gauge = ReferenceGauge(mode="mean", weights=None)
    residual_sec = np.asarray(apply_phase_gauge(residual_sec, gauge), dtype=np.float64)
    return residual_sec, np.asarray(pulse_number, dtype=np.float64), nphase
