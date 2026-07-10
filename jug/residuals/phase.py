"""Shared pulse-phase machinery (mirror of pint/phase.py).

Canonical phase/residual computation used by both the PINT-family and
tempo2-compatible host paths and by the fitter.
"""
from __future__ import annotations
import math
import numpy as np
from jug.io.par_reader import get_longdouble
from jug.utils.constants import SECS_PER_DAY


def _fortran_mod(value, period):
    """Fractional part using tempo2's Fortran-style modulo (tempo2Util.C)."""
    x = np.asarray(value, dtype=np.longdouble)
    p = np.longdouble(period)
    return x - np.trunc(x / p) * p


def _fortran_nlong(value):
    """Nearest integer with ties away from zero (tempo2Util.C fortran_nlong).

    Rounds in longdouble: at |phase5| ~ 1e11 turns a float64 downcast can
    round to the wrong integer near half-turn boundaries.
    """
    x = np.asarray(value, dtype=np.longdouble)
    scalar = x.ndim == 0
    if scalar:
        x = x.reshape(1)
    half = np.longdouble(0.5)
    out = np.empty(len(x), dtype=np.int64)
    pos = x > 0.0
    out[pos] = np.trunc(x[pos] + half).astype(np.int64)
    out[~pos] = np.trunc(x[~pos] - half).astype(np.int64)
    return out[0] if scalar else out


def _spin_taylor_phase(dt_sec, f_coeffs) -> np.ndarray:
    """Spin Taylor phase at emission-time offsets (longdouble)."""
    dt = np.asarray(dt_sec, dtype=np.longdouble)
    n_coeffs = len(f_coeffs)
    if dt.ndim == 0:
        phase = np.longdouble(0.0)
        for i in range(n_coeffs - 1, -1, -1):
            phase = (phase + f_coeffs[i] / np.longdouble(math.factorial(i + 1))) * dt
        return phase
    phase = np.longdouble(0.0)
    for i in range(n_coeffs - 1, -1, -1):
        phase = (phase + f_coeffs[i] / np.longdouble(math.factorial(i + 1))) * dt
    return phase


def compute_phase_residuals(dt_sec_ld, params, weights, subtract_mean=True,
                            tzr_phase=None, tdb_sec_ld=None, jump_phase=None,
                            external_pulse_numbers=None,
                            track_val=None,
                            external_pn_add=None,
                            bbat_mjd=None,
                            torb_sec=None,
                            use_native_bbat_phase5: bool = False,
                            addsat_sec=None,
                            mean_mode: str = "weighted"):
    """Compute phase residuals from emission-time offsets (canonical implementation).

    This is the single shared function used by both the evaluate-only and fitter
    codepaths to guarantee identical phase computation, wrapping, and conversion.

    Parameters
    ----------
    dt_sec_ld : np.ndarray (longdouble)
        Time since PEPOCH minus all delays, in seconds.
        Must be longdouble to preserve phase precision for large |dt|.
    params : dict
        Timing model parameters (needs F0, F1, F2).
    weights : np.ndarray (float64)
        1/sigma^2 weights for weighted mean subtraction.
    subtract_mean : bool
        Whether to subtract weighted mean from residuals.
    mean_mode : {"weighted", "unweighted"}
        Mean convention used when ``subtract_mean`` is true. Tempo2 removes
        the unweighted prefit mean; the existing PINT-compatible path keeps
        JUG's historical weighted-mean convention.
    tzr_phase : float or longdouble, optional
        Phase at the TZR reference point. If provided, subtracted from each
        TOA's phase before wrapping to ensure correct pulse numbering.
    tdb_sec_ld : np.ndarray (longdouble), optional
        TDB times in seconds (longdouble). Required for glitch computation.
        If None, glitch contributions are not computed.
    external_pulse_numbers : np.ndarray (longdouble), optional
        Externally provided pulse numbers (from -pn flags in tim file).
        With ``track_val=-2``, these are offsets from obsn[0] and are used
        with tempo2's ``pnAdd`` / ``addPhase`` logic in ``formResiduals.C``.
        Without TRACK -2, -pn flags are ignored.
    track_val : int, optional
        Tempo2 TRACK parameter value. When -2, enables pulse-number tracking.
    external_pn_add : np.ndarray (int64), optional
        Per-TOA cumulative ``-pnadd`` flag values (tim order). Tempo2
        initialises ``pnAdd`` to -1 before accumulating ``-pnadd`` flags.
    bbat_mjd : np.ndarray (float64), optional
        Barycentric arrival MJDs for tempo2 spin phase (``formResiduals.C``).
    torb_sec : np.ndarray (float64), optional
        Binary delay (seconds) included in tempo2 ``deltaT`` / ``ftpd``.
    use_native_bbat_phase5 : bool
        When True (and TRACK −2 + ``-pn``), use quarantined tempo2 ``phase5`` at
        ``bbat`` with ``track_minus2_frac_phase``. See
        ``jug.residuals.tempo2.graph_config.USE_NATIVE_BBAT_PHASE5``.
    addsat_sec : np.ndarray (float64), optional
        Per-TOA integer-second ``-addsat`` shifts (already applied to site MJD at
        read). ``-addsat`` is applied to SAT at timfile read; no extra phase
        correction is applied here (see ``PARITY_ROADMAP.md``).

    Returns
    -------
    residuals_us : np.ndarray (float64)
        Residuals in microseconds.
    residuals_sec : np.ndarray (float64)
        Residuals in seconds.
    pulse_number : np.ndarray (longdouble)
        Integer pulse numbers used for phase wrapping.
    """
    F0 = get_longdouble(params, 'F0')
    PEPOCH = get_longdouble(params, 'PEPOCH')
    dt = np.asarray(dt_sec_ld, dtype=np.longdouble)

    has_track_minus2_pn = (
        track_val is not None
        and int(track_val) == -2
        and external_pulse_numbers is not None
    )
    use_tempo2_bbat_phase5 = (
        use_native_bbat_phase5
        and bbat_mjd is not None
        and torb_sec is not None
        and has_track_minus2_pn
    )

    if use_tempo2_bbat_phase5:
        from jug.residuals.tempo2_spin import compute_tempo2_phase5

        jump_arr = None
        if jump_phase is not None:
            jump_arr = np.asarray(jump_phase, dtype=np.float64)
        torb_for_phase5 = np.asarray(torb_sec, dtype=np.float64)
        phase = compute_tempo2_phase5(
            bbat_mjd,
            torb_for_phase5,
            params,
            jump_phase=jump_arr,
            tzr_phase=tzr_phase,
        )
    else:
        # Collect all spin frequency derivatives F0, F1, F2, ... FN
        f_coeffs = [F0]
        k = 1
        while f'F{k}' in params:
            f_coeffs.append(get_longdouble(params, f'F{k}', default=0.0))
            k += 1

        dt = np.asarray(dt_sec_ld, dtype=np.longdouble)

        # Phase via Taylor series: phase = sum(F_k * dt^(k+1) / (k+1)!)
        n_coeffs = len(f_coeffs)
        phase = np.longdouble(0.0)
        for i in range(n_coeffs - 1, -1, -1):
            phase = (phase + f_coeffs[i] / np.longdouble(math.factorial(i + 1))) * dt

        # Glitch contributions at emission time (PINT convention).
        glitch_idx = 1
        while f'GLEP_{glitch_idx}' in params:
            glep = get_longdouble(params, f'GLEP_{glitch_idx}')
            glph = get_longdouble(params, f'GLPH_{glitch_idx}', default=0.0)
            glf0 = get_longdouble(params, f'GLF0_{glitch_idx}', default=0.0)
            glf1 = get_longdouble(params, f'GLF1_{glitch_idx}', default=0.0)
            glf0d = get_longdouble(params, f'GLF0D_{glitch_idx}', default=0.0)
            gltd = get_longdouble(params, f'GLTD_{glitch_idx}', default=0.0)

            dt_glitch = dt
            glep_dt = (glep - PEPOCH) * np.longdouble(SECS_PER_DAY)
            active = dt_glitch > glep_dt
            dt_since_glep = np.where(active, dt_glitch - glep_dt, np.longdouble(0.0))

            glitch_phase = (glph
                           + glf0 * dt_since_glep
                           + np.longdouble(0.5) * glf1 * dt_since_glep**2)

            if gltd != 0.0 and glf0d != 0.0:
                gltd_sec = gltd * np.longdouble(SECS_PER_DAY)
                glitch_phase += glf0d * gltd_sec * (
                    np.longdouble(1.0) - np.exp(-dt_since_glep / gltd_sec)
                )

            phase += np.where(active, glitch_phase, np.longdouble(0.0))
            glitch_idx += 1

        if jump_phase is not None:
            phase = phase + np.asarray(jump_phase, dtype=np.longdouble)

        if tzr_phase is not None:
            phase = phase - np.longdouble(tzr_phase)

        # Keep longdouble through wrapping: the Taylor phase carries the full
        # integer pulse count (~1e11 turns on decade-long data at F0~300 Hz),
        # where a float64 downcast quantizes phase at ~2e-5 turns (~60 ns).
        # tempo2 keeps phase5 in longdouble throughout formResiduals.C.
        phase = np.asarray(phase, dtype=np.longdouble)

    # Phase wrapping (Tempo2 formResiduals.C for TRACK -2; sequential connection otherwise).
    # Native ``track_minus2_frac_phase`` (tempo2 pnNew) is used with ``phase5`` when
    # ``use_tempo2_bbat_phase5``; legacy ``-pn_add`` wrapping remains for Taylor spin.
    if has_track_minus2_pn and use_tempo2_bbat_phase5:
        from jug.residuals.tempo2_spin import track_minus2_frac_phase

        if external_pn_add is not None:
            pn_add_arr = np.asarray(external_pn_add, dtype=np.int64)
        else:
            pn_add_arr = np.full(len(phase), -1, dtype=np.int64)

        pn_tim = np.asarray(external_pulse_numbers, dtype=np.int64)
        frac_phase, pulse_number = track_minus2_frac_phase(
            np.asarray(phase, dtype=np.float64),
            np.asarray(bbat_mjd, dtype=np.float64),
            float(F0),
            pn_tim,
            pn_add_arr,
        )
        pulse_number = np.asarray(pulse_number, dtype=np.longdouble)
    elif has_track_minus2_pn:
        # Legacy TRACK -2 on emission-time Taylor phase (PINT / partial tempo2).
        phas1 = _fortran_mod(phase[0], np.longdouble(1.0))
        phase5 = np.asarray(phase, dtype=np.longdouble) - phas1
        nphase = np.asarray(_fortran_nlong(phase5), dtype=np.longdouble)

        if external_pn_add is not None:
            pn_add_arr = np.asarray(external_pn_add, dtype=np.int64)
        else:
            pn_add_arr = np.full(len(phase5), -1, dtype=np.int64)

        pn_tim = np.asarray(external_pulse_numbers, dtype=np.int64)
        pn0 = np.int64(nphase[0]) + pn_add_arr[0]
        pulse_number = np.asarray(pn0 + pn_tim, dtype=np.longdouble)

        add_phase = -pn_add_arr.astype(np.float64)
        frac_phase = phase5 - nphase + add_phase
    else:
        # Phase-connected wrapping: anchor at earliest emission time (dt order).
        sort_idx = np.argsort(dt)
        pulse_number = np.zeros(len(phase), dtype=np.longdouble)
        pulse_number[sort_idx[0]] = np.round(phase[sort_idx[0]])
        for k in range(1, len(sort_idx)):
            i = sort_idx[k]
            i_prev = sort_idx[k - 1]
            predicted_n = phase[i] - (phase[i_prev] - pulse_number[i_prev])
            pulse_number[i] = np.round(predicted_n)
        frac_phase = phase - pulse_number

    # -addsat is applied to sat at timfile read (readTimfile.C); native/tempo2
    # paths must not apply a second phase-domain correction here.
    if addsat_sec is not None and np.any(np.asarray(addsat_sec) != 0.0):
        pass

    # Convert to float64 seconds
    residuals_sec = np.asarray(frac_phase / F0, dtype=np.float64)

    if subtract_mean:
        if mean_mode == "unweighted":
            residuals_sec = residuals_sec - np.mean(residuals_sec)
        elif mean_mode == "weighted":
            wm = np.sum(residuals_sec * weights) / np.sum(weights)
            residuals_sec = residuals_sec - wm
        else:
            raise ValueError(f"Unknown residual mean mode {mean_mode!r}")

    residuals_us = residuals_sec * 1e6
    return residuals_us, residuals_sec, pulse_number
