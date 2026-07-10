"""Longdouble orbit-count reduction for binary time arguments.

Binary phase is computed inside float64 JAX kernels as
``2*pi*(tt/PB_sec)``. With |tt| up to ~1e8 s and PB_sec ~1e4-1e6 s the
orbit count reaches 1e3-1e4, so float64 rounding of the division (and of
the float64 cast of ``tt`` itself) leaves an absolute phase error of
~1e-11 rad — a ~ps Roemer-delay floor (A1 * dPhi) with a deterministic,
non-Gaussian sawtooth structure (verified against PINT/longdouble on
J1738+0333: 0.39 ps std, correlation +1.0000 with the predicted float64
phase error field).

Integer orbit multiples drop out of every periodic (trig) term, so the
fix is to subtract ``n_orb = round(tt/P)`` whole periods from ``tt`` in
LONGDOUBLE, outside JIT, and feed the reduced time to the kernel's
LINEAR phase term. After reduction |tt_red| <= ~P/2 (+ the prebinary
shift subtracted inside the kernel), where float64 keeps the phase to
~1e-16 rad.

IMPORTANT: only the linear phase term may use the reduced time. Secular
and higher-order terms (PBDOT/FB1+ quadratics, XDOT/EDOT/OMDOT/EPSxDOT
evolution, DDK K96 terms, nhat) must keep the FULL time — their float64
error on the full time is negligible (they are small corrections), and
they are NOT periodic in the orbit count.
"""
import numpy as np

__all__ = ["reduce_binary_time_sec"]


def reduce_binary_time_sec(tt_sec_ld, pb_days=None, fb0_hz=None):
    """Reduce binary time by a whole number of orbital periods (longdouble).

    Parameters
    ----------
    tt_sec_ld : array_like (longdouble preferred)
        (t - binary_epoch) in seconds, computed in longdouble by the caller.
    pb_days : float, optional
        Orbital period in days. Used when fb0_hz is not given. Longdouble is
        preferred: subtracting whole precise periods preserves accumulated
        phase from PB refinements smaller than one float64 ULP. The downstream
        kernel may divide the reduced remainder by float64 PB; that loses only
        the non-accumulating sub-ULP correction within half an orbit.
    fb0_hz : float, optional
        FB0 orbital frequency in Hz. Takes precedence over pb_days: the FB
        linear phase term is FB0*tt, so the reduction period must be 1/FB0
        evaluated from the SAME float64 FB0 the kernel uses, making
        FB0*tt_red == FB0*tt - n_orb exact in longdouble.

    Returns
    -------
    tt_red_f64 : np.ndarray (float64)
        tt - round(tt/P)*P computed in longdouble, cast to float64.
        If neither pb_days nor fb0_hz is usable, returns tt cast to
        float64 unchanged (reduction is then a no-op fallback).
    """
    tt = np.asarray(tt_sec_ld, dtype=np.longdouble)
    if fb0_hz is not None and fb0_hz != 0.0:
        period_sec = np.longdouble(1.0) / np.longdouble(fb0_hz)
    elif pb_days is not None and pb_days != 0.0:
        period_sec = np.longdouble(pb_days) * np.longdouble(86400.0)
    else:
        return np.asarray(tt, dtype=np.float64)
    n_orb = np.round(tt / period_sec)
    return np.asarray(tt - n_orb * period_sec, dtype=np.float64)
