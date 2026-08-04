"""Legacy TEMPO terrestrial-to-celestial site transform (``T2CMETHOD TEMPO``).

Ports the ``t2cMethod == T2C_TEMPO`` branch of tempo2 ``get_obsCoord.C``
(lines 383-441), plus its helpers ``lmst``, ``get_precessionMatrix`` and the
``ut1red`` UT1 table interpolation from ``tai2ut1.C``.

Tempo2 enables this path for tempo1-emulation pulsars (``EPHVER < 5``) and
whenever the par file sets ``T2CMETHOD TEMPO`` — which covers all IPTA DR2
TDB par files.  The astropy GCRS transform differs from this legacy frame by
~10 m at the site, i.e. ~15 ns of Roemer delay.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from jug.utils.constants import C_KM_S

SECDAY = 86400.0

# tempo2.h: OBLQ used by get_precessionMatrix nutation rotation (degrees)
TEMPO_OBLQ_DEG = 23.4458333333333333


def _fortran_mod(a: np.ndarray, p: float) -> np.ndarray:
    """Fortran MOD: sign follows the dividend (C fmod semantics)."""
    return np.fmod(a, p)


def lmst(mjd: np.ndarray, olong_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Local mean sidereal time in turns + derivative (``get_obsCoord.C`` lmst).

    ``mjd`` should be UT1-corrected (``sat + correctionUT1/SECDAY``).
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    a = 24110.54841
    b = 8640184.812866
    c = 0.093104
    d = -6.2e-6

    nmjdu1 = mjd.astype(np.int64)
    fmjdu1 = mjd - nmjdu1

    tu0 = ((nmjdu1 - 51545).astype(np.float64) + 0.5) / 3.6525e4
    dtu = fmjdu1 / 3.6525e4
    tu = tu0 + dtu
    gmst0 = (a + tu0 * (b + tu0 * (c + tu0 * d))) / 86400.0
    seconds_per_jc = 86400.0 * 36525.0

    bprime = 1.0 + b / seconds_per_jc
    cprime = 2.0 * c / seconds_per_jc
    dprime = 3.0 * d / seconds_per_jc
    sdd = bprime + tu * (cprime + tu * dprime)

    gst = gmst0 + dtu * (
        seconds_per_jc + b + c * (tu + tu0) + d * (tu * tu + tu * tu0 + tu0 * tu0)
    ) / 86400.0
    xlst = gst - olong_deg / 360.0
    xlst = _fortran_mod(xlst, 1.0)
    xlst = np.where(xlst < 0.0, xlst + 1.0, xlst)
    return xlst, sdd


def get_precession_nutation_matrix(
    mjd: np.ndarray,
    delp: np.ndarray,
    dele: np.ndarray,
) -> np.ndarray:
    """IAU1976 precession x small-angle nutation matrix (``get_precessionMatrix``).

    Returns ``prn`` with shape ``(n, 3, 3)`` such that
    ``celestial = prn @ terrestrial_rotated`` per the C code's index order.
    """
    mjd = np.asarray(mjd, dtype=np.float64)
    n = len(mjd)
    par_zeta = (2306.2181, 0.30188, 0.017998)
    par_z = (2306.2181, 1.09468, 0.018203)
    par_theta = (2004.3109, -0.42665, -0.041833)
    seconds_per_rad = 3600.0 * 180.0 / np.pi

    t = (mjd - 51544.5) / 36525.0
    zeta = t * (par_zeta[0] + t * (par_zeta[1] + t * par_zeta[2])) / seconds_per_rad
    z = t * (par_z[0] + t * (par_z[1] + t * par_z[2])) / seconds_per_rad
    theta = t * (par_theta[0] + t * (par_theta[1] + t * par_theta[2])) / seconds_per_rad

    czeta, szeta = np.cos(zeta), np.sin(zeta)
    cz, sz = np.cos(z), np.sin(z)
    ctheta, stheta = np.cos(theta), np.sin(theta)

    prc = np.zeros((n, 3, 3), dtype=np.float64)
    prc[:, 0, 0] = czeta * ctheta * cz - szeta * sz
    prc[:, 1, 0] = czeta * ctheta * sz + szeta * cz
    prc[:, 2, 0] = czeta * stheta
    prc[:, 0, 1] = -szeta * ctheta * cz - czeta * sz
    prc[:, 1, 1] = -szeta * ctheta * sz + czeta * cz
    prc[:, 2, 1] = -szeta * stheta
    prc[:, 0, 2] = -stheta * cz
    prc[:, 1, 2] = -stheta * sz
    prc[:, 2, 2] = ctheta

    eps = TEMPO_OBLQ_DEG * np.pi / 180.0
    ceps, seps = np.cos(eps), np.sin(eps)
    delp = np.asarray(delp, dtype=np.float64)
    dele = np.asarray(dele, dtype=np.float64)

    nut = np.zeros((n, 3, 3), dtype=np.float64)
    nut[:, 0, 0] = 1.0
    nut[:, 0, 1] = -delp * ceps
    nut[:, 0, 2] = -delp * seps
    nut[:, 1, 0] = delp * ceps
    nut[:, 1, 1] = 1.0
    nut[:, 1, 2] = -dele
    nut[:, 2, 0] = delp * seps
    nut[:, 2, 1] = dele
    nut[:, 2, 2] = 1.0

    # PRCNUT.f: prn[j][i] = sum_k nut[i][k] * prc[k][j]
    prn = np.einsum("nik,nkj->nji", nut, prc)
    return prn


@lru_cache(maxsize=2)
def _load_ut1_table(clock_dir: str) -> tuple[float, float, np.ndarray]:
    """Parse tempo2 ``ut1.dat``: 5-day TAI-UT1 table in 1e-4 s units.

    Returns ``(first_line_mjd, last_line_mjd, entries)``; entry ``j``
    corresponds to MJD ``first_line_mjd + 5*j``.
    """
    path = Path(clock_dir) / "ut1.dat"
    entries: list[int] = []
    mjd0 = None
    mjd_last = None
    with open(path) as f:
        lines = f.readlines()
    stop = False
    for line in lines[2:]:
        # tempo2 chops the line at char 57 to drop the item-count column
        parts = line[:57].split()
        if len(parts) < 2:
            break
        if mjd0 is None:
            mjd0 = int(parts[0])
        mjd_last = int(parts[0])
        for tok in parts[1:7]:
            val = int(tok)
            if val == 0:
                stop = True
                break
            entries.append(val)
        if stop:
            break
    if mjd0 is None or mjd_last is None or not entries:
        raise ValueError(f"Could not parse UT1 table {path}")
    return float(mjd0), float(mjd_last), np.asarray(entries, dtype=np.float64)


def ut1red_sec(mjd: np.ndarray, clock_dir: str | None = None) -> np.ndarray:
    """UT1-TAI in seconds via tempo2 ``ut1red`` second-difference interpolation."""
    if clock_dir is None:
        from jug.io.clock import resolve_clock_dir

        clock_dir = str(resolve_clock_dir(compatibility="pint"))
    mjd0, mjd_last, entries = _load_ut1_table(str(clock_dir))
    mjd = np.asarray(mjd, dtype=np.float64)
    iint = 5.0
    units = 1.0e-4
    count2 = len(entries)
    mjd1 = mjd0 + iint
    mjd2 = mjd_last - iint

    t_all = (mjd - mjd1) / iint
    it = np.floor(t_all).astype(np.int64)
    t = t_all - it
    s = 1.0 - t

    in_range = (mjd > mjd1) & (mjd < mjd2)
    it_c = np.clip(it, 0, count2 - 4)
    tab = entries[it_c[:, None] + np.arange(4)[None, :]]  # (n, 4)

    f2_1 = (tab[:, 2] + tab[:, 0]) / 6.0
    y1_0 = 4.0 / 3.0 * tab[:, 1] - f2_1
    y2_0 = -1.0 / 3.0 * tab[:, 1] + f2_1
    f2_2 = (tab[:, 3] + tab[:, 1]) / 6.0
    y1_1 = 4.0 / 3.0 * tab[:, 2] - f2_2
    y2_1 = -1.0 / 3.0 * tab[:, 2] + f2_2

    ut1 = (t * (y1_1 + t * t * y2_1) + s * (y1_0 + s * s * y2_0)) * units
    lo = entries[0] * units
    hi = entries[-1] * units
    ut1 = np.where(in_range, ut1, np.where(mjd <= mjd1, lo, hi))
    return -ut1


def compute_correction_ut1_sec(
    sat_mjd: np.ndarray,
    correction_tt_sec: np.ndarray,
    clock_dir: str | None = None,
) -> np.ndarray:
    """``tai2ut1.C``: UT1-TOA = (UT1-TAI) + (TAI-TOA).

    ``correction_tt_sec`` is the full UTC(obs)->TT(TAI) chain correction that
    JUG already computes (``getCorrectionTT``); ``TAI-TOA`` is that minus the
    fixed 32.184 s TT-TAI offset.  This term rotates the site with the Earth,
    so the tens-of-seconds clock chain matters (~1 km of site position).
    """
    sat = np.asarray(sat_mjd, dtype=np.float64)
    correction_tai_sec = np.asarray(correction_tt_sec, dtype=np.float64) - 32.184
    return ut1red_sec(sat, clock_dir=clock_dir) + correction_tai_sec


def compute_nutations_rad(ephem_mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nutation angles (dpsi, deps) in radians.

    Tempo2 reads these from the JPL DE binary (``jpl_pleph`` target 14); the
    DE nutations follow the IAU 1980 theory, so ``erfa.nut80`` reproduces them
    to well under a milliarcsecond (sub-cm at the site).
    """
    import erfa

    mjd = np.asarray(ephem_mjd, dtype=np.float64)
    dpsi, deps = erfa.nut80(2400000.5, mjd)
    return dpsi, deps


def compute_tempo_t2c_observatory_earth(
    sat_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    *,
    correction_ut1_sec: np.ndarray,
    nutation_mjd: np.ndarray | None = None,
) -> np.ndarray:
    """Site geocentric position/velocity in km / km/s via the TEMPO method.

    Direct port of the ``t2cMethod != T2C_IAU2000B`` branch of
    ``get_obsCoord.C`` (equatorial output; any ecliptic rotation is applied
    downstream as for the astropy path).

    ``nutation_mjd`` is the ``readEphemeris.C`` epoch (``sat + TT + Teph``);
    tempo2 samples the DE nutations there.  Defaults to ``sat``.
    """
    sat = np.asarray(sat_mjd, dtype=np.float64)
    obs = np.asarray(obs_itrf_km, dtype=np.float64)
    n = len(sat)
    if obs.ndim == 1:
        obs = np.broadcast_to(obs.reshape(1, 3), (n, 3))

    ut1_mjd = sat + np.asarray(correction_ut1_sec, dtype=np.float64) / SECDAY

    dpsi, deps = compute_nutations_rad(sat if nutation_mjd is None else nutation_mjd)

    # Site cylindrical coordinates in light-seconds (obs x/y/z are in km here;
    # tempo2 uses metres with SPEED_LIGHT in m/s — identical ratio).
    erad = np.sqrt(np.sum(obs * obs, axis=1))
    hlt = np.arcsin(obs[:, 2] / erad)
    alng = np.arctan2(-obs[:, 1], obs[:, 0])
    hrd = erad / (C_KM_S * 499.004786)
    site0 = hrd * np.cos(hlt) * 499.004786  # distance from spin axis (lt-s)
    site1 = site0 * np.tan(hlt)  # z (lt-s)

    # Mean obliquity (IAU1976 polynomial, arcsec base 84381.448) at raw SAT
    toblq = (sat + 2400000.5 - 2451545.0) / 36525.0
    oblq = (((1.813e-3 * toblq - 5.9e-4) * toblq - 4.6815e1) * toblq + 84381.448) / 3600.0

    pc = np.cos(oblq * np.pi / 180.0 + deps) * dpsi

    tsid, _sdd = lmst(ut1_mjd, 0.0)
    tsid = tsid * 2.0 * np.pi

    ph = tsid + pc - alng
    eeq = np.zeros((n, 3), dtype=np.float64)
    eeq[:, 0] = site0 * np.cos(ph)
    eeq[:, 1] = site0 * np.sin(ph)
    eeq[:, 2] = site1

    prn = get_precession_nutation_matrix(ut1_mjd, dpsi, deps)
    obs_earth_ls = np.einsum("nij,nj->ni", prn, eeq)

    # Site velocity (get_obsCoord.C L424-431), lt-s/s
    speed = 2.0 * np.pi * site0 / (86400.0 / 1.00273)
    sitera = np.where(
        speed > 1.0e-10,
        np.arctan2(obs_earth_ls[:, 1], obs_earth_ls[:, 0]),
        0.0,
    )
    site_vel_ls = np.zeros((n, 3), dtype=np.float64)
    site_vel_ls[:, 0] = -np.sin(sitera) * speed
    site_vel_ls[:, 1] = np.cos(sitera) * speed

    out = np.zeros((n, 6), dtype=np.float64)
    out[:, :3] = obs_earth_ls * C_KM_S
    out[:, 3:] = site_vel_ls * C_KM_S
    return out
