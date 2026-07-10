"""DDGR binary model: GR-derived post-Keplerian parameters.

The DDGR model (Taylor & Weisberg 1989, Eqns. 15-25; tempo ``mass2dd``;
PINT ``DDGR_model.py``) assumes General Relativity is correct and DERIVES the
measurable post-Keplerian quantities -- inclination (SINI), Einstein delay
(GAMMA), orbital decay (PBDOT), periastron advance (OMDOT), and the
relativistic-deformation parameters (DR, DTH) -- from the system masses
(MTOT, M2) and the Keplerian elements (PB, A1, ECC).  These derived values are
then fed to the ordinary DD delay.

This module computes those PK parameters in geometric (T_sun) units so the
result matches PINT to ~1e-10 relative without separate G / c / M_sun factors:
``G M / c^3 = T_SUN_SEC * (M / M_sun)`` seconds.

Excess (non-GR) periastron advance / orbital decay are supported via
``xomdot_deg_yr`` / ``xpbdot`` (PINT's XOMDOT / XPBDOT, default 0).
"""

import numpy as np

from jug.utils.constants import T_SUN_SEC, SECS_PER_DAY

_SECS_PER_YEAR = 365.25 * SECS_PER_DAY


def compute_ddgr_pk_params(mtot_msun, m2_msun, pb_days, a1_ltsec, ecc,
                           xomdot_deg_yr=0.0, xpbdot=0.0, artol=1e-12,
                           max_iter=100):
    """Derive the DDGR post-Keplerian parameters from masses + Keplerian elements.

    Parameters
    ----------
    mtot_msun : float
        Total system mass (M_sun).
    m2_msun : float
        Companion mass (M_sun).
    pb_days : float
        Orbital period (days).
    a1_ltsec : float
        Projected semi-major axis of the pulsar orbit (light-seconds).
    ecc : float
        Eccentricity.
    xomdot_deg_yr : float, optional
        Excess periastron advance beyond GR (deg/yr), added to OMDOT.
    xpbdot : float, optional
        Excess orbital-period derivative beyond GR (dimensionless), added to PBDOT.
    artol : float, optional
        Fractional tolerance for the relativistic-Kepler iteration.
    max_iter : int, optional
        Iteration cap for the relativistic semi-major axis.

    Returns
    -------
    dict with keys:
        ``sini`` (dimensionless), ``gamma_sec`` (s), ``pbdot`` (dimensionless,
        s/s), ``omdot_deg_yr`` (deg/yr), ``dr`` and ``dth`` (dimensionless).

    Notes
    -----
    Geometric formulation (lengths in light-seconds = length / c):
      T = T_SUN_SEC, n = 2*pi/PB, masses m = M/M_sun.
      arr0 = (T*mtot / n^2)^(1/3)                      [Newtonian, lt-s]
      arr  = arr0 * (1 + (m1*m2/mtot^2 - 9) * T*mtot/(2*arr))^(2/3)  [relativistic]
      SINI  = a1 / (arr * m2/mtot)
      GAMMA = ecc * T * m2 * (m1 + 2*m2) / (mtot * n * arr0)
      PBDOT = -(192*pi/5) * (T*n)^(5/3) * m1*m2*mtot^(-1/3) * fe
      OMDOT = 3 * n^(5/3) / (1-e^2) * (T*mtot)^(2/3)    [rad/s -> deg/yr]
      DR    = T * (3*m1^2 + 6*m1*m2 + 2*m2^2)  / (mtot*arr)
      DTH   = T * (3.5*m1^2 + 6*m1*m2 + 2*m2^2) / (mtot*arr)
    Validated against PINT DDGR_model._updatePK to ~1e-10 relative (J0955-6150).
    """
    T = float(T_SUN_SEC)
    mtot = float(mtot_msun)
    m2 = float(m2_msun)
    m1 = mtot - m2
    e = float(ecc)
    n = 2.0 * np.pi / (float(pb_days) * SECS_PER_DAY)  # rad/s

    arr0 = (T * mtot / n ** 2) ** (1.0 / 3.0)          # light-seconds
    arr = arr0
    for _ in range(max_iter):
        prev = arr
        arr = arr0 * (1.0 + (m1 * m2 / mtot ** 2 - 9.0)
                      * (T * mtot / (2.0 * arr))) ** (2.0 / 3.0)
        if abs((arr - prev) / arr) < artol:
            break

    sini = float(a1_ltsec) / (arr * (m2 / mtot))
    gamma_sec = e * T * m2 * (m1 + 2.0 * m2) / (mtot * n * arr0)

    fe = (1.0 + (73.0 / 24.0) * e ** 2 + (37.0 / 96.0) * e ** 4) \
        * (1.0 - e ** 2) ** (-3.5)
    pbdot = -(192.0 * np.pi / 5.0) * (T * n) ** (5.0 / 3.0) \
        * m1 * m2 * mtot ** (-1.0 / 3.0) * fe + float(xpbdot)

    omdot_rad_s = 3.0 * n ** (5.0 / 3.0) / (1.0 - e ** 2) * (T * mtot) ** (2.0 / 3.0)
    omdot_deg_yr = omdot_rad_s * (180.0 / np.pi) * _SECS_PER_YEAR + float(xomdot_deg_yr)

    dr = T * (3.0 * m1 ** 2 + 6.0 * m1 * m2 + 2.0 * m2 ** 2) / (mtot * arr)
    dth = T * (3.5 * m1 ** 2 + 6.0 * m1 * m2 + 2.0 * m2 ** 2) / (mtot * arr)

    return {
        'sini': sini, 'gamma_sec': gamma_sec, 'pbdot': pbdot,
        'omdot_deg_yr': omdot_deg_yr, 'dr': dr, 'dth': dth,
    }


def compute_ddgr_pk_derivatives(mtot_msun, m2_msun, pb_days, a1_ltsec, ecc,
                                artol=1e-12, max_iter=100):
    """Analytic d(PK)/d(param) for the DDGR-derived post-Keplerian parameters.

    Returns a dict of scalar partials of SINI, GAMMA, PBDOT, OMDOT(deg/yr) with
    respect to MTOT, M2, PB(days), A1(lt-s), ECC. Used to chain the DDGR mass
    fit onto JUG's existing per-PK DD delay derivatives:

        d(delay)/d(MTOT) = sum_PK  d(delay)/d(PK) * d(PK)/d(MTOT)

    Mirrors PINT DDGR_model (d_SINI_d_*, d_GAMMA_d_*, d_PBDOT_d_*, ... ). The
    relativistic semi-major axis ``arr`` is differentiated IMPLICITLY from its
    fixed-point definition (equivalent to PINT's expanded d_arr_d_*, validated
    numerically). DR/DTH partials are omitted: their delay contribution chained
    onto d(MTOT)/d(M2) is < 1e-3 of the SINI/GAMMA terms (negligible for the
    fit; the forward model still carries DR/DTH exactly).
    """
    T = float(T_SUN_SEC)
    mtot = float(mtot_msun); m2 = float(m2_msun); m1 = mtot - m2
    e = float(ecc); a1 = float(a1_ltsec)
    n = 2.0 * np.pi / (float(pb_days) * SECS_PER_DAY)

    arr0 = (T * mtot / n ** 2) ** (1.0 / 3.0)
    arr = arr0
    for _ in range(max_iter):
        prev = arr
        arr = arr0 * (1.0 + (m1 * m2 / mtot ** 2 - 9.0) * (T * mtot / (2.0 * arr))) ** (2.0 / 3.0)
        if abs((arr - prev) / arr) < artol:
            break

    # arr0 depends on mtot, n (i.e. PB) only.
    d_arr0_d_mtot = arr0 / (3.0 * mtot)
    d_arr0_d_n = -2.0 / 3.0 * arr0 / n
    # X = (m1*m2/mtot^2 - 9) * T*mtot/2 ; arr = arr0*(1 + X/arr)^(2/3)
    C = m1 * m2 / mtot ** 2 - 9.0
    X = C * T * mtot / 2.0
    g = 1.0 + X / arr                       # base of the (..)^(2/3)
    # dF/d_arr for F = arr - arr0*g^(2/3)
    dF_darr = 1.0 + (2.0 / 3.0) * arr0 * X / arr ** 2 * g ** (-1.0 / 3.0)

    def d_arr(dp_arr0, dp_X):
        # dF/dp = -dp_arr0*g^(2/3) - arr0*(2/3)*g^(-1/3)*(dp_X/arr)
        dFdp = -dp_arr0 * g ** (2.0 / 3.0) - arr0 * (2.0 / 3.0) * g ** (-1.0 / 3.0) * (dp_X / arr)
        return -dFdp / dF_darr

    # dX/d_param.  C = (mtot-m2)*m2/mtot^2 - 9 = (m2/mtot) - (m2/mtot)^2 - 9
    dC_dmtot = -m2 / mtot ** 2 + 2.0 * m2 ** 2 / mtot ** 3
    dC_dm2 = 1.0 / mtot - 2.0 * m2 / mtot ** 2
    dX_dmtot = dC_dmtot * T * mtot / 2.0 + C * T / 2.0
    dX_dm2 = dC_dm2 * T * mtot / 2.0
    d_arr_d_mtot = d_arr(d_arr0_d_mtot, dX_dmtot)
    d_arr_d_m2 = d_arr(0.0, dX_dm2)  # arr0 independent of m2
    # PB enters only through n; chain via arr0 (X has no n).
    dn_dpb = -n / float(pb_days)
    d_arr_d_pb = d_arr(d_arr0_d_n * dn_dpb, 0.0)

    # SINI = a1 * mtot / (arr * m2)
    sini = a1 * mtot / (arr * m2)
    d = {}
    d['sini_mtot'] = sini * (1.0 / mtot - d_arr_d_mtot / arr)
    d['sini_m2'] = sini * (-1.0 / m2 - d_arr_d_m2 / arr)
    d['sini_pb'] = sini * (-d_arr_d_pb / arr)
    d['sini_a1'] = sini / a1
    d['sini_ecc'] = 0.0

    # GAMMA = ecc*T*m2*(mtot+m2)/(mtot*n*arr0)
    gamma = e * T * m2 * (mtot + m2) / (mtot * n * arr0)
    d['gamma_mtot'] = gamma * (1.0 / (mtot + m2) - 1.0 / mtot - d_arr0_d_mtot / arr0)
    d['gamma_m2'] = gamma * (1.0 / m2 + 1.0 / (mtot + m2))
    d['gamma_pb'] = gamma * (-(1.0 / n + d_arr0_d_n / arr0) * dn_dpb)
    d['gamma_a1'] = 0.0
    d['gamma_ecc'] = gamma / e

    # PBDOT = -(192pi/5)*(T*n)^(5/3)*m1*m2*mtot^(-1/3)*fe
    fe = (1.0 + (73.0 / 24.0) * e ** 2 + (37.0 / 96.0) * e ** 4) * (1.0 - e ** 2) ** (-3.5)
    pbdot = -(192.0 * np.pi / 5.0) * (T * n) ** (5.0 / 3.0) * m1 * m2 * mtot ** (-1.0 / 3.0) * fe
    d['pbdot_mtot'] = pbdot * (1.0 / m1 - 1.0 / (3.0 * mtot))  # d ln|pbdot|/d mtot: +1/m1 -1/(3mtot)
    d['pbdot_m2'] = pbdot * (1.0 / m2 - 1.0 / m1)
    d['pbdot_pb'] = pbdot * ((5.0 / 3.0) / n * dn_dpb)
    d['pbdot_a1'] = 0.0
    dfe_de = ((73.0 / 12.0) * e + (37.0 / 24.0) * e ** 3) * (1.0 - e ** 2) ** (-3.5) \
        + (1.0 + (73.0 / 24.0) * e ** 2 + (37.0 / 96.0) * e ** 4) * 7.0 * e * (1.0 - e ** 2) ** (-4.5)
    d['pbdot_ecc'] = pbdot * (dfe_de / fe)

    # OMDOT(deg/yr) = 3*n^(5/3)/(1-e^2)*(T*mtot)^(2/3) * (180/pi)*SECS_PER_YEAR
    omdot = 3.0 * n ** (5.0 / 3.0) / (1.0 - e ** 2) * (T * mtot) ** (2.0 / 3.0) \
        * (180.0 / np.pi) * _SECS_PER_YEAR
    d['omdot_mtot'] = omdot * (2.0 / 3.0) / mtot
    d['omdot_m2'] = 0.0
    d['omdot_pb'] = omdot * ((5.0 / 3.0) / n * dn_dpb)
    d['omdot_a1'] = 0.0
    d['omdot_ecc'] = omdot * (2.0 * e / (1.0 - e ** 2))
    return d
