"""Tempo2-native T2 binary delay (DD branch with Kopeikin terms).

Line-by-line port of the eccentric (DD) branch of tempo2 ``T2model.C``,
including the additive Kopeikin corrections tempo2 applies when KIN/KOM and
proper motion are set:

* ``DSR``  — secular proper-motion terms (Kopeikin 1996), linear in ``tt0``
* ``DAOP`` — annual-orbital parallax (Kopeikin 1995) from the SSB earth vector
* ``DOP``  — orbital parallax delay

JUG's generic DDK branch follows the PINT convention (modified A1/OM with a
rotated KOM); tempo2 instead keeps KIN/KOM in the IAU convention and adds the
Kopeikin delays directly to ``d2bar``.  Tempo2 also accumulates the true
anomaly as ``ae ∈ [0, 2π)`` per orbit (not ``(-π, π]``), which changes the
OMDOT-advanced omega by ``k·2π`` on half the TOAs — several µs for e.g.
J0437−4715.

Sign convention: returns the JUG binary delay (``+d2bar``); tempo2 stores
``torb = −d2bar``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jug.delays.binary_dd import solve_kepler
from jug.utils.constants import SECS_PER_DAY

# tempo2.h / T2model.C constants
AULTSC = 499.00478364
SUNMASS_SEC = 4.925490947e-6  # solar mass in seconds (T2model.C)
PX_MAS_TO_RAD = jnp.pi / 180.0 / 3600.0 * 1e-3


def t2_tempo2_binary_delay(
    t_bbat_mjd,
    pb_days,
    a1_lt_sec,
    ecc0,
    om_deg,
    t0_mjd,
    gamma_sec,
    pbdot,
    omdot_deg_yr,
    xdot,
    edot,
    sini,
    m2_msun,
    kin_deg,
    kom_deg,
    px_mas,
    pmra_rad_per_sec,
    pmdec_rad_per_sec,
    earth_ssb_ls,
    sin_alpha,
    cos_alpha,
    sin_delta,
    cos_delta,
    use_kopeikin,
):
    """T2model.C DD-branch delay at one ``bbat`` epoch (seconds).

    ``earth_ssb_ls`` is the per-TOA SSB→observer vector in light-seconds in
    the coordinate frame of the pulsar angles (equatorial, or ecliptic-rotated
    for ECL par files).  ``sin/cos_alpha/delta`` are the pulsar direction
    components tempo2 takes from ``psrPos``.
    """
    pb_sec = pb_days * SECS_PER_DAY
    an = 2.0 * jnp.pi / pb_sec
    tt0 = (t_bbat_mjd - t0_mjd) * SECS_PER_DAY

    ecc = ecc0 + edot * tt0
    x = a1_lt_sec + xdot * tt0
    m2 = m2_msun * SUNMASS_SEC
    omz_rad = jnp.deg2rad(om_deg)
    # getPostKeplerian: omdot [deg/yr] → rad/s ÷ an → rad per rad of ae
    omdot_k = jnp.deg2rad(omdot_deg_yr) / (365.25 * SECS_PER_DAY) / an

    orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec) ** 2
    norbits = jnp.floor(orbits)
    phase = 2.0 * jnp.pi * (orbits - norbits)

    u = solve_kepler(phase, ecc)
    su = jnp.sin(u)
    cu = jnp.cos(u)
    onemecu = 1.0 - ecc * cu

    # DD 17b/17c true anomaly, accumulated tempo2-style in [0, 2π) per orbit
    cae = (cu - ecc) / onemecu
    sae = jnp.sqrt(1.0 - ecc**2) * su / onemecu
    ae = jnp.arctan2(sae, cae)
    ae = jnp.where(ae < 0.0, ae + 2.0 * jnp.pi, ae)
    ae = 2.0 * jnp.pi * norbits + ae

    omega = omz_rad + omdot_k * ae
    sw = jnp.sin(omega)
    cw = jnp.cos(omega)
    sqr1me2 = jnp.sqrt(1.0 - ecc**2)
    cume = cu - ecc

    # er = ecc*(1+dr), eth = ecc*(1+dth); dr = dth = 0 here
    er = ecc
    eth = ecc

    # --- Kopeikin terms (KopeikinTerms + T2model.C L240-269) ---
    ki = jnp.deg2rad(kin_deg)
    si_k = jnp.where(use_kopeikin, jnp.sin(ki), 1.0)
    tani = jnp.where(use_kopeikin, jnp.tan(ki), 1.0)
    sin_kom = jnp.sin(jnp.deg2rad(kom_deg))
    cos_kom = jnp.cos(jnp.deg2rad(kom_deg))
    dpara = px_mas * PX_MAS_TO_RAD

    # Kopeikin 1995 eq 15/16 (earth vector in AU)
    ex = earth_ssb_ls[0] / AULTSC
    ey = earth_ssb_ls[1] / AULTSC
    ez = earth_ssb_ls[2] / AULTSC
    delta_i0 = -ex * sin_alpha + ey * cos_alpha
    delta_j0 = -ex * sin_delta * cos_alpha - ey * sin_delta * sin_alpha + ez * cos_delta

    dk011 = -x * dpara / si_k * delta_i0 * sin_kom
    dk012 = -x * dpara / si_k * delta_j0 * cos_kom
    dk021 = x * dpara / tani * delta_i0 * cos_kom
    dk022 = -x * dpara / tani * delta_j0 * sin_kom
    dk031 = x * tt0 / si_k * pmra_rad_per_sec * sin_kom
    dk032 = x * tt0 / si_k * pmdec_rad_per_sec * cos_kom
    dk041 = x * tt0 / tani * pmra_rad_per_sec * cos_kom
    dk042 = -x * tt0 / tani * pmdec_rad_per_sec * sin_kom

    c_geom = cw * (cu - er) - jnp.sqrt(1.0 - eth**2) * sw * su
    s_geom = sw * (cu - er) + cw * jnp.sqrt(1.0 - eth**2) * su

    daop = jnp.where(use_kopeikin, (dk011 + dk012) * c_geom - (dk021 + dk022) * s_geom, 0.0)
    dsr = jnp.where(use_kopeikin, (dk031 + dk032) * c_geom + (dk041 + dk042) * s_geom, 0.0)
    dop = jnp.where(
        use_kopeikin,
        dpara / AULTSC / 2.0 * x**2 * (
            si_k**-2.0 - 0.5
            + 0.5 * ecc**2 * (1.0 + sw**2 - 3.0 / si_k**2)
            - 2.0 * ecc * (si_k**-2.0 - sw**2) * cume
            - sqr1me2 * 2.0 * sw * cw * su * cume
            + 0.5 * (jnp.cos(2.0 * omega) + ecc**2 * (si_k**-2.0 + cu**2)) * jnp.cos(2.0 * u)
        ),
        0.0,
    )

    # --- DD equations 26, 46-52 ---
    brace = onemecu - sini * (sw * cume + sqr1me2 * cw * su)
    dlogbr = jnp.log(brace)
    ds = -2.0 * m2 * dlogbr

    alpha = x * sw
    beta = x * jnp.sqrt(1.0 - eth**2) * cw
    bg = beta + gamma_sec
    dre = alpha * (cu - er) + bg * su
    drep = -alpha * su + bg * cu
    drepp = -alpha * cu - bg * su
    anhat = an / onemecu

    d2bar = dre * (
        1.0
        - anhat * drep
        + anhat**2 * (drep**2 + 0.5 * dre * drepp - 0.5 * ecc * su * dre * drep / onemecu)
    ) + ds + daop + dsr + dop

    return d2bar
