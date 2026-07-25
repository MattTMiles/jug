"""JAX (jnp) twins of the barycentric astrometry functions.

These mirror the formulae of ``jug.delays.barycentric`` (the NumPy path) exactly,
but are traceable / differentiable in the astrometry parameters (RAJ, DECJ, PMRA,
PMDEC, PX) and the position epoch, so they can live inside a jitted likelihood.

Precision handling (matches the NumPy path):
  * The years-long ``dt = t - POSEPOCH`` baseline is reduced in the host/NumPy
    layer (longdouble subtract → float64) and passed in as ``dt_days``; the JAX
    twin never forms ``t_mjd - posepoch`` itself (which would lose precision at
    MJD ~58000).  This mirrors how ``combined_delays`` receives a precomputed
    ``tt_binary_sec`` rather than subtracting epochs inside the trace.
  * The SSB observatory position/velocity come from astropy (longdouble Time
    floor/frac split) in ``compute_ssb_obs_pos_vel`` and are already float64-
    reduced upstream; the dot-product delays below carry no further longdouble
    criticality, so float64 is the correct floor.

The NumPy functions in ``barycentric.py`` are NOT modified — JUG's own fitter
depends on them.  These are additive twins.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp

from jug.utils.constants import C_KM_S, AU_KM, KPC_TO_KM

# x64 is required for parity with the NumPy float64 path.
jax.config.update('jax_enable_x64', True)


@jax.jit
def pulsar_direction_jax(dt_days, ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day):
    """JAX twin of ``compute_pulsar_direction`` (rigorous great-circle PM).

    Parameters
    ----------
    dt_days : array (n_times,)
        Precomputed ``t_mjd - POSEPOCH`` in days, reduced in longdouble on the
        host then cast to float64.  (The twin does not subtract the epoch.)
    ra_rad, dec_rad : scalar
        Right ascension / declination at POSEPOCH (radians).
    pmra_rad_day, pmdec_rad_day : scalar
        On-sky proper motion rates (rad/day); ``pmra`` already includes cos(dec).

    Returns
    -------
    array (n_times, 3)
        Pulsar direction unit vectors in the celestial frame.

    Notes
    -----
    Great-circle propagation  p(t) = p0·cos(theta) + mhat·sin(theta), identical
    to the NumPy path (matches PINT/ERFA ``apply_space_motion`` to O((PM·dt)²)).
    When |PM| = 0, theta = 0 so sin(theta) kills the mhat term → p0; the mhat
    denominator is guarded so no NaN is produced in that (un-sampled) case.
    """
    dt_days = jnp.atleast_1d(dt_days)
    cos_dec0 = jnp.cos(dec_rad); sin_dec0 = jnp.sin(dec_rad)
    cos_ra0  = jnp.cos(ra_rad);  sin_ra0  = jnp.sin(ra_rad)

    # Direction at POSEPOCH and the on-sky tangent basis.
    p0    = jnp.stack([cos_dec0 * cos_ra0, cos_dec0 * sin_ra0, sin_dec0])
    e_ra  = jnp.stack([-sin_ra0, cos_ra0, jnp.zeros_like(sin_ra0)])
    e_dec = jnp.stack([-sin_dec0 * cos_ra0, -sin_dec0 * sin_ra0, cos_dec0])

    mu_vec = pmra_rad_day * e_ra + pmdec_rad_day * e_dec           # (3,)
    mu_mag = jnp.hypot(pmra_rad_day, pmdec_rad_day)                # scalar
    mu_safe = jnp.where(mu_mag > 0.0, mu_mag, 1.0)
    mhat   = mu_vec / mu_safe                                      # (3,); 0 when PM=0
    theta  = mu_mag * dt_days                                      # (n_times,)
    return (jnp.outer(jnp.cos(theta), p0) + jnp.outer(jnp.sin(theta), mhat))


@jax.jit
def roemer_delay_jax(ssb_obs_pos_km, L_hat, parallax_mas=0.0):
    """JAX twin of ``compute_roemer_delay`` (geometric delay + parallax).

    ``ssb_obs_pos_km`` (n,3) and ``L_hat`` (n,3) are the precomputed observatory
    position (float64, from astropy) and the pulsar direction.  Differentiable in
    ``parallax_mas`` and ``L_hat``.
    """
    re_dot_L  = jnp.sum(ssb_obs_pos_km * L_hat, axis=1)
    roemer_sec = -re_dot_L / C_KM_S

    re_sqr = jnp.sum(ssb_obs_pos_km ** 2, axis=1)
    re_safe = jnp.where(re_sqr > 0, re_sqr, 1.0)
    parallax_sec = jnp.where(
        re_sqr > 0,
        0.5 * re_sqr * (parallax_mas / KPC_TO_KM) * (1.0 - re_dot_L ** 2 / re_safe) / C_KM_S,
        0.0,
    )
    return roemer_sec + parallax_sec


@jax.jit
def shapiro_delay_jax(obs_body_pos_km, L_hat, T_body):
    """JAX twin of ``compute_shapiro_delay``:  -2·T·ln((r - r·cosθ)/AU)."""
    r         = jnp.sqrt(jnp.sum(obs_body_pos_km ** 2, axis=1))
    rcostheta = jnp.sum(obs_body_pos_km * L_hat, axis=1)
    return -2.0 * T_body * jnp.log((r - rcostheta) / AU_KM)


@jax.jit
def barycentric_freq_jax(freq_topo_mhz, ssb_obs_vel_km_s, L_hat, einstein_rate=None):
    """JAX twin of ``compute_barycentric_freq``:  f·(1 - v_radial/c) [/ einstein]."""
    v_radial  = jnp.sum(ssb_obs_vel_km_s * L_hat, axis=1)
    freq_bary = freq_topo_mhz * (1.0 - v_radial / C_KM_S)
    if einstein_rate is not None:
        freq_bary = freq_bary / einstein_rate
    return freq_bary
