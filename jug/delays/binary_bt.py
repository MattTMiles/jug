"""Blandford-Teukolsky (BT) binary model with JAX JIT compilation.

This module implements the Blandford-Teukolsky (BT) and Damour-Deruelle (DD)
binary delay models for eccentric pulsar orbits using Keplerian + 1PN parameters.

References
----------
- Blandford & Teukolsky (1976), ApJ 205, 580
- Damour & Deruelle (1985), Ann. Inst. H. Poincare (Physique Theorique) 43, 107
- Tempo2: T2model_BTmodel.C
- PINT: pint/models/binary_bt.py, pint/models/binary_dd.py
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jug.io.par_reader import get_longdouble
from jug.utils.constants import SECS_PER_DAY

JULIAN_YEAR_SEC = 365.25 * SECS_PER_DAY


@jax.jit
def solve_kepler(mean_anomaly, eccentricity, tol=1e-12, max_iter=20):
    """Solve Kepler's equation E - e*sin(E) = M using Newton-Raphson.
    
    Parameters
    ----------
    mean_anomaly : float or jnp.ndarray
        Mean anomaly M (radians)
    eccentricity : float
        Orbital eccentricity (0 <= e < 1)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float or jnp.ndarray
        Eccentric anomaly E (radians)
        
    Notes
    -----
    Starting guess: E0 = M (works well for low to moderate eccentricity)
    Newton-Raphson: E_{n+1} = E_n - f/f' where f = E - e*sin(E) - M
    """
    E = mean_anomaly
    
    # Fixed number of iterations for JAX JIT
    def iteration(carry, _):
        E = carry
        f = E - eccentricity * jnp.sin(E) - mean_anomaly
        fp = 1.0 - eccentricity * jnp.cos(E)
        E_new = E - f / fp
        return E_new, None
    
    E, _ = jax.lax.scan(iteration, E, None, length=max_iter)
    return E


@jax.jit
def bt_binary_delay_from_tt0(
    tt0_sec, pb, a1, ecc, om, gamma, pbdot,
    omdot=0.0, xdot=0.0, edot=0.0,
):
    """Compute PINT-compatible BT delay from ``(BAT - T0)`` in seconds."""
    pb_sec = pb * SECS_PER_DAY
    orbits = tt0_sec / pb_sec - 0.5 * pbdot * (tt0_sec / pb_sec) ** 2
    mean_anomaly = 2.0 * jnp.pi * orbits

    ecc_current = ecc + edot * tt0_sec
    a1_current = a1 + xdot * tt0_sec
    om_current = om + omdot * tt0_sec / JULIAN_YEAR_SEC
    om_rad = jnp.deg2rad(om_current)

    E = solve_kepler(mean_anomaly, ecc_current)
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    sinOm = jnp.sin(om_rad)
    cosOm = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1.0 - ecc_current**2)

    delay_l1 = a1_current * sinOm * (cosE - ecc_current)
    delay_l2 = (a1_current * cosOm * sqrt_1_e2 + gamma) * sinE

    numerator = (
        a1_current * cosOm * sqrt_1_e2 * cosE
        - a1_current * sinOm * sinE
    )
    denominator = 1.0 - ecc_current * cosE
    pb_prime_sec = pb_sec + pbdot * tt0_sec
    delay_r = 1.0 - 2.0 * jnp.pi * numerator / (denominator * pb_prime_sec)
    return (delay_l1 + delay_l2) * delay_r


@jax.jit
def bt_binary_delay(
    t_topo_tdb, pb, a1, ecc, om, t0, gamma, pbdot,
    m2=0.0, sini=0.0, omdot=0.0, xdot=0.0, edot=0.0,
):
    """Backward-compatible MJD entry point."""
    tt0_sec = (t_topo_tdb - t0) * SECS_PER_DAY
    return bt_binary_delay_from_tt0(
        tt0_sec, pb, a1, ecc, om, gamma, pbdot, omdot, xdot, edot
    )


@jax.jit
def bt_binary_delay_vectorized(
    t_topo_tdb_array, pb, a1, ecc, om, t0, gamma, pbdot,
    m2=0.0, sini=0.0, omdot=0.0, xdot=0.0, edot=0.0,
):
    """Backward-compatible vectorized MJD entry point."""
    return bt_binary_delay(
        t_topo_tdb_array, pb, a1, ecc, om, t0, gamma, pbdot,
        m2, sini, omdot, xdot, edot,
    )


def _compute_tt0_sec(toas_bary_mjd, t0):
    return np.asarray(
        (np.asarray(toas_bary_mjd, dtype=np.longdouble) - np.longdouble(t0))
        * np.longdouble(SECS_PER_DAY),
        dtype=np.float64,
    )


def compute_bt_binary_delay(toas_bary_mjd, params, **kwargs):
    """Fitter/registry BT delay entry point using high-precision epoch subtraction."""
    tt0_sec = _compute_tt0_sec(
        toas_bary_mjd, get_longdouble(params, "T0", default=0.0)
    )
    return bt_binary_delay_from_tt0(
        jnp.asarray(tt0_sec),
        float(params.get("PB", 0.0)),
        float(params.get("A1", 0.0)),
        float(params.get("ECC", params.get("E", 0.0))),
        float(params.get("OM", 0.0)),
        float(params.get("GAMMA", 0.0)),
        float(params.get("PBDOT", 0.0)),
        float(params.get("OMDOT", 0.0)),
        float(params.get("XDOT", params.get("A1DOT", 0.0))),
        float(params.get("EDOT", 0.0)),
    )


def compute_binary_derivatives_bt(params, toas_bary_mjd, fit_params):
    """BT delay derivatives in seconds per par-file unit."""
    tt0 = jnp.asarray(_compute_tt0_sec(
        toas_bary_mjd, get_longdouble(params, "T0", default=0.0)
    ))
    values = (
        float(params.get("PB", 0.0)),
        float(params.get("A1", 0.0)),
        float(params.get("ECC", params.get("E", 0.0))),
        float(params.get("OM", 0.0)),
        float(params.get("GAMMA", 0.0)),
        float(params.get("PBDOT", 0.0)),
        float(params.get("OMDOT", 0.0)),
        float(params.get("XDOT", params.get("A1DOT", 0.0))),
        float(params.get("EDOT", 0.0)),
    )

    def delay_for_args(pb, a1, ecc, om, gamma, pbdot, omdot, xdot, edot):
        return bt_binary_delay_from_tt0(
            tt0, pb, a1, ecc, om, gamma, pbdot, omdot, xdot, edot
        )

    arg_index = {
        "PB": 0, "A1": 1, "ECC": 2, "E": 2, "OM": 3, "GAMMA": 4,
        "PBDOT": 5, "OMDOT": 6, "XDOT": 7, "A1DOT": 7, "EDOT": 8,
    }
    derivatives = {}
    for param in fit_params:
        name = param.upper()
        if name == "T0":
            def scalar_delay(t, *args):
                return bt_binary_delay_from_tt0(t, *args)
            d_dt = jax.vmap(
                jax.grad(scalar_delay, argnums=0),
                in_axes=(0,) + (None,) * 9,
            )(tt0, *values)
            derivatives[param] = d_dt * -SECS_PER_DAY
        elif name in arg_index:
            derivatives[param] = jax.jacfwd(
                delay_for_args, argnums=arg_index[name]
            )(*values)
        else:
            derivatives[param] = jnp.zeros_like(tt0)
    return derivatives
