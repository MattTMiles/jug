"""BT/DD binary model implementation with JAX JIT compilation.

This module implements the Blandford-Teukolsky (BT) and Damour-Deruelle (DD)
binary delay models for eccentric pulsar orbits using Keplerian + 1PN parameters.

References
----------
- Blandford & Teukolsky (1976), ApJ 205, 580
- Damour & Deruelle (1985), Ann. Inst. H. Poincaré (Physique Théorique) 43, 107
- Tempo2: T2model_BTmodel.C
- PINT: pint/models/binary_bt.py, pint/models/binary_dd.py
"""

import jax
import jax.numpy as jnp
from jug.utils.constants import SECS_PER_DAY


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
def bt_binary_delay(
    t_topo_tdb, pb, a1, ecc, om, t0, gamma, pbdot, 
    m2, sini, omdot=0.0, xdot=0.0
):
    """Compute BT/DD binary delay at given topocentric time.
    
    Parameters
    ----------
    t_topo_tdb : float
        Topocentric TDB time (MJD) at pulsar after removing non-binary delays
    pb : float
        Orbital period (days)
    a1 : float
        Projected semi-major axis (light-seconds)
    ecc : float
        Orbital eccentricity
    om : float
        Longitude of periastron (degrees)
    t0 : float
        Time of periastron passage (MJD)
    gamma : float
        Einstein delay parameter (seconds)
    pbdot : float
        Orbital period derivative (dimensionless, dP/dt)
    m2 : float
        Companion mass (solar masses) - for Shapiro delay
    sini : float
        Sine of inclination angle - for Shapiro delay
    omdot : float
        Periastron advance (degrees/year) - DD only
    xdot : float
        Rate of change of projected semi-major axis (light-sec/sec) - DD only
        
    Returns
    -------
    float
        Total binary delay (seconds): Roemer + Einstein + Shapiro
        
    Notes
    -----
    This implements the full BT/DD model with:
    1. Roemer delay: Light travel time in elliptical orbit
    2. Einstein delay: Time dilation from orbital motion
    3. Shapiro delay: Gravitational light bending by companion
    
    The DD model extends BT with OMDOT (periastron advance) and better
    Shapiro delay treatment. Both use the same Kepler solver.
    """
    # Time since periastron
    dt_days = t_topo_tdb - t0
    
    # Apply OMDOT correction (DD model)
    om_rad = jnp.deg2rad(om + omdot * dt_days / 365.25)
    
    # Mean motion with PBDOT correction
    pb_eff = pb * (1.0 + pbdot * dt_days / pb)
    n = 2.0 * jnp.pi / (pb_eff * SECS_PER_DAY)
    
    # Mean anomaly
    mean_anomaly = n * dt_days * SECS_PER_DAY
    
    # Solve Kepler's equation for eccentric anomaly
    ecc_anomaly = solve_kepler(mean_anomaly, ecc)
    
    # True anomaly (from eccentric anomaly)
    # tan(nu/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    true_anomaly = 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + ecc) * jnp.sin(ecc_anomaly / 2.0),
        jnp.sqrt(1.0 - ecc) * jnp.cos(ecc_anomaly / 2.0)
    )
    
    # Apply XDOT correction to semi-major axis
    a1_eff = jnp.where(xdot != 0.0, a1 + xdot * dt_days * SECS_PER_DAY, a1)
    
    # === ROEMER DELAY ===
    # Projected position along line of sight
    # x = a1 * [cos(E) - e]  (perpendicular to line of sight)
    # z = a1 * sqrt(1-e^2) * sin(E)  (along line of sight for edge-on orbit)
    # Projected delay = a1 * [sin(omega + nu) + e*sin(omega)]
    sin_omega_nu = jnp.sin(om_rad + true_anomaly)
    roemer_delay = a1_eff * (sin_omega_nu + ecc * jnp.sin(om_rad))
    
    # === EINSTEIN DELAY ===
    # Time dilation in eccentric orbit: gamma * sin(E)
    einstein_delay = jnp.where(gamma != 0.0, gamma * jnp.sin(ecc_anomaly), 0.0)
    
    # === SHAPIRO DELAY ===
    # Light bending by companion: -2*r*log(1 - s*sin(omega+nu))
    # r = RANGE = T_sun * M2 (companion mass in time units)
    # s = SINI = sin(inclination)
    T_SUN = 4.925490947e-6  # Solar mass in seconds
    r_shap = T_SUN * m2
    shapiro_delay = jnp.where(
        (m2 > 0.0) & (sini > 0.0),
        -2.0 * r_shap * jnp.log(1.0 - sini * sin_omega_nu),
        0.0
    )
    
    return roemer_delay + einstein_delay + shapiro_delay


@jax.jit  
def bt_binary_delay_vectorized(
    t_topo_tdb_array, pb, a1, ecc, om, t0, gamma, pbdot,
    m2, sini, omdot, xdot
):
    """Vectorized BT/DD binary delay for array of times.
    
    Parameters
    ----------
    t_topo_tdb_array : jnp.ndarray
        Array of topocentric TDB times (MJD)
    pb, a1, ecc, om, t0, gamma, pbdot, m2, sini, omdot, xdot : float
        Binary parameters (see bt_binary_delay)
        
    Returns
    -------
    jnp.ndarray
        Array of binary delays (seconds)
    """
    return jax.vmap(
        lambda t: bt_binary_delay(t, pb, a1, ecc, om, t0, gamma, pbdot, m2, sini, omdot, xdot)
    )(t_topo_tdb_array)
