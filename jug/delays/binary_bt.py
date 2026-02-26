"""BT/DD binary model implementation with JAX JIT compilation.

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
from jug.utils.constants import SECS_PER_DAY, T_SUN


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
        Periastron advance (degrees/year) - DD model only
    xdot : float
        Rate of change of projected semi-major axis (light-sec/sec) - DD model only

    Returns
    -------
    float
        Total binary delay (seconds)

    Notes
    -----
    This implements the DD (Damour-Deruelle) binary model following PINT's
    implementation in pint/models/stand_alone_psr_binaries/DD_model.py

    References:
    - Damour & Deruelle (1985), Ann. Inst. H. Poincare 43, 107
    - Damour & Deruelle (1986), paper with equations [25]-[52]

    The DD model includes:
    1. Inverse delay: Roemer + Einstein with coordinate time correction
    2. Shapiro delay: Gravitational light bending by companion
    3. Aberration delay: A0, B0 terms (assumed zero if not provided)

    For BT model (OMDOT=0, XDOT=0), this reduces to standard Keplerian + 1PN.
    """
    # Time since periastron
    dt_days = t_topo_tdb - t0
    dt_sec = dt_days * SECS_PER_DAY

    # Apply OMDOT correction (DD model)
    # omega(t) = OM + OMDOT * (t - T0) / 1 year
    om_current = om + omdot * dt_days / 365.25
    om_rad = jnp.deg2rad(om_current)

    # Mean motion with PBDOT correction
    pb_eff = pb * (1.0 + pbdot * dt_days / pb)

    # Mean anomaly
    n = 2.0 * jnp.pi / (pb_eff * SECS_PER_DAY)
    mean_anomaly = n * dt_sec

    # Solve Kepler's equation for eccentric anomaly E
    E = solve_kepler(mean_anomaly, ecc)

    # Trigonometric functions of E and omega
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    sinOm = jnp.sin(om_rad)
    cosOm = jnp.cos(om_rad)

    # Apply XDOT correction to semi-major axis
    a1_eff = a1 + xdot * dt_sec

    # =========================================================================
    # INVERSE DELAY (Roemer + Einstein with coordinate time correction)
    # Following PINT's DD_model.py: delayInverse()
    # Reference: Damour & Deruelle (1986) equations [43], [46-52]
    # =========================================================================

    # Alpha and Beta parameters (D&D eqs [46], [47])
    # Note: er = ecc*(1 + DR), eTheta = ecc*(1 + DTH)
    # For now, assume DR=0, DTH=0 (standard DD model)
    er = ecc
    eTheta = ecc

    alpha = a1_eff * sinOm  # eq [46]
    beta = a1_eff * jnp.sqrt(1.0 - eTheta**2) * cosOm  # eq [47]

    # Roemer delay: delayR = alpha*(cos(E) - er) + beta*sin(E)
    # (D&D eq [48] with delayE separated out)
    delayR = alpha * (cosE - er) + beta * sinE

    # Einstein delay: delayE = GAMMA * sin(E)  (D&D eq [25])
    delayE = gamma * sinE

    # Dre = Roemer + Einstein (D&D eq [48])
    Dre = delayR + delayE

    # First derivative: Drep = d(Dre)/dE  (D&D eq [49])
    Drep = -alpha * sinE + (beta + gamma) * cosE

    # Second derivative: Drepp = d^2(Dre)/dE^2  (D&D eq [50])
    Drepp = -alpha * cosE - (beta + gamma) * sinE

    # nhat = dE/dt  (D&D eq [51])
    nhat = (2.0 * jnp.pi / (pb_eff * SECS_PER_DAY)) / (1.0 - ecc * cosE)

    # Inverse delay transformation (D&D eq [52])
    # This accounts for the fact that delays affect orbital position
    # delayInverse = Dre(t - Dre(t - Dre(t))) approximated via Taylor expansion
    correction_factor = (
        1.0
        - nhat * Drep
        + (nhat * Drep)**2
        + 0.5 * nhat**2 * Dre * Drepp
        - 0.5 * ecc * sinE / (1.0 - ecc * cosE) * nhat**2 * Dre * Drep
    )

    delayInverse = Dre * correction_factor

    # =========================================================================
    # SHAPIRO DELAY
    # Following PINT's DD_model.py: delayS()
    # Reference: Damour & Deruelle (1986) equation [26]
    # =========================================================================

    # DD Shapiro delay uses full orbital geometry, not just sin(omega+nu)
    # delayS = -2*r*log(1 - e*cos(E) - SINI*[sin(omega)*(cos(E)-e) +
    #                                         sqrt(1-e^2)*cos(omega)*sin(E)])
    shapiro_arg = (
        1.0
        - ecc * cosE
        - sini * (sinOm * (cosE - ecc) + jnp.sqrt(1.0 - ecc**2) * cosOm * sinE)
    )

    delayS = jnp.where(
        (m2 > 0.0) & (sini > 0.0),
        -2.0 * T_SUN * m2 * jnp.log(shapiro_arg),
        0.0
    )

    # =========================================================================
    # ABERRATION DELAY
    # Following PINT's DD_model.py: delayA()
    # Reference: Damour & Deruelle (1986) equation [27]
    # =========================================================================
    # For now, assume A0=0, B0=0 (not provided in typical par files)
    # delayA = A0*(sin(omega+nu) + e*sin(omega)) + B0*(cos(omega+nu) + e*cos(omega))
    delayA = 0.0

    # Total DD delay
    return delayInverse + delayS + delayA


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
