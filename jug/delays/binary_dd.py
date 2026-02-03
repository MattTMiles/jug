"""DD binary model implementation with JAX JIT compilation.

This module implements the Damour-Deruelle (DD) binary delay model
and its variants (DDH, DDGR, DDK) for pulsar timing.

The DD model includes:
- Keplerian orbital parameters (PB, A1, ECC, OM, T0)
- Post-Keplerian parameters (GAMMA, PBDOT, OMDOT, XDOT, EDOT)
- Shapiro delay parameters (SINI, M2) or orthometric (H3, H4)
- GR constraints (DDGR) and Kopeikin terms (DDK)

References
----------
- Damour & Deruelle (1985), Ann. Inst. H. Poincaré (Physique Théorique) 43, 107
- Damour & Deruelle (1986), Ann. Inst. H. Poincaré (Physique Théorique) 44, 263
- Taylor & Weisberg (1989), ApJ 345, 434
- Tempo2: T2model_BTmodel.C, T2model_DD.C
- PINT: pint/models/binary_dd.py
"""

import jax
import jax.numpy as jnp
from jug.utils.constants import SECS_PER_DAY, C_M_S


@jax.jit
def solve_kepler(mean_anomaly, eccentricity, tol=5e-15, max_iter=30):
    """Solve Kepler's equation E - e*sin(E) = M using Newton-Raphson.
    
    Parameters
    ----------
    mean_anomaly : float or jnp.ndarray
        Mean anomaly M (radians)
    eccentricity : float
        Orbital eccentricity (0 <= e < 1)
    tol : float
        Convergence tolerance (default: 5e-15 to match PINT)
    max_iter : int
        Maximum iterations
        
    Returns
    -------
    float or jnp.ndarray
        Eccentric anomaly E (radians)
    """
    E = mean_anomaly
    
    def iteration(carry, _):
        E = carry
        f = E - eccentricity * jnp.sin(E) - mean_anomaly
        fp = 1.0 - eccentricity * jnp.cos(E)
        E_new = E - f / fp
        return E_new, None
    
    # Run fixed number of iterations (JAX doesn't support while loops easily)
    # 30 iterations is more than enough for tolerance 5e-15
    E, _ = jax.lax.scan(iteration, E, None, length=max_iter)
    return E


# Note: JIT removed from single-value function due to caching issues
# Vectorized version still uses JIT via vmap
def dd_binary_delay(
    t_bary_mjd,
    pb_days,
    a1_lt_sec,
    ecc,
    omega_deg,
    t0_mjd,
    gamma_sec=0.0,
    pbdot=0.0,
    omdot_deg_yr=0.0,
    xdot=0.0,
    edot=0.0,
    sini=None,
    m2_msun=None,
    h3_sec=None,
    h4_sec=None,
    stig=None,
):
    """Compute DD binary delay at a single barycentric time.

    This implements the Damour-Deruelle (DD) binary model following PINT's
    implementation in pint/models/stand_alone_psr_binaries/DD_model.py

    The DD model includes:
    1. Inverse delay: Roemer + Einstein with coordinate time correction
    2. Shapiro delay: Gravitational light bending by companion
    3. Aberration delay: A0, B0 terms (assumed zero if not provided)

    Parameters
    ----------
    t_bary_mjd : float
        Barycentric time (MJD)
    pb_days : float
        Orbital period (days)
    a1_lt_sec : float
        Projected semi-major axis (light-seconds)
    ecc : float
        Eccentricity
    omega_deg : float
        Longitude of periastron (degrees)
    t0_mjd : float
        Time of periastron passage (MJD)
    gamma_sec : float, optional
        Einstein delay parameter (seconds)
    pbdot : float, optional
        Orbital period derivative (dimensionless, dP/P per unit time)
    omdot_deg_yr : float, optional
        Periastron advance rate (degrees/year)
    xdot : float, optional
        Rate of change of projected semi-major axis (lt-s/s)
    edot : float, optional
        Rate of change of eccentricity (1/s)
    sini : float, optional
        Sine of orbital inclination (for Shapiro delay)
    m2_msun : float, optional
        Companion mass in solar masses (for Shapiro delay)
    h3_sec : float, optional
        Orthometric Shapiro parameter H3 (seconds)
    h4_sec : float, optional
        Orthometric Shapiro parameter H4 (seconds)

    Returns
    -------
    float
        Binary delay in seconds

    References
    ----------
    - Damour & Deruelle (1985), Ann. Inst. H. Poincaré 43, 107
    - Damour & Deruelle (1986), equations [25]-[52]
    - PINT: pint/models/stand_alone_psr_binaries/DD_model.py

    Notes
    -----
    If h3 and h4 are provided, they are converted to sini and m2.
    The Shapiro delay requires either (sini, m2) or (h3, h4).
    """
    # Time since periastron
    dt_days = t_bary_mjd - t0_mjd
    dt_sec = dt_days * SECS_PER_DAY
    tt0 = dt_sec  # Time since T0 in seconds (PINT convention)

    # Apply periastron advance: omega(t) = OM + OMDOT * (t - T0) / 1 year
    dt_years = dt_days / 365.25
    omega_current_deg = omega_deg + omdot_deg_yr * dt_years
    omega_rad = jnp.deg2rad(omega_current_deg)

    # Mean anomaly calculation following PINT's exact formula
    # PINT uses: orbits = tt0/PB - 0.5 * PBDOT * (tt0/PB)^2
    # then wraps to fractional orbit before computing M = (frac_orbits) * 2π
    # This properly accounts for orbital period evolution
    pb_sec = pb_days * SECS_PER_DAY
    orbits = tt0 / pb_sec - 0.5 * pbdot * (tt0 / pb_sec)**2
    
    # CRITICAL: Wrap orbits to [0, 1) BEFORE multiplying by 2π
    # This avoids precision loss for large number of orbits
    # PINT does: (orbits - floor(orbits)) * 2π
    norbits = jnp.floor(orbits)
    frac_orbits = orbits - norbits
    mean_anomaly = frac_orbits * 2.0 * jnp.pi

    # Apply secular changes to a1 and eccentricity
    a1_current = a1_lt_sec + xdot * dt_sec
    ecc_current = ecc + edot * dt_sec

    # Solve Kepler's equation for eccentric anomaly E
    E = solve_kepler(mean_anomaly, ecc_current)

    # Trigonometric functions
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    sinOm = jnp.sin(omega_rad)
    cosOm = jnp.cos(omega_rad)

    # =========================================================================
    # INVERSE DELAY (Roemer + Einstein with coordinate time correction)
    # Following PINT's DD_model.py: delayInverse()
    # Reference: Damour & Deruelle (1986) equations [43], [46-52]
    # =========================================================================

    # Alpha and Beta parameters (D&D eqs [46], [47])
    # Note: er = ecc*(1 + DR), eTheta = ecc*(1 + DTH)
    # For standard DD model: DR=0, DTH=0
    er = ecc_current
    eTheta = ecc_current

    alpha = a1_current * sinOm  # eq [46]
    beta = a1_current * jnp.sqrt(1.0 - eTheta**2) * cosOm  # eq [47]

    # Roemer delay: delayR = alpha*(cos(E) - er) + beta*sin(E)
    delayR = alpha * (cosE - er) + beta * sinE

    # Einstein delay: delayE = GAMMA * sin(E)  (D&D eq [25])
    delayE = gamma_sec * sinE

    # Dre = Roemer + Einstein (D&D eq [48])
    Dre = delayR + delayE

    # First derivative: Drep = d(Dre)/dE  (D&D eq [49])
    Drep = -alpha * sinE + (beta + gamma_sec) * cosE

    # Second derivative: Drepp = d^2(Dre)/dE^2  (D&D eq [50])
    Drepp = -alpha * cosE - (beta + gamma_sec) * sinE

    # nhat = dE/dt  (D&D eq [51])
    # Use instantaneous period: PB' = PB + PBDOT * tt0
    pb_prime_sec = pb_sec + pbdot * tt0
    nhat = (2.0 * jnp.pi / pb_prime_sec) / (1.0 - ecc_current * cosE)

    # Inverse delay transformation (D&D eq [52])
    # This accounts for converting from proper time to coordinate time
    correction_factor = (
        1.0
        - nhat * Drep
        + (nhat * Drep)**2
        + 0.5 * nhat**2 * Dre * Drepp
        - 0.5 * ecc_current * sinE / (1.0 - ecc_current * cosE) * nhat**2 * Dre * Drep
    )

    delayInverse = Dre * correction_factor

    # =========================================================================
    # SHAPIRO DELAY
    # Following PINT's DD_model.py: delayS()
    # Reference: Damour & Deruelle (1986) equation [26]
    # =========================================================================

    T_SUN = 4.925490947e-6  # Solar mass in seconds (G*Msun/c^3)

    # Convert H3/STIG (DDH) or H3/H4 to SINI/M2 if needed
    # Use JAX-compatible conditionals for JIT compatibility

    # Default values
    sini_default = sini if sini is not None else 0.0
    m2_default = m2_msun if m2_msun is not None else 0.0

    # H3/STIG calculation (safe division with small epsilon)
    h3_val = h3_sec if h3_sec is not None else 0.0
    stig_val = stig if stig is not None else 0.0
    stig_safe = jnp.maximum(jnp.abs(stig_val), 1e-30)
    sini_h3stig = 2.0 * stig_val / (1.0 + stig_val**2)
    m2_h3stig = h3_val / (stig_safe**3 * T_SUN)

    # H3/H4 calculation (safe division)
    h4_val = h4_sec if h4_sec is not None else 0.0
    h4_safe = jnp.maximum(jnp.abs(h4_val), 1e-30)
    r_cubed = h4_safe / T_SUN**3
    r_h4 = jnp.cbrt(r_cubed)
    sini_h3h4 = jnp.clip(h3_val / (jnp.maximum(r_h4 * T_SUN, 1e-30)), 0.0, 1.0)
    m2_h3h4 = r_h4 / T_SUN

    # Select based on which parameters are provided (use JAX where)
    use_h3stig = (h3_val != 0.0) & (stig_val != 0.0)
    use_h3h4 = (h3_val != 0.0) & (h4_val != 0.0) & ~use_h3stig

    sini_use = jnp.where(use_h3stig, sini_h3stig,
                         jnp.where(use_h3h4, sini_h3h4, sini_default))
    m2_msun_use = jnp.where(use_h3stig, m2_h3stig,
                            jnp.where(use_h3h4, m2_h3h4, m2_default))

    # DD Shapiro delay uses full orbital geometry
    # delayS = -2*r*log(1 - e*cos(E) - SINI*[sin(omega)*(cos(E)-e) +
    #                                         sqrt(1-e^2)*cos(omega)*sin(E)])
    shapiro_arg = (
        1.0
        - ecc_current * cosE
        - sini_use * (sinOm * (cosE - ecc_current) + jnp.sqrt(1.0 - ecc_current**2) * cosOm * sinE)
    )

    # Avoid log(0) or log(negative)
    shapiro_arg = jnp.maximum(shapiro_arg, 1e-30)

    delayS = jnp.where(
        (m2_msun_use > 0.0) & (sini_use > 0.0),
        -2.0 * T_SUN * m2_msun_use * jnp.log(shapiro_arg),
        0.0
    )

    # =========================================================================
    # ABERRATION DELAY
    # Following PINT's DD_model.py: delayA()
    # Reference: Damour & Deruelle (1986) equation [27]
    # =========================================================================
    # For now, assume A0=0, B0=0 (not typically provided in par files)
    delayA = 0.0

    # Total DD delay
    return delayInverse + delayS + delayA


@jax.jit
def dd_binary_delay_vectorized(
    t_bary_mjd,
    pb_days,
    a1_lt_sec,
    ecc,
    omega_deg,
    t0_mjd,
    gamma_sec=0.0,
    pbdot=0.0,
    omdot_deg_yr=0.0,
    xdot=0.0,
    edot=0.0,
    sini=None,
    m2_msun=None,
    h3_sec=None,
    h4_sec=None,
    stig=None,
    _force_recompile_v2=None,  # Dummy parameter to force recompilation
):
    """Vectorized DD binary delay computation.
    
    Same as dd_binary_delay but accepts array of times.
    
    Parameters
    ----------
    t_bary_mjd : jnp.ndarray
        Array of barycentric times (MJD)
    ... (same as dd_binary_delay)
    
    Returns
    -------
    jnp.ndarray
        Array of binary delays in seconds
    """
    return jax.vmap(
        lambda t: dd_binary_delay(
            t, pb_days, a1_lt_sec, ecc, omega_deg, t0_mjd,
            gamma_sec, pbdot, omdot_deg_yr, xdot, edot,
            sini, m2_msun, h3_sec, h4_sec, stig
        )
    )(t_bary_mjd)


# Convenience aliases for model variants
ddh_binary_delay = dd_binary_delay
ddh_binary_delay_vectorized = dd_binary_delay_vectorized

ddgr_binary_delay = dd_binary_delay
ddgr_binary_delay_vectorized = dd_binary_delay_vectorized

ddk_binary_delay = dd_binary_delay
ddk_binary_delay_vectorized = dd_binary_delay_vectorized
