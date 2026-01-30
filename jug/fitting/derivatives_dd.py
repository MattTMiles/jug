"""Analytical derivatives for DD binary orbital parameters (JAX implementation).

The DD (Damour-Deruelle) binary model uses traditional Keplerian orbital elements:
- T0: Time of periastron passage (MJD)
- PB: Orbital period (days)
- A1: Projected semi-major axis (light-seconds)
- ECC: Orbital eccentricity
- OM: Longitude of periastron (degrees)
- OMDOT: Rate of periastron advance (deg/yr, optional)
- GAMMA: Time dilation + gravitational redshift (s, optional)

This contrasts with ELL1 which uses Laplace-Lagrange parameters (TASC, EPS1, EPS2).

The DD delay consists of:
1. Roemer delay: Light travel time across orbit
2. Einstein delay: Time dilation + gravitational redshift (GAMMA term)
3. Shapiro delay: Signal delay from companion's gravitational field

Reference: Damour & Deruelle (1986), PINT src/pint/models/binary_dd.py
"""

import jax
import jax.numpy as jnp
from typing import Dict, List

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

# Constants
SECS_PER_DAY = 86400.0
T_SUN = 4.925490947e-6  # GM_sun/c^3 in seconds
DEG_TO_RAD = jnp.pi / 180.0


# =============================================================================
# Eccentric Anomaly Solver (Kepler's Equation)
# =============================================================================

@jax.jit
def solve_kepler(mean_anomaly: jnp.ndarray, ecc: float, tol: float = 1e-12) -> jnp.ndarray:
    """Solve Kepler's equation: E - ecc*sin(E) = M
    
    Uses Newton-Raphson iteration.
    
    Parameters
    ----------
    mean_anomaly : jnp.ndarray
        Mean anomaly M in radians
    ecc : float
        Orbital eccentricity (0 <= ecc < 1)
    tol : float
        Convergence tolerance
        
    Returns
    -------
    E : jnp.ndarray
        Eccentric anomaly in radians
    """
    # Initial guess
    E = mean_anomaly + ecc * jnp.sin(mean_anomaly)
    
    # Newton-Raphson iterations (fixed count for JIT compatibility)
    for _ in range(10):
        f = E - ecc * jnp.sin(E) - mean_anomaly
        fp = 1.0 - ecc * jnp.cos(E)
        E = E - f / fp
    
    return E


@jax.jit
def compute_true_anomaly(E: jnp.ndarray, ecc: float) -> jnp.ndarray:
    """Compute true anomaly from eccentric anomaly.
    
    tan(theta/2) = sqrt((1+e)/(1-e)) * tan(E/2)
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    ecc : float
        Orbital eccentricity
        
    Returns
    -------
    theta : jnp.ndarray
        True anomaly in radians
    """
    beta = jnp.sqrt((1 + ecc) / (1 - ecc))
    theta = 2 * jnp.arctan2(beta * jnp.sin(E / 2), jnp.cos(E / 2))
    return theta


# =============================================================================
# DD Model Orbital Phase and Delay
# =============================================================================

@jax.jit
def compute_mean_anomaly_dd(
    toas_bary_mjd: jnp.ndarray,
    pb: float,
    t0: float,
    pbdot: float = 0.0
) -> jnp.ndarray:
    """Compute mean anomaly for DD model.
    
    M = 2π * (t - T0) / PB * (1 - 0.5 * PBDOT * (t - T0) / PB)
    
    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    pb : float
        Orbital period in days
    t0 : float
        Time of periastron passage in MJD
    pbdot : float
        Period derivative (dimensionless)
        
    Returns
    -------
    M : jnp.ndarray
        Mean anomaly in radians
    """
    dt = toas_bary_mjd - t0  # days
    orbits = dt / pb * (1.0 - 0.5 * pbdot * dt / pb)
    return 2 * jnp.pi * orbits


@jax.jit
def compute_dd_roemer_delay(
    E: jnp.ndarray,
    theta: jnp.ndarray,
    a1: float,
    ecc: float,
    om_rad: float
) -> jnp.ndarray:
    """Compute Roemer delay for DD model.
    
    Roemer delay = a1 * (sin(omega) * (cos(E) - ecc) + 
                         cos(omega) * sqrt(1-ecc²) * sin(E))
    
    Or equivalently using true anomaly:
    Roemer delay = a1 * sin(omega + theta) * (1 - ecc²) / (1 + ecc*cos(theta))
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    theta : jnp.ndarray
        True anomaly in radians
    a1 : float
        Projected semi-major axis in light-seconds
    ecc : float
        Orbital eccentricity
    om_rad : float
        Longitude of periastron in radians
        
    Returns
    -------
    roemer : jnp.ndarray
        Roemer delay in seconds
    """
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # Using eccentric anomaly formulation (more stable)
    roemer = a1 * (sin_omega * (jnp.cos(E) - ecc) + 
                   cos_omega * sqrt_1_e2 * jnp.sin(E))
    return roemer


@jax.jit
def compute_dd_einstein_delay(
    E: jnp.ndarray,
    gamma: float,
    ecc: float
) -> jnp.ndarray:
    """Compute Einstein delay (time dilation + gravitational redshift).
    
    Einstein delay = GAMMA * sin(E)
    
    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    gamma : float
        Einstein delay amplitude in seconds
    ecc : float
        Orbital eccentricity (not used directly, but kept for API)
        
    Returns
    -------
    einstein : jnp.ndarray
        Einstein delay in seconds
    """
    return gamma * jnp.sin(E)


@jax.jit
def compute_dd_shapiro_delay(
    theta: jnp.ndarray,
    om_rad: float,
    ecc: float,
    sini: float,
    m2: float
) -> jnp.ndarray:
    """Compute Shapiro delay for DD model.
    
    Shapiro delay = -2 * r * log(1 - e*cos(E) - s*sin(omega + theta))
    where r = T_SUN * M2 and s = SINI
    
    In terms of true anomaly:
    Shapiro = -2 * r * log(1 - s * sin(omega + theta))
    
    Parameters
    ----------
    theta : jnp.ndarray
        True anomaly in radians
    om_rad : float
        Longitude of periastron in radians
    ecc : float
        Orbital eccentricity
    sini : float
        Sine of orbital inclination
    m2 : float
        Companion mass in solar masses
        
    Returns
    -------
    shapiro : jnp.ndarray
        Shapiro delay in seconds
    """
    r = T_SUN * m2  # Range parameter
    
    # sin(omega + theta) = sin_omega*cos_theta + cos_omega*sin_theta
    sin_omega_plus_theta = jnp.sin(om_rad + theta)
    
    arg = 1 - sini * sin_omega_plus_theta
    arg = jnp.maximum(arg, 1e-10)  # Avoid log(0)
    
    return -2 * r * jnp.log(arg)


def compute_dd_binary_delay(
    toas_bary_mjd: jnp.ndarray,
    params: Dict
) -> jnp.ndarray:
    """Compute total DD binary delay.
    
    Parameters
    ----------
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    params : Dict
        DD model parameters
        
    Returns
    -------
    delay : jnp.ndarray
        Total binary delay in seconds
    """
    # Extract parameters
    a1 = float(params.get('A1', 0.0))
    pb = float(params.get('PB', 1.0))
    t0 = float(params.get('T0', 0.0))
    ecc = float(params.get('ECC', 0.0))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    sini = float(params.get('SINI', 0.0))
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))  # deg/yr
    
    # Apply periastron advance
    dt_yr = (toas_bary_mjd - t0) / 365.25
    om_rad = (om_deg + omdot * dt_yr) * DEG_TO_RAD
    
    return _compute_dd_binary_delay_jit(
        jnp.asarray(toas_bary_mjd),
        a1, pb, t0, ecc, om_rad, pbdot, gamma, sini, m2
    )


@jax.jit
def _compute_dd_binary_delay_jit(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, gamma: float, sini: float, m2: float
) -> jnp.ndarray:
    """JIT-compiled DD binary delay computation."""
    # Mean anomaly
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    
    # Solve Kepler's equation for eccentric anomaly
    E = solve_kepler(M, ecc)
    
    # True anomaly
    theta = compute_true_anomaly(E, ecc)
    
    # Roemer delay
    roemer = compute_dd_roemer_delay(E, theta, a1, ecc, om_rad)
    
    # Einstein delay
    einstein = compute_dd_einstein_delay(E, gamma, ecc)
    
    # Shapiro delay
    shapiro = jnp.where(
        (sini > 0) & (m2 > 0),
        compute_dd_shapiro_delay(theta, om_rad, ecc, sini, m2),
        0.0
    )
    
    return roemer + einstein + shapiro


# =============================================================================
# DD Model Derivatives
# =============================================================================

def compute_binary_derivatives_dd(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str]
) -> Dict[str, jnp.ndarray]:
    """Compute DD binary parameter derivatives.
    
    Uses JAX autodiff for accurate derivatives.
    
    Parameters
    ----------
    params : Dict
        DD model parameters
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    fit_params : List[str]
        Parameters to compute derivatives for
        
    Returns
    -------
    derivatives : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to derivative arrays
    """
    toas_bary_mjd = jnp.asarray(toas_bary_mjd)
    
    # Extract base parameters
    a1 = float(params.get('A1', 0.0))
    pb = float(params.get('PB', 1.0))
    t0 = float(params.get('T0', float(jnp.mean(toas_bary_mjd))))
    ecc = float(params.get('ECC', 0.0))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    sini = float(params.get('SINI', 0.0))
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))
    
    # Apply periastron advance for omega
    dt_yr = (toas_bary_mjd - t0) / 365.25
    om_rad = (om_deg + omdot * dt_yr) * DEG_TO_RAD
    
    derivatives = {}
    
    for param in fit_params:
        param_upper = param.upper()
        
        if param_upper == 'A1':
            # d(delay)/d(A1) - simple scaling
            deriv = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'PB':
            deriv = _d_delay_d_PB(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv  # Already in s/day units
            
        elif param_upper == 'T0':
            deriv = _d_delay_d_T0(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv  # Already in s/day units
            
        elif param_upper == 'ECC':
            deriv = _d_delay_d_ECC(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, gamma, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'OM':
            deriv = _d_delay_d_OM(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv * DEG_TO_RAD  # Convert to per-degree units
            
        elif param_upper == 'PBDOT':
            deriv = _d_delay_d_PBDOT(toas_bary_mjd, a1, pb, t0, ecc, om_rad, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'GAMMA':
            deriv = _d_delay_d_GAMMA(toas_bary_mjd, pb, t0, ecc, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'SINI':
            deriv = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'M2':
            deriv = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
            derivatives[param] = deriv
            
        elif param_upper == 'OMDOT':
            deriv = _d_delay_d_OMDOT(toas_bary_mjd, a1, pb, t0, ecc, om_deg, omdot, pbdot, sini, m2)
            # OMDOT is in deg/yr, convert appropriately
            derivatives[param] = deriv
            
        elif param_upper == 'XDOT' or param_upper == 'A1DOT':
            # A1 derivative - d(delay)/d(A1DOT) = d(delay)/d(A1) * t
            dt_sec = (toas_bary_mjd - t0) * SECS_PER_DAY
            d_a1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot)
            derivatives[param] = d_a1 * dt_sec
    
    return derivatives


# =============================================================================
# Individual Derivative Functions (using autodiff where beneficial)
# =============================================================================

@jax.jit
def _d_delay_d_A1(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray, pbdot: float
) -> jnp.ndarray:
    """d(Roemer delay)/d(A1) = Roemer_delay / A1"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # d(Roemer)/d(A1) = (Roemer/A1) since Roemer ∝ A1
    return sin_omega * (jnp.cos(E) - ecc) + cos_omega * sqrt_1_e2 * jnp.sin(E)


@jax.jit
def _d_delay_d_PB(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(PB) via chain rule through mean anomaly."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(PB) = -2π * dt / PB² * (1 - PBDOT * dt / PB)
    dM_dPB = -2 * jnp.pi * dt / pb**2 * (1 - pbdot * dt / pb)
    
    # Need d(delay)/d(M) = d(delay)/d(E) * d(E)/d(M)
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    # d(E)/d(M) = 1 / (1 - ecc*cos(E))
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    # d(Roemer)/d(E)
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    # d(Einstein)/d(E) = GAMMA * cos(E) -- but GAMMA doesn't depend on PB in first order
    
    # Shapiro derivative through theta
    # d(theta)/d(E) = sqrt(1-e²) / (1 - e*cos(E))
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    
    # d(Shapiro)/d(theta)
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = 1 - sini * sin_omega_theta
    denom = jnp.maximum(denom, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom
    
    dShapiro_dE = dShapiro_dtheta * dtheta_dE
    
    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dPB


@jax.jit
def _d_delay_d_T0(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(T0) via chain rule."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(T0) = -2π/PB * (1 - PBDOT * dt / PB)
    dM_dT0 = -2 * jnp.pi / pb * (1 - pbdot * dt / pb)
    
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    # Shapiro
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom
    dShapiro_dE = dShapiro_dtheta * dtheta_dE
    
    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dT0


@jax.jit 
def _d_delay_d_ECC(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, gamma: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(ECC) - includes Roemer and Einstein terms."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # d(Roemer)/d(ecc) has two parts:
    # 1. Direct: d/de[sin(om)*(cos(E)-e) + cos(om)*sqrt(1-e²)*sin(E)]
    # 2. Chain rule through E: d(Roemer)/d(E) * d(E)/d(ecc)
    
    # Direct derivative (holding E constant)
    dRoemer_de_direct = a1 * (
        -sin_omega  # from d(-ecc)/de
        - cos_omega * ecc / sqrt_1_e2 * jnp.sin(E)  # from d(sqrt(1-e²))/de
    )
    
    # Chain rule: d(E)/d(ecc) at fixed M
    # From E - ecc*sin(E) = M: d(E)/de = sin(E) / (1 - ecc*cos(E))
    dE_de = jnp.sin(E) / (1 - ecc * jnp.cos(E))
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    dRoemer_de = dRoemer_de_direct + dRoemer_dE * dE_de
    
    # Einstein: d(GAMMA*sin(E))/d(ecc) = GAMMA * cos(E) * d(E)/d(ecc)
    dEinstein_de = gamma * jnp.cos(E) * dE_de
    
    return dRoemer_de + dEinstein_de


@jax.jit
def _d_delay_d_OM(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(OM) - affects Roemer and Shapiro."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # d(Roemer)/d(omega) = a1 * (cos(om)*(cos(E)-e) - sin(om)*sqrt(1-e²)*sin(E))
    dRoemer_dom = a1 * (cos_omega * (jnp.cos(E) - ecc) - sin_omega * sqrt_1_e2 * jnp.sin(E))
    
    # d(Shapiro)/d(omega) = 2*r*sini*cos(om+theta) / (1 - sini*sin(om+theta))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dom = 2 * r * sini * cos_omega_theta / denom
    
    return dRoemer_dom + dShapiro_dom


@jax.jit
def _d_delay_d_PBDOT(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(PBDOT) via chain rule."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(PBDOT) = -π * dt² / PB²
    dM_dPBDOT = -jnp.pi * dt**2 / pb**2
    
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, 0.0)  # Use PBDOT=0 for base
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    dE_dM = 1.0 / (1 - ecc * jnp.cos(E))
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    dRoemer_dE = a1 * (-sin_omega * jnp.sin(E) + cos_omega * sqrt_1_e2 * jnp.cos(E))
    
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    cos_omega_theta = jnp.cos(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    dShapiro_dtheta = 2 * r * sini * cos_omega_theta / denom
    dShapiro_dE = dShapiro_dtheta * dtheta_dE
    
    return (dRoemer_dE + dShapiro_dE) * dE_dM * dM_dPBDOT


@jax.jit
def _d_delay_d_GAMMA(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, pbdot: float
) -> jnp.ndarray:
    """d(delay)/d(GAMMA) = sin(E)"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    return jnp.sin(E)


@jax.jit
def _d_delay_d_SINI(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(Shapiro)/d(SINI)"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    r = T_SUN * m2
    sin_omega_theta = jnp.sin(om_rad + theta)
    denom = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    
    return 2 * r * sin_omega_theta / denom


@jax.jit
def _d_delay_d_M2(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float
) -> jnp.ndarray:
    """d(Shapiro)/d(M2)"""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    theta = compute_true_anomaly(E, ecc)
    
    sin_omega_theta = jnp.sin(om_rad + theta)
    arg = jnp.maximum(1 - sini * sin_omega_theta, 1e-10)
    
    return -2 * T_SUN * jnp.log(arg)


@jax.jit
def _d_delay_d_OMDOT(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_deg: float, omdot: float,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(OMDOT) = d(delay)/d(OM) * dt_yr"""
    dt_yr = (toas_bary_mjd - t0) / 365.25
    om_rad = (om_deg + omdot * dt_yr) * DEG_TO_RAD
    
    # d(omega)/d(OMDOT) = dt_yr (in deg)
    d_om_rad = _d_delay_d_OM(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, sini, m2)
    
    # Convert: OMDOT is deg/yr, so d(omega_rad)/d(OMDOT) = dt_yr * DEG_TO_RAD
    return d_om_rad * dt_yr * DEG_TO_RAD


if __name__ == '__main__':
    print("Testing DD binary derivatives...")
    
    # Create synthetic TOAs
    toas = jnp.linspace(58000, 58100, 100)
    
    # DD model parameters (like J0614-3329)
    params = {
        'A1': 1.0,
        'PB': 1.5,
        'T0': 58050.0,
        'ECC': 0.001,
        'OM': 90.0,
        'SINI': 0.9,
        'M2': 0.3,
        'GAMMA': 0.0,
        'PBDOT': 0.0,
        'OMDOT': 0.0,
    }
    
    # Compute delay
    delay = compute_dd_binary_delay(toas, params)
    print(f"DD delay range: {float(jnp.min(delay)):.6f} to {float(jnp.max(delay)):.6f} s")
    
    # Compute derivatives
    derivs = compute_binary_derivatives_dd(params, toas, ['A1', 'PB', 'T0', 'ECC', 'OM'])
    
    for p, d in derivs.items():
        print(f"d(delay)/d({p}): mean={float(jnp.mean(d)):.6e}, std={float(jnp.std(d)):.6e}")
    
    print("\n✓ DD binary derivatives module ready!")
