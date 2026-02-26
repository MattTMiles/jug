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

import warnings

import jax
import jax.numpy as jnp
from typing import Dict, List

from jug.utils.constants import SECS_PER_DAY, SECS_PER_YEAR, T_SUN, DEG_TO_RAD, PC_TO_LIGHT_SEC

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)


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
    
    M = 2pi * (t - T0) / PB * (1 - 0.5 * PBDOT * (t - T0) / PB)
    
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
                         cos(omega) * sqrt(1-ecc^2) * sin(E))
    
    Or equivalently using true anomaly:
    Roemer delay = a1 * sin(omega + theta) * (1 - ecc^2) / (1 + ecc*cos(theta))
    
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
    E: jnp.ndarray,
    theta: jnp.ndarray,
    om_rad: float,
    ecc: float,
    sini: float,
    m2: float
) -> jnp.ndarray:
    """Compute Shapiro delay for DD model.

    Implements Damour & Deruelle (1986) equation [26]:
        Shapiro = -2 * r * log(1 - e*cos(E) - s*(sin(omega)*(cos(E)-e) + sqrt(1-e^2)*cos(omega)*sin(E)))

    where r = T_SUN * M2 and s = SINI.

    Parameters
    ----------
    E : jnp.ndarray
        Eccentric anomaly in radians
    theta : jnp.ndarray
        True anomaly in radians (unused, kept for clarity)
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

    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    cos_E = jnp.cos(E)
    sin_E = jnp.sin(E)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)

    # D&D 1986 eq. [26]: argument of log
    arg = 1 - ecc * cos_E - sini * (sin_omega * (cos_E - ecc) + sqrt_1_e2 * cos_omega * sin_E)
    arg = jnp.maximum(arg, 1e-10)  # Avoid log(0) for edge-on orbits

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
    m2 = float(params.get('M2', 0.0))
    
    # Handle SINI - can be numeric or 'KIN' (DDK convention: SINI = sin(KIN))
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        # DDK convention: SINI derived from KIN (orbital inclination)
        kin_deg = float(params.get('KIN', 0.0))
        sini = jnp.sin(jnp.deg2rad(kin_deg))
    else:
        sini = float(sini_raw)
    
    # Check for orthometric parameters if SINI/M2 not set
    if sini == 0.0 or m2 == 0.0:
        h3 = float(params.get('H3', 0.0))
        stig = float(params.get('STIG', params.get('STIGMA', 0.0)))
        h4 = float(params.get('H4', 0.0))

        if h3 != 0.0 and stig != 0.0:
            if h4 != 0.0:
                warnings.warn(
                    "Both STIG and H4 are nonzero; using H3/STIG parameterization (H4 ignored)",
                    UserWarning, stacklevel=2
                )
            # H3/STIG parameterization (DDH)
            sini = 2 * stig / (1 + stig**2)
            m2 = h3 / (stig**3 * T_SUN)
        elif h3 != 0.0 and h4 != 0.0:
            # H3/H4 parameterization (Freire & Wex 2010, PINT convention)
            h3h4_denom = h3**2 + h4**2
            sini = min(2.0 * h3 * h4 / h3h4_denom, 1.0)
            m2 = h3**4 / (h4**3 * T_SUN)
        elif h3 != 0.0 and h4 == 0.0:
            warnings.warn(
                "H3/H4 parameterization with H4=0: M2 is ill-conditioned; Shapiro delay will be zero",
                UserWarning, stacklevel=2
            )

    omdot = float(params.get('OMDOT', 0.0))  # deg/yr

    return _compute_dd_binary_delay_jit(
        jnp.asarray(toas_bary_mjd),
        a1, pb, t0, ecc, om_deg, omdot, pbdot, gamma, float(sini), m2
    )


@jax.jit
def _compute_dd_binary_delay_jit(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_deg: float, omdot_deg_yr: float,
    pbdot: float, gamma: float, sini: float, m2: float
) -> jnp.ndarray:
    """JIT-compiled DD binary delay computation."""
    # Mean anomaly
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)

    # Solve Kepler's equation for eccentric anomaly
    E = solve_kepler(M, ecc)

    # True anomaly
    theta = compute_true_anomaly(E, ecc)

    # Periastron advance: D&D 1986 eq [25]: omega = omega_0 + k*Ae
    # k = OMDOT / n (dimensionless); Ae = accumulated true anomaly
    dt = toas_bary_mjd - t0  # days
    orbits = dt / pb - 0.5 * pbdot * (dt / pb) ** 2
    norbits = jnp.floor(orbits)
    Ae = 2.0 * jnp.pi * norbits + theta  # accumulated true anomaly
    k_omdot = omdot_deg_yr * pb / (360.0 * 365.25)
    om_rad = jnp.deg2rad(om_deg) + k_omdot * Ae

    # Roemer delay
    roemer = compute_dd_roemer_delay(E, theta, a1, ecc, om_rad)

    # Einstein delay
    einstein = compute_dd_einstein_delay(E, gamma, ecc)

    # Shapiro delay
    shapiro = jnp.where(
        (sini > 0) & (m2 > 0),
        compute_dd_shapiro_delay(E, theta, om_rad, ecc, sini, m2),
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
    
    Uses hand-coded analytical derivatives, JIT-compiled with JAX.
    
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
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))
    
    # Handle SINI - can be numeric or 'KIN' (DDK convention: SINI = sin(KIN))
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        kin_deg = float(params.get('KIN', 0.0))
        sini = float(jnp.sin(jnp.deg2rad(kin_deg)))
    else:
        sini = float(sini_raw)
    
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
            
        elif param_upper == 'H3':
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            h4_val = float(params.get('H4', 0.0))
            if stig_val != 0.0:
                if h4_val != 0.0:
                    warnings.warn(
                        "Both STIG and H4 are nonzero; using H3/STIG parameterization (H4 ignored)",
                        UserWarning, stacklevel=2
                    )
                # DDH model: H3/STIG parameterization
                deriv = _d_delay_d_H3(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, stig_val)
            elif h4_val != 0.0:
                # H3/H4 parameterization (Freire & Wex 2010)
                deriv = _d_delay_d_H3_h3h4(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, h3_val, h4_val)
            else:
                if h3_val != 0.0:
                    warnings.warn(
                        "H3/H4 parameterization with H4=0: M2 is ill-conditioned; derivative will be zero",
                        UserWarning, stacklevel=2
                    )
                deriv = jnp.zeros_like(toas_bary_mjd)
            derivatives[param] = deriv

        elif param_upper in ('STIG', 'STIGMA'):
            # DDH model: H3/STIG parameterization
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            deriv = _d_delay_d_STIG(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, h3_val, stig_val)
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

        elif param_upper == 'EDOT':
            # Eccentricity derivative - d(delay)/d(EDOT) = d(delay)/d(ECC) * dt_sec
            # Analogous to XDOT through A1: ecc_current = ecc + edot * dt_sec
            dt_sec = (toas_bary_mjd - t0) * SECS_PER_DAY
            d_ecc = _d_delay_d_ECC(toas_bary_mjd, a1, pb, t0, ecc, om_rad, pbdot, gamma, sini, m2)
            derivatives[param] = d_ecc * dt_sec

        elif param_upper == 'H4':
            # Orthometric Shapiro parameter H4 (DD/DDH model, H3/H4 parameterization)
            h3 = float(params.get('H3', 0.0))
            h4 = float(params.get('H4', 0.0))
            deriv = _d_delay_d_H4(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, h3, h4)
            derivatives[param] = deriv

    return derivatives


# =============================================================================
# Individual Derivative Functions (analytical, JIT-compiled)
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
    
    # d(Roemer)/d(A1) = (Roemer/A1) since Roemer ~ A1
    return sin_omega * (jnp.cos(E) - ecc) + cos_omega * sqrt_1_e2 * jnp.sin(E)


@jax.jit
def _d_delay_d_PB(
    toas_bary_mjd: jnp.ndarray,
    a1: float, pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, sini: float, m2: float
) -> jnp.ndarray:
    """d(delay)/d(PB) via chain rule through mean anomaly."""
    dt = toas_bary_mjd - t0
    
    # d(M)/d(PB) = -2pi * dt / PB^2 * (1 - PBDOT * dt / PB)
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
    # d(theta)/d(E) = sqrt(1-e^2) / (1 - e*cos(E))
    dtheta_dE = sqrt_1_e2 / (1 - ecc * jnp.cos(E))
    
    # d(Shapiro)/d(theta)
    # DD Shapiro delay: Deltat_S = -2r ln(1 - s sin(omega+theta))
    # where r = T_Sun M2, s = sin(i).
    # d(Deltat_S)/dtheta = 2r s cos(omega+theta) / [1 - s sin(omega+theta)]
    #
    # NOTE: This is the DD analogue of the ELL1 cos(Phi) factor.
    # PINT and Tempo2 omit cos(omega+theta) in the equivalent expression.
    # PINT's ELL1H model (d_delayS_H3_STIGMA_exact_d_Phi)
    # includes cos(Phi), while ELL1 model does not.
    #
    # Wolfram Alpha: d/dx [-2*a*ln(1 - b*sin(x))]  ->  2*a*b*cos(x)/(1-b*sin(x))
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
    
    # d(M)/d(T0) = -2pi/PB * (1 - PBDOT * dt / PB)
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
    """d(delay)/d(ECC) - includes Roemer, Einstein, and Shapiro terms."""
    M = compute_mean_anomaly_dd(toas_bary_mjd, pb, t0, pbdot)
    E = solve_kepler(M, ecc)
    sinE = jnp.sin(E)
    cosE = jnp.cos(E)
    
    sin_omega = jnp.sin(om_rad)
    cos_omega = jnp.cos(om_rad)
    sqrt_1_e2 = jnp.sqrt(1 - ecc**2)
    
    # Chain rule: d(E)/d(ecc) at fixed M
    # From E - ecc*sin(E) = M: d(E)/de = sin(E) / (1 - ecc*cos(E))
    oneMecccosE = 1 - ecc * cosE
    dE_de = sinE / oneMecccosE
    
    # --- Roemer term ---
    # d(Roemer)/d(ecc) has two parts:
    # 1. Direct: d/de[sin(om)*(cos(E)-e) + cos(om)*sqrt(1-e^2)*sin(E)]
    # 2. Chain rule through E: d(Roemer)/d(E) * d(E)/d(ecc)
    dRoemer_de_direct = a1 * (
        -sin_omega  # from d(-ecc)/de
        - cos_omega * ecc / sqrt_1_e2 * sinE  # from d(sqrt(1-e^2))/de
    )
    dRoemer_dE = a1 * (-sin_omega * sinE + cos_omega * sqrt_1_e2 * cosE)
    dRoemer_de = dRoemer_de_direct + dRoemer_dE * dE_de
    
    # --- Einstein term ---
    # d(GAMMA*sin(E))/d(ecc) = GAMMA * cos(E) * d(E)/d(ecc)
    dEinstein_de = gamma * cosE * dE_de
    
    # --- Shapiro term ---
    # Shapiro delay: -2*r*ln(1 - e*cosE - s*(sin(om)*(cosE-e) + sqrt(1-e^2)*cos(om)*sinE))
    # where r = T_SUN * M2, s = SINI
    # d(Shapiro)/d(ecc) = d(Shapiro)/d(ecc)|_E + d(Shapiro)/d(E) * dE/de
    r = T_SUN * m2
    logArg = 1 - ecc * cosE - sini * (sin_omega * (cosE - ecc) + sqrt_1_e2 * cos_omega * sinE)
    logArg = jnp.maximum(logArg, 1e-10)
    
    # Direct partial (holding E constant):
    # d(logArg)/de = -cosE - sini*(-sin(om) - e*cos(om)*sinE/sqrt(1-e^2))
    #              = -cosE - sini*(-sin(om) + e/(sqrt(1-e^2))*(-cos(om)*sinE))
    dlogArg_de = -cosE - sini * (-sin_omega - ecc * cos_omega * sinE / sqrt_1_e2)
    dShapiro_de_direct = -2 * r * dlogArg_de / logArg
    
    # Chain rule through E:
    # d(logArg)/dE = e*sinE - sini*(sqrt(1-e^2)*cosE*cos(om) - sinE*sin(om))
    dlogArg_dE = ecc * sinE - sini * (sqrt_1_e2 * cosE * cos_omega - sinE * sin_omega)
    dShapiro_dE = -2 * r * dlogArg_dE / logArg
    
    dShapiro_de = dShapiro_de_direct + dShapiro_dE * dE_de
    
    return dRoemer_de + dEinstein_de + dShapiro_de


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
    
    # d(Roemer)/d(omega) = a1 * (cos(om)*(cos(E)-e) - sin(om)*sqrt(1-e^2)*sin(E))
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
    
    # d(M)/d(PBDOT) = -pi * dt^2 / PB^2
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


# =============================================================================
# DDK (Kopeikin 1995/1996) Partial Derivatives for KIN and KOM
# =============================================================================
# The DDK model applies corrections to A1 and OM based on:
#   1. K96 proper motion corrections (Kopeikin 1996)
#   2. Annual orbital parallax corrections (Kopeikin 1995)
#
# The total derivatives use the chain rule:
#   d(delay)/d(KIN) = d(delay)/d(A1_eff) * d(A1_eff)/d(KIN) 
#                   + d(delay)/d(OM_eff) * d(OM_eff)/d(KIN)
#                   + d(delay)/d(SINI_eff) * d(SINI_eff)/d(KIN)
#
# where A1_eff = A1 + delta_A1_pm + delta_A1_px
#       OM_eff = OM + delta_OM_pm + delta_OM_px  (in degrees)
#       SINI_eff = sin(KIN_eff) for DDK when SINI not explicitly set
#
# References:
#   - Kopeikin 1995: Annual orbital parallax
#   - Kopeikin 1996: Proper motion (K96) corrections
#   - PINT src/pint/models/binary_ddk.py for implementation details


@jax.jit
def _compute_ddk_correction_derivatives_KIN(
    tt0_sec: jnp.ndarray,
    a1: float,
    kin_rad: float,
    kom_rad: float,
    pmra_rad_per_sec: float,
    pmdec_rad_per_sec: float,
    delta_I0: jnp.ndarray,
    delta_J0: jnp.ndarray,
    d_ls: float,
    use_k96: bool,
    has_parallax: bool
) -> tuple:
    """
    Compute d(delta_A1)/d(KIN) and d(delta_OM)/d(KIN) for DDK corrections.
    
    The K96 proper motion corrections (Kopeikin 1996) are:
        delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * (t - T0)
        delta_A1_pm = A1 * delta_KIN_pm / tan(KIN_eff)
        delta_OM_pm = (1/sin(KIN_eff)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * (t - T0)
    
    The Kopeikin 1995 parallax corrections are:
        delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
        delta_OM_px = -(1 / sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    
    For the chain rule, we need:
        d(delta_A1_pm)/d(KIN) - involves d/d(KIN)[delta_KIN_pm / tan(KIN_eff)]
        d(delta_OM_pm)/d(KIN) - involves d/d(KIN)[1/sin(KIN_eff)]
        d(delta_A1_px)/d(KIN) - involves d/d(KIN)[1/tan(KIN)]
        d(delta_OM_px)/d(KIN) - involves d/d(KIN)[1/sin(KIN)]
    
    Returns
    -------
    d_A1_eff_d_KIN : array
        d(A1_eff)/d(KIN) in light-seconds per radian
    d_OM_eff_d_KIN : array
        d(OM_eff)/d(KIN) in radians per radian (dimensionless)
    d_SINI_eff_d_KIN : array
        d(SINI_eff)/d(KIN) in 1/radian
    """
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    sin_kin = jnp.sin(kin_rad)
    cos_kin = jnp.cos(kin_rad)
    sin2_kin = sin_kin ** 2
    
    # K96 proper motion corrections
    # delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0_sec
    pm_term = -pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom
    delta_kin_pm = jnp.where(use_k96, pm_term * tt0_sec, 0.0)
    
    # d(delta_KIN_pm)/d(KIN) = 0 (no explicit KIN dependence in the definition)
    
    # delta_A1_pm = A1 * delta_KIN_pm / tan(KIN_eff)
    # where KIN_eff = KIN + delta_KIN_pm
    # Approximate: d(KIN_eff)/d(KIN) ~= 1 (delta_KIN_pm doesn't depend on KIN)
    # d(A1 * delta_KIN / tan(KIN))/d(KIN) = A1 * delta_KIN * d(1/tan(KIN))/d(KIN)
    #                                      = A1 * delta_KIN * (-1/sin^2(KIN))
    d_A1_pm_d_KIN = jnp.where(
        use_k96,
        -a1 * delta_kin_pm / sin2_kin,
        0.0
    )
    
    # delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0_sec
    # d/d(KIN)[1/sin(KIN)] = -cos(KIN)/sin^2(KIN)
    pm_omega_term = pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom
    d_OM_pm_d_KIN = jnp.where(
        use_k96,
        -cos_kin / sin2_kin * pm_omega_term * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995 parallax corrections
    # delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
    # d/d(KIN)[A1 / tan(KIN) / d] = A1 / d * (-1/sin^2(KIN))
    parallax_a1_term = delta_I0 * sin_kom - delta_J0 * cos_kom
    d_A1_px_d_KIN = jnp.where(
        has_parallax,
        -a1 / d_ls / sin2_kin * parallax_a1_term,
        0.0
    )
    
    # delta_OM_px = -(1/sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    # d/d(KIN)[1/sin(KIN)] = -cos(KIN)/sin^2(KIN)
    parallax_om_term = delta_I0 * cos_kom + delta_J0 * sin_kom
    d_OM_px_d_KIN = jnp.where(
        has_parallax,
        cos_kin / sin2_kin / d_ls * parallax_om_term,  # Note: negative from chain rule cancels original negative
        0.0
    )
    
    # Total derivatives
    d_A1_eff_d_KIN = d_A1_pm_d_KIN + d_A1_px_d_KIN
    d_OM_eff_d_KIN = d_OM_pm_d_KIN + d_OM_px_d_KIN  # in radians/radian
    
    # SINI_eff = sin(KIN_eff) where KIN_eff ~= KIN for small corrections
    # d(sin(KIN))/d(KIN) = cos(KIN)
    d_SINI_eff_d_KIN = cos_kin
    
    return d_A1_eff_d_KIN, d_OM_eff_d_KIN, d_SINI_eff_d_KIN


@jax.jit
def _compute_ddk_correction_derivatives_KOM(
    tt0_sec: jnp.ndarray,
    a1: float,
    kin_rad: float,
    kom_rad: float,
    pmra_rad_per_sec: float,
    pmdec_rad_per_sec: float,
    delta_I0: jnp.ndarray,
    delta_J0: jnp.ndarray,
    d_ls: float,
    use_k96: bool,
    has_parallax: bool
) -> tuple:
    """
    Compute d(delta_A1)/d(KOM) and d(delta_OM)/d(KOM) for DDK corrections.
    
    K96 proper motion:
        delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0
        delta_A1_pm = A1 * delta_KIN_pm / tan(KIN)
        delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0
    
    Kopeikin 1995 parallax:
        delta_A1_px = (A1 / tan(KIN) / d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
        delta_OM_px = -(1/sin(KIN) / d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    
    Returns
    -------
    d_A1_eff_d_KOM : array
        d(A1_eff)/d(KOM) in light-seconds per radian
    d_OM_eff_d_KOM : array
        d(OM_eff)/d(KOM) in radians per radian (dimensionless)
    """
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    sin_kin = jnp.sin(kin_rad)
    tan_kin = jnp.tan(kin_rad)
    
    # Safe denominators
    sin_kin_safe = jnp.where(jnp.abs(sin_kin) < 1e-10, 1e-10, sin_kin)
    tan_kin_safe = jnp.where(jnp.abs(tan_kin) < 1e-10, 1e-10, tan_kin)
    
    # K96 proper motion: delta_KIN_pm = (-mu_RA * sin(KOM) + mu_DEC * cos(KOM)) * tt0
    # d(delta_KIN_pm)/d(KOM) = (-mu_RA * cos(KOM) - mu_DEC * sin(KOM)) * tt0
    d_delta_kin_pm_d_KOM = jnp.where(
        use_k96,
        (-pmra_rad_per_sec * cos_kom - pmdec_rad_per_sec * sin_kom) * tt0_sec,
        0.0
    )
    
    # delta_A1_pm = A1 * delta_KIN_pm / tan(KIN)
    # d(delta_A1_pm)/d(KOM) = A1 / tan(KIN) * d(delta_KIN_pm)/d(KOM)
    d_A1_pm_d_KOM = jnp.where(
        use_k96,
        a1 / tan_kin_safe * d_delta_kin_pm_d_KOM,
        0.0
    )
    
    # delta_OM_pm = (1/sin(KIN)) * (mu_RA * cos(KOM) + mu_DEC * sin(KOM)) * tt0
    # d/d(KOM)[mu_RA * cos(KOM) + mu_DEC * sin(KOM)] = -mu_RA * sin(KOM) + mu_DEC * cos(KOM)
    d_OM_pm_d_KOM = jnp.where(
        use_k96,
        (1.0 / sin_kin_safe) * (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995: delta_A1_px = (A1/tan(KIN)/d) * (delta_I0 * sin(KOM) - delta_J0 * cos(KOM))
    # d/d(KOM)[delta_I0 * sin(KOM) - delta_J0 * cos(KOM)] = delta_I0 * cos(KOM) + delta_J0 * sin(KOM)
    d_A1_px_d_KOM = jnp.where(
        has_parallax,
        a1 / tan_kin_safe / d_ls * (delta_I0 * cos_kom + delta_J0 * sin_kom),
        0.0
    )
    
    # delta_OM_px = -(1/sin(KIN)/d) * (delta_I0 * cos(KOM) + delta_J0 * sin(KOM))
    # d/d(KOM)[delta_I0 * cos(KOM) + delta_J0 * sin(KOM)] = -delta_I0 * sin(KOM) + delta_J0 * cos(KOM)
    d_OM_px_d_KOM = jnp.where(
        has_parallax,
        -(1.0 / sin_kin_safe / d_ls) * (-delta_I0 * sin_kom + delta_J0 * cos_kom),
        0.0
    )
    
    # Total derivatives
    d_A1_eff_d_KOM = d_A1_pm_d_KOM + d_A1_px_d_KOM
    d_OM_eff_d_KOM = d_OM_pm_d_KOM + d_OM_px_d_KOM  # in radians/radian
    
    # SINI_eff = sin(KIN_eff) where KIN_eff = KIN + delta_KIN_pm(KOM)
    # d(SINI_eff)/d(KOM) = cos(KIN_eff) * d(delta_KIN_pm)/d(KOM)
    cos_kin = jnp.cos(kin_rad)
    d_SINI_eff_d_KOM = jnp.where(use_k96, cos_kin * d_delta_kin_pm_d_KOM, 0.0)
    
    return d_A1_eff_d_KOM, d_OM_eff_d_KOM, d_SINI_eff_d_KOM


def compute_binary_derivatives_ddk(
    params: Dict,
    toas_bary_mjd: jnp.ndarray,
    fit_params: List[str],
    obs_pos_ls: jnp.ndarray = None,
) -> Dict[str, jnp.ndarray]:
    """
    Compute DDK binary parameter derivatives including KIN and KOM.
    
    DDK extends DD with Kopeikin corrections. For standard DD parameters,
    we use the DD derivatives evaluated at the effective A1/OM values.
    For KIN and KOM, we use the chain rule through the Kopeikin corrections.
    
    Parameters
    ----------
    params : Dict
        DDK model parameters (must include KIN, KOM, and optionally PX, PMRA, PMDEC)
    toas_bary_mjd : jnp.ndarray
        Barycentric TOA times in MJD
    fit_params : List[str]
        Parameters to compute derivatives for
    obs_pos_ls : jnp.ndarray, optional
        Observer position in light-seconds relative to SSB, shape (N, 3).
        Required for Kopeikin 1995 parallax corrections.
        
    Returns
    -------
    derivatives : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to derivative arrays
    """
    toas_bary_mjd = jnp.asarray(toas_bary_mjd)
    n_toas = len(toas_bary_mjd)
    
    # Extract base DD parameters
    a1 = float(params.get('A1', 0.0))
    pb = float(params.get('PB', 1.0))
    t0 = float(params.get('T0', float(jnp.mean(toas_bary_mjd))))
    ecc = float(params.get('ECC', 0.0))
    om_deg = float(params.get('OM', 0.0))
    pbdot = float(params.get('PBDOT', 0.0))
    gamma = float(params.get('GAMMA', 0.0))
    m2 = float(params.get('M2', 0.0))
    omdot = float(params.get('OMDOT', 0.0))
    
    # DDK-specific parameters
    kin_deg = float(params.get('KIN', 0.0))
    
    kom_deg = float(params.get('KOM', 0.0))
    kin_rad = jnp.deg2rad(kin_deg)
    kom_rad = jnp.deg2rad(kom_deg)
    
    px_mas = float(params.get('PX', 0.0))
    
    # Proper motion (for K96).
    # For ecliptic pulsars, use PMELONG/PMELAT (stored as _ecliptic_pm_lon/lat)
    # so that the K96 formula uses the same coordinate frame as KOM.
    MAS_PER_YR_TO_RAD_PER_SEC = (jnp.pi / 180.0 / 3600.0 / 1000.0) / SECS_PER_YEAR
    _is_ecliptic = bool(params.get('_ecliptic_coords', False))
    if _is_ecliptic:
        pmra_mas_yr = float(params.get('_ecliptic_pm_lon', 0.0))   # PMELONG
        pmdec_mas_yr = float(params.get('_ecliptic_pm_lat', 0.0))  # PMELAT
    else:
        pmra_mas_yr = float(params.get('PMRA', 0.0))
        pmdec_mas_yr = float(params.get('PMDEC', 0.0))
    pmra_rad_per_sec = pmra_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    pmdec_rad_per_sec = pmdec_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    
    # K96 flag
    k96_flag = True
    if 'K96' in params:
        k96_param = params['K96']
        if isinstance(k96_param, bool):
            k96_flag = k96_param
        elif isinstance(k96_param, str):
            k96_flag = k96_param.upper() not in ('N', 'NO', 'FALSE', '0', 'F')
        else:
            k96_flag = bool(k96_param)
    use_k96 = k96_flag and (pmra_mas_yr != 0 or pmdec_mas_yr != 0)
    
    # Check for valid parallax
    has_parallax = px_mas > 0.0 and jnp.abs(kin_deg) > 0.0
    
    # Distance in light-seconds from parallax
    px_safe = max(abs(px_mas), 1e-10)
    d_ls = 1000.0 * PC_TO_LIGHT_SEC / px_safe
    
    # Time since T0
    dt_days = toas_bary_mjd - t0
    tt0_sec = dt_days * SECS_PER_DAY
    
    # Observer position for Kopeikin projections.
    # For ecliptic pulsars, rotate ICRS obs_pos to ecliptic frame.
    if obs_pos_ls is None:
        obs_pos_ls = jnp.zeros((n_toas, 3))
    obs_pos_ls = jnp.asarray(obs_pos_ls)
    if _is_ecliptic:
        from jug.io.par_reader import OBLIQUITY_ARCSEC
        _ecl_frame = str(params.get('_ecliptic_frame', 'IERS2010')).upper()
        _obl_rad = OBLIQUITY_ARCSEC.get(_ecl_frame, OBLIQUITY_ARCSEC['IERS2010']) * float(jnp.pi) / (180.0 * 3600.0)
        _cos_obl = jnp.cos(_obl_rad)
        _sin_obl = jnp.sin(_obl_rad)
        _x = obs_pos_ls[:, 0]
        _y = obs_pos_ls[:, 1] * _cos_obl + obs_pos_ls[:, 2] * _sin_obl
        _z = -obs_pos_ls[:, 1] * _sin_obl + obs_pos_ls[:, 2] * _cos_obl
        obs_pos_ls = jnp.column_stack([_x, _y, _z])

    # Get pulsar position for K95 projections (delta_I0, delta_J0).
    # For ecliptic pulsars, use ecliptic lon/lat instead of RA/DEC so that
    # the projections are consistent with the ecliptic KOM frame.
    if _is_ecliptic:
        _ecl_lon_rad = float(jnp.pi) / 180.0 * float(params.get('_ecliptic_lon_deg', 0.0))
        _ecl_lat_rad = float(jnp.pi) / 180.0 * float(params.get('_ecliptic_lat_deg', 0.0))
        sin_ra = jnp.sin(_ecl_lon_rad)
        cos_ra = jnp.cos(_ecl_lon_rad)
        sin_dec = jnp.sin(_ecl_lat_rad)
        cos_dec = jnp.cos(_ecl_lat_rad)
    else:
        # Handle both radians (float) and sexagesimal strings
        from jug.io.par_reader import parse_ra, parse_dec
        raj_val = params.get('RAJ', 0.0)
        decj_val = params.get('DECJ', 0.0)

        if isinstance(raj_val, str) and ':' in raj_val:
            ra_rad = parse_ra(raj_val)
        else:
            ra_rad = float(raj_val)

        if isinstance(decj_val, str) and ':' in decj_val:
            dec_rad = parse_dec(decj_val)
        else:
            dec_rad = float(decj_val)

        sin_ra = jnp.sin(ra_rad)
        cos_ra = jnp.cos(ra_rad)
        sin_dec = jnp.sin(dec_rad)
        cos_dec = jnp.cos(dec_rad)
    
    # Kopeikin projection terms (per-TOA)
    x = obs_pos_ls[:, 0]
    y = obs_pos_ls[:, 1]
    z = obs_pos_ls[:, 2]
    delta_I0 = -x * sin_ra + y * cos_ra
    delta_J0 = -x * sin_dec * cos_ra - y * sin_dec * sin_ra + z * cos_dec
    
    # Compute effective parameters (matching combined.py branch_ddk)
    sin_kom = jnp.sin(kom_rad)
    cos_kom = jnp.cos(kom_rad)
    
    # K96 corrections
    delta_kin_pm = jnp.where(
        use_k96,
        (-pmra_rad_per_sec * sin_kom + pmdec_rad_per_sec * cos_kom) * tt0_sec,
        0.0
    )
    kin_eff_rad = kin_rad + delta_kin_pm
    
    tan_kin_eff = jnp.tan(kin_eff_rad)
    tan_kin_eff_safe = jnp.where(jnp.abs(tan_kin_eff) < 1e-10, 1e-10, tan_kin_eff)
    sin_kin_eff = jnp.sin(kin_eff_rad)
    sin_kin_eff_safe = jnp.where(jnp.abs(sin_kin_eff) < 1e-10, 1e-10, sin_kin_eff)
    
    delta_a1_pm = jnp.where(use_k96, a1 * delta_kin_pm / tan_kin_eff_safe, 0.0)
    delta_omega_pm_rad = jnp.where(
        use_k96,
        (1.0 / sin_kin_eff_safe) * (pmra_rad_per_sec * cos_kom + pmdec_rad_per_sec * sin_kom) * tt0_sec,
        0.0
    )
    
    # Kopeikin 1995 parallax corrections
    delta_a1_px = jnp.where(
        has_parallax,
        (a1 / tan_kin_eff_safe / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom),
        0.0
    )
    delta_omega_px_rad = jnp.where(
        has_parallax,
        -(1.0 / sin_kin_eff_safe / d_ls) * (delta_I0 * cos_kom + delta_J0 * sin_kom),
        0.0
    )
    
    # Effective parameters
    a1_eff = a1 + delta_a1_pm + delta_a1_px
    om_eff_deg = om_deg + jnp.rad2deg(delta_omega_pm_rad) + jnp.rad2deg(delta_omega_px_rad)
    
    # For SINI: use sin(KIN_eff) if SINI not explicitly set or if SINI='KIN'
    sini_raw = params.get('SINI', 0.0)
    if isinstance(sini_raw, str) and sini_raw.upper() == 'KIN':
        # DDK convention: SINI derived from KIN
        sini_explicit = 0.0  # Treat as not explicitly set
    else:
        sini_explicit = float(sini_raw)
    sini_eff = jnp.where(
        (sini_explicit == 0.0) & (jnp.abs(kin_deg) > 0.0),
        jnp.sin(kin_eff_rad),
        sini_explicit
    )
    
    # Get the base DD derivatives evaluated at effective parameters
    # We need to handle the time-varying omega
    dt_yr = (toas_bary_mjd - t0) / 365.25
    om_rad_eff = (om_eff_deg + omdot * dt_yr) * DEG_TO_RAD
    
    derivatives = {}
    
    # Check which parameters need KIN/KOM-specific handling
    fit_params_upper = [p.upper() for p in fit_params]
    needs_kin = 'KIN' in fit_params_upper
    needs_kom = 'KOM' in fit_params_upper
    
    # First, handle standard DD parameters using effective values
    dd_params = [p for p in fit_params if p.upper() not in ('KIN', 'KOM')]
    
    for param in dd_params:
        param_upper = param.upper()
        
        if param_upper == 'A1':
            # For A1, the effective derivative needs adjustment
            # d(delay)/d(A1) through effective A1
            # d(A1_eff)/d(A1) = 1 + d(delta_A1)/d(A1) where delta_A1 terms ~ A1
            # delta_A1_pm = A1 * delta_kin_pm / tan(KIN) -> d/dA1 = delta_kin_pm / tan(KIN)
            # delta_A1_px = A1 / tan(KIN) / d * (...) -> d/dA1 = 1/tan(KIN)/d * (...)
            
            # Get base derivative for A1
            deriv = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot)
            
            # Adjustment factor for effective A1 dependence on A1
            d_A1_eff_d_A1 = 1.0
            if use_k96:
                d_A1_eff_d_A1 = d_A1_eff_d_A1 + delta_kin_pm / tan_kin_eff_safe
            if has_parallax:
                d_A1_eff_d_A1 = d_A1_eff_d_A1 + (1.0 / tan_kin_eff_safe / d_ls) * (delta_I0 * sin_kom - delta_J0 * cos_kom)
            
            derivatives[param] = deriv * d_A1_eff_d_A1
            
        elif param_upper == 'PB':
            deriv = _d_delay_d_PB(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'T0':
            deriv = _d_delay_d_T0(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'ECC':
            deriv = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, gamma, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'OM':
            # d(delay)/d(OM) - OM_eff = OM + corrections, so d(OM_eff)/d(OM) = 1
            deriv = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv * DEG_TO_RAD
            
        elif param_upper == 'PBDOT':
            deriv = _d_delay_d_PBDOT(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'GAMMA':
            deriv = _d_delay_d_GAMMA(toas_bary_mjd, pb, t0, ecc, pbdot)
            derivatives[param] = deriv
            
        elif param_upper == 'SINI':
            deriv = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'M2':
            deriv = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, sini_eff)
            derivatives[param] = deriv
            
        elif param_upper == 'OMDOT':
            deriv = _d_delay_d_OMDOT(toas_bary_mjd, a1_eff, pb, t0, ecc, float(jnp.mean(om_eff_deg)), omdot, pbdot, sini_eff, m2)
            derivatives[param] = deriv
            
        elif param_upper == 'XDOT' or param_upper == 'A1DOT':
            dt_sec = (toas_bary_mjd - t0) * SECS_PER_DAY
            d_a1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot)
            derivatives[param] = d_a1 * dt_sec

        elif param_upper == 'EDOT':
            dt_sec = (toas_bary_mjd - t0) * SECS_PER_DAY
            d_ecc = _d_delay_d_ECC(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, gamma, sini_eff, m2)
            derivatives[param] = d_ecc * dt_sec

        elif param_upper == 'H3':
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            h4_val = float(params.get('H4', 0.0))
            if stig_val != 0.0:
                if h4_val != 0.0:
                    warnings.warn(
                        "Both STIG and H4 are nonzero; using H3/STIG parameterization (H4 ignored)",
                        UserWarning, stacklevel=2
                    )
                deriv = _d_delay_d_H3(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, stig_val)
            elif h4_val != 0.0:
                deriv = _d_delay_d_H3_h3h4(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, h3_val, h4_val)
            else:
                if h3_val != 0.0:
                    warnings.warn(
                        "H3/H4 parameterization with H4=0: M2 is ill-conditioned; derivative will be zero",
                        UserWarning, stacklevel=2
                    )
                deriv = jnp.zeros_like(toas_bary_mjd)
            derivatives[param] = deriv

        elif param_upper in ('STIG', 'STIGMA'):
            h3_val = float(params.get('H3', 0.0))
            stig_val = float(params.get('STIG', params.get('STIGMA', 0.0)))
            deriv = _d_delay_d_STIG(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, h3_val, stig_val)
            derivatives[param] = deriv

        elif param_upper == 'H4':
            h3_val = float(params.get('H3', 0.0))
            h4_val = float(params.get('H4', 0.0))
            deriv = _d_delay_d_H4(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, h3_val, h4_val)
            derivatives[param] = deriv

    # Now handle KIN and KOM using chain rule
    if needs_kin:
        # d(delay)/d(KIN) = d(delay)/d(A1_eff) * d(A1_eff)/d(KIN)
        #                 + d(delay)/d(OM_eff) * d(OM_eff)/d(KIN)
        #                 + d(delay)/d(SINI_eff) * d(SINI_eff)/d(KIN)
        
        # Compute correction derivatives
        d_A1_eff_d_KIN, d_OM_eff_d_KIN_rad, d_SINI_eff_d_KIN = _compute_ddk_correction_derivatives_KIN(
            tt0_sec, a1, float(kin_rad), float(kom_rad),
            pmra_rad_per_sec, pmdec_rad_per_sec,
            delta_I0, delta_J0, d_ls,
            use_k96, has_parallax
        )
        
        # Get base derivatives
        d_delay_d_A1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot)
        d_delay_d_OM = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
        d_delay_d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
        
        # Chain rule (note: d_OM_eff_d_KIN_rad is in radians/radian, d_delay_d_OM is in sec/radian)
        d_delay_d_KIN_rad = (
            d_delay_d_A1 * d_A1_eff_d_KIN +
            d_delay_d_OM * d_OM_eff_d_KIN_rad +
            d_delay_d_SINI * d_SINI_eff_d_KIN
        )
        
        # Convert from per-radian to per-degree (KIN is in degrees)
        derivatives['KIN'] = d_delay_d_KIN_rad * DEG_TO_RAD
    
    if needs_kom:
        # d(delay)/d(KOM) = d(delay)/d(A1_eff) * d(A1_eff)/d(KOM)
        #                 + d(delay)/d(OM_eff) * d(OM_eff)/d(KOM)
        
        d_A1_eff_d_KOM, d_OM_eff_d_KOM_rad, _ = _compute_ddk_correction_derivatives_KOM(
            tt0_sec, a1, float(kin_rad), float(kom_rad),
            pmra_rad_per_sec, pmdec_rad_per_sec,
            delta_I0, delta_J0, d_ls,
            use_k96, has_parallax
        )
        
        d_delay_d_A1 = _d_delay_d_A1(toas_bary_mjd, pb, t0, ecc, om_rad_eff, pbdot)
        d_delay_d_OM = _d_delay_d_OM(toas_bary_mjd, a1_eff, pb, t0, ecc, om_rad_eff, pbdot, sini_eff, m2)
        
        d_delay_d_KOM_rad = (
            d_delay_d_A1 * d_A1_eff_d_KOM +
            d_delay_d_OM * d_OM_eff_d_KOM_rad
        )
        
        # Convert from per-radian to per-degree
        derivatives['KOM'] = d_delay_d_KOM_rad * DEG_TO_RAD
    
    return derivatives


# =============================================================================
# H3/STIG Orthometric Shapiro Delay Derivatives
# =============================================================================
# DDH uses orthometric parameterization instead of SINI/M2:
#   SINI = 2 * STIG / (1 + STIG^2)
#   M2 = H3 / (STIG^3 * T_SUN)
#
# We use chain rule:
#   d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
#   d(delay)/d(STIG) = d(delay)/d(M2) * d(M2)/d(STIG) + d(delay)/d(SINI) * d(SINI)/d(STIG)


@jax.jit
def _d_delay_d_H3(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, stig: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H3) for DDH orthometric parameterization.
    
    From M2 = H3 / (STIG^3 * T_SUN):
        d(M2)/d(H3) = 1 / (STIG^3 * T_SUN)
    
    So:
        d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
                       = d_delay_d_M2 / (STIG^3 * T_SUN)
    """
    # Compute SINI and M2 from H3/STIG for the Shapiro delay calculation
    sini = 2 * stig / (1 + stig**2)
    
    # d(delay)/d(M2) = -2 * T_SUN * log(1 - sini * sin(omega + theta))
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    
    # d(M2)/d(H3) = 1 / (STIG^3 * T_SUN)
    dM2_dH3 = 1.0 / (stig**3 * T_SUN)
    
    return d_M2 * dM2_dH3


@jax.jit
def _d_delay_d_STIG(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, stig: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(STIG) for DDH orthometric parameterization.
    
    From:
        SINI = 2 * STIG / (1 + STIG^2)
        M2 = H3 / (STIG^3 * T_SUN)
    
    Derivatives:
        d(SINI)/d(STIG) = 2 * (1 - STIG^2) / (1 + STIG^2)^2
        d(M2)/d(STIG) = -3 * H3 / (STIG^4 * T_SUN) = -3 * M2 / STIG
    
    Chain rule:
        d(delay)/d(STIG) = d(delay)/d(M2) * d(M2)/d(STIG) + d(delay)/d(SINI) * d(SINI)/d(STIG)
    """
    # Compute derived quantities
    stig2 = stig**2
    sini = 2 * stig / (1 + stig2)
    m2 = h3 / (stig**3 * T_SUN)
    
    # Get individual derivatives
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)
    
    # Compute Jacobian terms
    dM2_dSTIG = -3 * m2 / stig  # = -3 * H3 / (STIG^4 * T_SUN)
    dSINI_dSTIG = 2 * (1 - stig2) / (1 + stig2)**2
    
    return d_M2 * dM2_dSTIG + d_SINI * dSINI_dSTIG


@jax.jit
def _d_delay_d_H3_h3h4(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, h4: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H3) for H3/H4 orthometric parameterization.

    Freire & Wex (2010), PINT/Tempo2 convention:
        SINI = 2*H3*H4 / (H3^2 + H4^2)
        M2   = H3^4 / (H4^3 * T_SUN)

    Derivatives:
        d(SINI)/d(H3) = 2*H4*(H4^2 - H3^2) / (H3^2 + H4^2)^2
        d(M2)/d(H3)   = 4*H3^3 / (H4^3 * T_SUN) = 4*M2/H3

    Chain rule:
        d(delay)/d(H3) = d(delay)/d(M2) * d(M2)/d(H3)
                        + d(delay)/d(SINI) * d(SINI)/d(H3)
    """
    h4_safe = jnp.maximum(jnp.abs(h4), 1e-30)
    h3h4_denom = jnp.maximum(h3**2 + h4**2, 1e-60)
    sini = jnp.clip(2.0 * h3 * h4 / h3h4_denom, 0.0, 1.0)
    m2 = h3**4 / (h4_safe**3 * T_SUN)

    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)

    dM2_dH3 = 4.0 * h3**3 / (h4_safe**3 * T_SUN)
    dSINI_dH3 = 2.0 * h4 * (h4**2 - h3**2) / h3h4_denom**2

    return d_M2 * dM2_dH3 + d_SINI * dSINI_dH3


@jax.jit
def _d_delay_d_H4(
    toas_bary_mjd: jnp.ndarray,
    pb: float, t0: float, ecc: float, om_rad: jnp.ndarray,
    pbdot: float, h3: float, h4: float
) -> jnp.ndarray:
    """d(Shapiro delay)/d(H4) for H3/H4 orthometric parameterization.

    Freire & Wex (2010), PINT/Tempo2 convention:
        SINI = 2*H3*H4 / (H3^2 + H4^2)
        M2   = H3^4 / (H4^3 * T_SUN)

    Derivatives:
        d(SINI)/d(H4) = 2*H3*(H3^2 - H4^2) / (H3^2 + H4^2)^2
        d(M2)/d(H4)   = -3*M2/H4

    Chain rule:
        d(delay)/d(H4) = d(delay)/d(M2) * d(M2)/d(H4)
                        + d(delay)/d(SINI) * d(SINI)/d(H4)
    """
    # Compute SINI and M2 from H3/H4 (PINT convention)
    h4_safe = jnp.maximum(jnp.abs(h4), 1e-30)
    h3h4_denom = jnp.maximum(h3**2 + h4**2, 1e-60)
    sini = jnp.clip(2.0 * h3 * h4 / h3h4_denom, 0.0, 1.0)
    m2 = h3**4 / (h4_safe**3 * T_SUN)

    # Get individual derivatives of delay w.r.t. SINI and M2
    d_M2 = _d_delay_d_M2(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini)
    d_SINI = _d_delay_d_SINI(toas_bary_mjd, pb, t0, ecc, om_rad, pbdot, sini, m2)

    # Jacobian terms
    dM2_dH4 = -3.0 * m2 / h4_safe
    dSINI_dH4 = 2.0 * h3 * (h3**2 - h4**2) / h3h4_denom**2

    return d_M2 * dM2_dH4 + d_SINI * dSINI_dH4



