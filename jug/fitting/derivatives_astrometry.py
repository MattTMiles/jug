"""Analytical derivatives for astrometry parameters (RAJ, DECJ, PMRA, PMDEC, PX,
ELONG, ELAT, PMELONG, PMELAT).

This module implements PINT-compatible analytical derivatives for
astrometric parameters. The formulas follow PINT's astrometry.py.

Ecliptic derivatives (ELONG, ELAT, PMELONG, PMELAT) are computed via the
Jacobian chain rule: d(delay)/d(ecl_param) = J^T @ d(delay)/d(eq_param),
where J is the coordinate transformation Jacobian.

The astrometric delay (Roemer delay) depends on the Earth-pulsar geometry:
    τ_astro = r · n̂ / c

where r is the SSB-to-observatory vector and n̂ is the unit vector to the pulsar.

Key insight: The design matrix columns are ∂(delay)/∂(param).
For fitting, PINT divides these by F0 to get ∂(phase)/∂(param).

Reference: PINT src/pint/models/astrometry.py
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional
import math

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s
AU_METERS = 149597870700.0  # meters per AU
SECS_PER_DAY = 86400.0
SECS_PER_YEAR = 365.25 * SECS_PER_DAY
MAS_PER_RAD = 180.0 * 3600.0 * 1000.0 / math.pi  # ~206264806 mas/rad
HOURANGLE_PER_RAD = 12.0 / math.pi  # ~3.819 hourangle/rad
DEG_PER_RAD = 180.0 / math.pi

# Obliquity values (arcseconds) for ecliptic coordinate frames
from jug.io.par_reader import OBLIQUITY_ARCSEC


@jax.jit
def compute_earth_position_angles(ssb_obs_pos: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Compute Earth position angles from SSB-to-observatory position vectors.
    
    Parameters
    ----------
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
        
    Returns
    -------
    dict
        Dictionary with:
        - 'ssb_obs_r': distance from SSB to observatory (light-seconds)
        - 'earth_ra': Earth right ascension (radians)
        - 'earth_dec': Earth declination (radians)
    """
    # Convert from light-seconds to meters for consistency
    # Note: ssb_obs_pos from JUG is in light-seconds, from PINT may be in meters
    x = ssb_obs_pos[:, 0]
    y = ssb_obs_pos[:, 1]
    z = ssb_obs_pos[:, 2]
    
    r = jnp.sqrt(x**2 + y**2 + z**2)
    xy = jnp.sqrt(x**2 + y**2)
    
    earth_ra = jnp.arctan2(y, x)
    earth_dec = jnp.arctan2(z, xy)
    
    return {
        'ssb_obs_r': r,
        'earth_ra': earth_ra,
        'earth_dec': earth_dec,
        'x': x,
        'y': y,
        'z': z,
    }


@jax.jit
def d_delay_d_RAJ(
    psr_ra: float,
    psr_dec: float,
    ssb_obs_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute derivative of delay with respect to RAJ.
    
    The astrometric delay is:
        τ = r * [cos(dec_e)*cos(dec_p)*cos(ra_e - ra_p) + sin(dec_e)*sin(dec_p)] / c
    
    The derivative w.r.t. RA (in radians) is:
        ∂τ/∂RA = r * cos(dec_e) * cos(dec_p) * sin(ra_p - ra_e) / c
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension in radians
    psr_dec : float
        Pulsar declination in radians
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
        
    Returns
    -------
    derivative : jnp.ndarray
        ∂(delay)/∂(RAJ) in units of seconds/radian
        Shape (n_toas,)
        
    Notes
    -----
    PINT returns this in units of seconds/hourangle for the design matrix,
    then divides by F0. We return seconds/radian here for consistency.
    """
    rd = compute_earth_position_angles(ssb_obs_pos)
    
    # Geometric factor: cos(dec_e) * cos(dec_p) * sin(ra_p - ra_e)
    geom = jnp.cos(rd['earth_dec']) * jnp.cos(psr_dec) * jnp.sin(psr_ra - rd['earth_ra'])
    
    # ∂τ/∂RA = r * geom / c
    # ssb_obs_r is in light-seconds, so r/c = ssb_obs_r (in seconds)
    dd_draj = rd['ssb_obs_r'] * geom  # seconds/radian
    
    return dd_draj


@jax.jit
def d_delay_d_DECJ(
    psr_ra: float,
    psr_dec: float,
    ssb_obs_pos: jnp.ndarray,
) -> jnp.ndarray:
    """Compute derivative of delay with respect to DECJ.
    
    The derivative w.r.t. DEC (in radians) is:
        ∂τ/∂DEC = r * [cos(dec_e)*sin(dec_p)*cos(ra_e - ra_p) - sin(dec_e)*cos(dec_p)] / c
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension in radians
    psr_dec : float
        Pulsar declination in radians
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
        
    Returns
    -------
    derivative : jnp.ndarray
        ∂(delay)/∂(DECJ) in units of seconds/radian
        Shape (n_toas,)
    """
    rd = compute_earth_position_angles(ssb_obs_pos)
    
    # Geometric factor
    geom = (jnp.cos(rd['earth_dec']) * jnp.sin(psr_dec) * jnp.cos(psr_ra - rd['earth_ra'])
            - jnp.sin(rd['earth_dec']) * jnp.cos(psr_dec))
    
    dd_ddecj = rd['ssb_obs_r'] * geom  # seconds/radian
    
    return dd_ddecj


@jax.jit
def d_delay_d_PMRA(
    psr_ra: float,
    psr_dec: float,
    ssb_obs_pos: jnp.ndarray,
    toas_mjd: jnp.ndarray,
    posepoch_mjd: float,
) -> jnp.ndarray:
    """Compute derivative of delay with respect to PMRA (proper motion in RA).
    
    Proper motion in RA causes the pulsar position to change linearly with time:
        RA(t) = RA_0 + PMRA * (t - POSEPOCH)
    
    The derivative w.r.t. PMRA is:
        ∂τ/∂PMRA = (t - POSEPOCH) * ∂τ/∂RA
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension in radians
    psr_dec : float
        Pulsar declination in radians (needed for RA derivative)
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
    toas_mjd : jnp.ndarray
        TOA times in MJD (TDB), shape (n_toas,)
    posepoch_mjd : float
        Position epoch in MJD (TDB)
        
    Returns
    -------
    derivative : jnp.ndarray
        ∂(delay)/∂(PMRA) in units of seconds/(rad/year)
        Shape (n_toas,)
        
    Notes
    -----
    PMRA in pulsar timing is μ_α* = dα/dt × cos(δ), i.e., proper motion
    in RA already includes the cos(dec) factor. Therefore the derivative
    does NOT include cos(psr_dec) in the geometric factor (unlike d_delay_d_RAJ).
    """
    rd = compute_earth_position_angles(ssb_obs_pos)
    
    # Time since POSEPOCH in years
    dt_days = toas_mjd - posepoch_mjd
    dt_years = dt_days / 365.25
    
    # PMRA geometric factor: cos(earth_dec) * sin(psr_ra - earth_ra)
    # Note: NO cos(psr_dec) because PMRA = dRA/dt * cos(dec)
    geom = jnp.cos(rd['earth_dec']) * jnp.sin(psr_ra - rd['earth_ra'])
    
    # ∂τ/∂PMRA = dt * r * geom / c
    dd_dpmra = dt_years * rd['ssb_obs_r'] * geom  # seconds/(rad/year)
    
    return dd_dpmra


@jax.jit
def d_delay_d_PMDEC(
    psr_ra: float,
    psr_dec: float,
    ssb_obs_pos: jnp.ndarray,
    toas_mjd: jnp.ndarray,
    posepoch_mjd: float,
) -> jnp.ndarray:
    """Compute derivative of delay with respect to PMDEC (proper motion in DEC).
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension in radians
    psr_dec : float
        Pulsar declination in radians
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
    toas_mjd : jnp.ndarray
        TOA times in MJD (TDB), shape (n_toas,)
    posepoch_mjd : float
        Position epoch in MJD (TDB)
        
    Returns
    -------
    derivative : jnp.ndarray
        ∂(delay)/∂(PMDEC) in units of seconds/(rad/year)
        Shape (n_toas,)
    """
    rd = compute_earth_position_angles(ssb_obs_pos)
    
    # Time since POSEPOCH in years
    dt_days = toas_mjd - posepoch_mjd
    dt_years = dt_days / 365.25
    
    # DEC geometric factor (same as d_delay_d_DECJ)
    geom = (jnp.cos(rd['earth_dec']) * jnp.sin(psr_dec) * jnp.cos(psr_ra - rd['earth_ra'])
            - jnp.sin(rd['earth_dec']) * jnp.cos(psr_dec))
    
    # ∂τ/∂PMDEC = dt * r * geom / c
    dd_dpmdec = dt_years * rd['ssb_obs_r'] * geom  # seconds/(rad/year)
    
    return dd_dpmdec


def compute_pulsar_unit_vector(
    psr_ra: float,
    psr_dec: float,
    toas_mjd: jnp.ndarray = None,
    posepoch_mjd: float = None,
    pmra_rad_yr: float = 0.0,
    pmdec_rad_yr: float = 0.0,
) -> tuple:
    # NOTE: Not JIT'd because of Python-level None checks and if/else branching.
    # The branches call jnp ops which ARE traced when called from JIT'd callers.
    """Compute pulsar unit vector, optionally with proper motion correction.
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension at POSEPOCH in radians
    psr_dec : float
        Pulsar declination at POSEPOCH in radians
    toas_mjd : jnp.ndarray, optional
        TOA times in MJD for proper motion correction
    posepoch_mjd : float, optional
        Position epoch in MJD
    pmra_rad_yr : float, optional
        Proper motion in RA (already multiplied by cos(dec)) in rad/yr
    pmdec_rad_yr : float, optional
        Proper motion in DEC in rad/yr
        
    Returns
    -------
    n_x, n_y, n_z : jnp.ndarray or float
        Components of unit vector pointing to pulsar
    """
    if toas_mjd is not None and posepoch_mjd is not None and (pmra_rad_yr != 0 or pmdec_rad_yr != 0):
        # Apply proper motion correction
        dt_years = (toas_mjd - posepoch_mjd) / 365.25
        
        # Update RA and DEC for proper motion
        # Note: PMRA is μ_α* = dα/dt × cos(δ), so we divide by cos(dec) to get dα/dt
        cos_dec = jnp.cos(psr_dec)
        ra_corrected = psr_ra + (pmra_rad_yr / cos_dec) * dt_years
        dec_corrected = psr_dec + pmdec_rad_yr * dt_years
        
        # Compute time-varying unit vector
        cos_dec_t = jnp.cos(dec_corrected)
        sin_dec_t = jnp.sin(dec_corrected)
        cos_ra_t = jnp.cos(ra_corrected)
        sin_ra_t = jnp.sin(ra_corrected)
        
        n_x = cos_dec_t * cos_ra_t
        n_y = cos_dec_t * sin_ra_t
        n_z = sin_dec_t
    else:
        # Fixed position (no proper motion correction)
        cos_dec = jnp.cos(psr_dec)
        sin_dec = jnp.sin(psr_dec)
        cos_ra = jnp.cos(psr_ra)
        sin_ra = jnp.sin(psr_ra)
        
        n_x = cos_dec * cos_ra
        n_y = cos_dec * sin_ra
        n_z = sin_dec
    
    return n_x, n_y, n_z


def d_delay_d_PX(
    psr_ra: float,
    psr_dec: float,
    ssb_obs_pos: jnp.ndarray,
    toas_mjd: jnp.ndarray = None,
    posepoch_mjd: float = None,
    pmra_rad_yr: float = 0.0,
    pmdec_rad_yr: float = 0.0,
) -> jnp.ndarray:
    # NOTE: Not JIT'd because compute_pulsar_unit_vector uses Python None checks.
    """Compute derivative of delay with respect to PX (parallax).
    
    The parallax delay is approximately:
        τ_px = 0.5 * (px_r^2 / AU) * PX / c
    
    where px_r is the transverse distance (perpendicular to line of sight):
        px_r = sqrt(r^2 - (r·n̂)^2)
    
    The derivative w.r.t. PX (in radians) is:
        ∂τ/∂PX = 0.5 * px_r^2 / (AU * c)
    
    Parameters
    ----------
    psr_ra : float
        Pulsar right ascension at POSEPOCH in radians
    psr_dec : float
        Pulsar declination at POSEPOCH in radians
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
    toas_mjd : jnp.ndarray, optional
        TOA times in MJD for proper motion correction
    posepoch_mjd : float, optional
        Position epoch in MJD
    pmra_rad_yr : float, optional
        Proper motion in RA (μ_α* = dα/dt × cos(δ)) in rad/yr
    pmdec_rad_yr : float, optional
        Proper motion in DEC in rad/yr
        
    Returns
    -------
    derivative : jnp.ndarray
        ∂(delay)/∂(PX) in units of seconds/radian
        Shape (n_toas,)
        
    Notes
    -----
    When proper motion parameters are provided, the pulsar direction is
    updated for each TOA epoch to match PINT's behavior. This provides
    sub-picosecond accuracy even for 20+ year datasets.
    """
    rd = compute_earth_position_angles(ssb_obs_pos)
    
    # Compute pulsar unit vector (with optional proper motion correction)
    n_x, n_y, n_z = compute_pulsar_unit_vector(
        psr_ra, psr_dec, toas_mjd, posepoch_mjd, pmra_rad_yr, pmdec_rad_yr
    )
    
    # Inner product r · n̂
    r_dot_n = rd['x'] * n_x + rd['y'] * n_y + rd['z'] * n_z
    
    # Transverse distance squared: px_r^2 = r^2 - (r·n̂)^2
    px_r_sq = rd['ssb_obs_r']**2 - r_dot_n**2
    
    # ∂τ/∂PX = 0.5 * px_r^2 / (AU * c)
    # AU in light-seconds: AU_METERS / SPEED_OF_LIGHT
    AU_LS = AU_METERS / SPEED_OF_LIGHT  # ~499.005 light-seconds
    
    dd_dpx = 0.5 * px_r_sq / AU_LS  # seconds/radian
    
    return dd_dpx


def _ecliptic_to_equatorial_jacobian(ecl_lon_deg, ecl_lat_deg, obl_rad):
    """Compute Jacobian ∂(RAJ, DECJ) / ∂(ELONG, ELAT) in radians/radians.

    Returns a 2x2 matrix J where:
        J[0,0] = ∂RAJ/∂ELONG,  J[0,1] = ∂RAJ/∂ELAT
        J[1,0] = ∂DECJ/∂ELONG, J[1,1] = ∂DECJ/∂ELAT

    All angles in radians (input ecliptic coords converted from degrees).
    """
    lon = np.radians(ecl_lon_deg)
    lat = np.radians(ecl_lat_deg)
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    cos_obl, sin_obl = np.cos(obl_rad), np.sin(obl_rad)

    # Equatorial Cartesian from ecliptic
    x = cos_lon * cos_lat
    y = sin_lon * cos_lat * cos_obl - sin_lat * sin_obl
    z = sin_lon * cos_lat * sin_obl + sin_lat * cos_obl

    # ∂(x,y,z)/∂(lon)  (lon in radians)
    dx_dlon = -sin_lon * cos_lat
    dy_dlon = cos_lon * cos_lat * cos_obl
    dz_dlon = cos_lon * cos_lat * sin_obl

    # ∂(x,y,z)/∂(lat)  (lat in radians)
    dx_dlat = -cos_lon * sin_lat
    dy_dlat = -sin_lon * sin_lat * cos_obl - cos_lat * sin_obl
    dz_dlat = -sin_lon * sin_lat * sin_obl + cos_lat * cos_obl

    # RA = atan2(y, x), DEC = asin(z)
    r_xy_sq = x**2 + y**2

    # ∂RA/∂lon = (x * dy_dlon - y * dx_dlon) / (x² + y²)
    dra_dlon = (x * dy_dlon - y * dx_dlon) / r_xy_sq
    dra_dlat = (x * dy_dlat - y * dx_dlat) / r_xy_sq

    # ∂DEC/∂lon = dz_dlon / sqrt(1 - z²) = dz_dlon / sqrt(x² + y²)
    r_xy = np.sqrt(r_xy_sq)
    ddec_dlon = dz_dlon / r_xy
    ddec_dlat = dz_dlat / r_xy

    return np.array([[dra_dlon, dra_dlat],
                     [ddec_dlon, ddec_dlat]])


def _ecliptic_pm_jacobian(ecl_lon_deg, ecl_lat_deg, obl_rad):
    """Compute Jacobian ∂(PMRA, PMDEC) / ∂(PMELONG, PMELAT).

    Both ecliptic and equatorial proper motions include the cos(lat) factor:
        PMELONG = dλ·cos(β),  PMRA = dα·cos(δ)

    The Jacobian relates the proper motion vectors through the same rotation
    as the position Jacobian, but projected onto the local tangent planes.

    Returns a 2x2 matrix J_pm where:
        J_pm[0,0] = ∂PMRA/∂PMELONG,    J_pm[0,1] = ∂PMRA/∂PMELAT
        J_pm[1,0] = ∂PMDEC/∂PMELONG,   J_pm[1,1] = ∂PMDEC/∂PMELAT
    """
    lon = np.radians(ecl_lon_deg)
    lat = np.radians(ecl_lat_deg)
    cos_lon, sin_lon = np.cos(lon), np.sin(lon)
    cos_lat, sin_lat = np.cos(lat), np.sin(lat)
    cos_obl, sin_obl = np.cos(obl_rad), np.sin(obl_rad)

    # Equatorial position (needed for RA, DEC)
    x = cos_lon * cos_lat
    y = sin_lon * cos_lat * cos_obl - sin_lat * sin_obl
    z = sin_lon * cos_lat * sin_obl + sin_lat * cos_obl
    ra_rad = np.arctan2(y, x) % (2 * np.pi)
    dec_rad = np.arcsin(np.clip(z, -1, 1))
    cos_ra, sin_ra = np.cos(ra_rad), np.sin(ra_rad)
    cos_dec, sin_dec = np.cos(dec_rad), np.sin(dec_rad)

    # For PMELONG (= dλ·cos(β)): Cartesian velocity = dλ·cos(β) × (-sin(λ), cos(λ)·cos(ε), cos(λ)·sin(ε))
    # For PMELAT (= dβ): Cartesian velocity = dβ × (-cos(λ)·sin(β), -sin(λ)·sin(β)·cos(ε) - cos(β)·sin(ε),
    #                                                  -sin(λ)·sin(β)·sin(ε) + cos(β)·cos(ε))
    # Unit PMELONG → Cartesian
    vx_lon = -sin_lon
    vy_lon = cos_lon * cos_obl
    vz_lon = cos_lon * sin_obl

    # Unit PMELAT → Cartesian
    vx_lat = -cos_lon * sin_lat
    vy_lat = -sin_lon * sin_lat * cos_obl - cos_lat * sin_obl
    vz_lat = -sin_lon * sin_lat * sin_obl + cos_lat * cos_obl

    # Project onto equatorial PM: PMRA·cos(δ) = -sin(α)·vx + cos(α)·vy
    #                              PMDEC = -cos(α)·sin(δ)·vx - sin(α)·sin(δ)·vy + cos(δ)·vz
    pmra_from_lon = -sin_ra * vx_lon + cos_ra * vy_lon
    pmra_from_lat = -sin_ra * vx_lat + cos_ra * vy_lat

    pmdec_from_lon = -cos_ra * sin_dec * vx_lon - sin_ra * sin_dec * vy_lon + cos_dec * vz_lon
    pmdec_from_lat = -cos_ra * sin_dec * vx_lat - sin_ra * sin_dec * vy_lat + cos_dec * vz_lat

    return np.array([[pmra_from_lon, pmra_from_lat],
                     [pmdec_from_lon, pmdec_from_lat]])


def compute_astrometric_delay(
    params: Dict,
    toas_mjd: jnp.ndarray,
    ssb_obs_pos_ls: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the astrometric (Roemer) delay contribution.

    The astrometric delay is the light travel time from the observatory to
    the SSB along the pulsar direction:
        τ_astro = -r · n̂ / c  (with parallax correction)

    Parameters
    ----------
    params : Dict
        Parameter dictionary with:
        - RAJ, DECJ (radians or HMS/DMS strings) - required
        - POSEPOCH (MJD) - optional, defaults to mean TOA time
        - PMRA, PMDEC (mas/yr) - optional, for proper motion correction
        - PX (mas) - optional, for parallax correction
    toas_mjd : jnp.ndarray
        TOA times in MJD (TDB), shape (n_toas,)
    ssb_obs_pos_ls : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds

    Returns
    -------
    astrometric_delay_sec : jnp.ndarray
        Astrometric delay in seconds, shape (n_toas,)

    Notes
    -----
    This function computes the full astrometric delay including:
    1. Basic Roemer delay: -r · n̂ / c
    2. Proper motion correction to the pulsar direction
    3. Second-order parallax correction
    """
    from jug.io.par_reader import parse_ra, parse_dec

    # Ensure inputs are arrays
    ssb_obs_pos_ls = jnp.asarray(ssb_obs_pos_ls, dtype=jnp.float64)
    toas_mjd = jnp.asarray(toas_mjd, dtype=jnp.float64)

    # Extract pulsar position (in radians)
    # Use jnp.float64 instead of float() so JAX can trace through this function
    raj_value = params['RAJ']
    if isinstance(raj_value, str):
        psr_ra = jnp.float64(parse_ra(raj_value))
    else:
        psr_ra = jnp.float64(raj_value)

    decj_value = params['DECJ']
    if isinstance(decj_value, str):
        psr_dec = jnp.float64(parse_dec(decj_value))
    else:
        psr_dec = jnp.float64(decj_value)

    posepoch = jnp.float64(params.get('POSEPOCH', jnp.mean(toas_mjd)))

    # Get proper motion (convert from mas/yr to rad/yr if present)
    pmra_mas_yr = params.get('PMRA', 0.0)
    pmdec_mas_yr = params.get('PMDEC', 0.0)
    pmra_rad_yr = jnp.float64(pmra_mas_yr) * (jnp.pi / 180 / 3600000)
    pmdec_rad_yr = jnp.float64(pmdec_mas_yr) * (jnp.pi / 180 / 3600000)

    # Get parallax (in mas)
    px_mas = jnp.float64(params.get('PX', 0.0))

    # Compute pulsar unit vector (with proper motion correction)
    n_x, n_y, n_z = compute_pulsar_unit_vector(
        psr_ra, psr_dec, toas_mjd, posepoch, pmra_rad_yr, pmdec_rad_yr
    )

    # Observatory position components (in light-seconds)
    x = ssb_obs_pos_ls[:, 0]
    y = ssb_obs_pos_ls[:, 1]
    z = ssb_obs_pos_ls[:, 2]

    # Inner product r · n̂ (in light-seconds)
    r_dot_n = x * n_x + y * n_y + z * n_z

    # Basic Roemer delay: -r · n̂ / c = -r · n̂ (since ssb_obs_pos is in light-seconds)
    roemer_delay = -r_dot_n  # seconds

    # Parallax correction (second-order effect)
    # Always compute — when px_mas=0 the correction is zero.
    # Using jnp.where avoids Python conditional which breaks JAX tracing.
    px_rad = px_mas * (jnp.pi / 180.0 / 3600.0 / 1000.0)

    # Distance in light-seconds: d_ls = AU_ls / px_rad
    AU_LS = AU_METERS / SPEED_OF_LIGHT  # ~499.005 light-seconds
    # Guard against division by zero — use a large distance when px_rad == 0
    d_ls = jnp.where(px_rad != 0, AU_LS / px_rad, jnp.inf)

    # r^2 magnitude
    r_sq = x**2 + y**2 + z**2

    # Transverse distance squared: px_r^2 = r^2 - (r·n̂)^2
    px_r_sq = r_sq - r_dot_n**2

    # Parallax delay: 0.5 * px_r^2 / d
    parallax_delay = 0.5 * px_r_sq / d_ls  # seconds

    roemer_delay = roemer_delay + parallax_delay

    return roemer_delay


def compute_astrometry_derivatives(
    params: Dict,
    toas_mjd: jnp.ndarray,
    ssb_obs_pos: jnp.ndarray,
    fit_params: List[str],
) -> Dict[str, jnp.ndarray]:
    """Compute astrometry parameter derivatives for the design matrix.
    
    The design matrix contains d(residual)/d(param) where residual = observed - model.
    
    In pulsar timing:
    - PINT subtracts delay from TDB: t_model = t_tdb - delay
    - JUG adds delay to dt: dt = ... + delay + ...
    
    Since PINT's delay and JUG's delay have the same sign (both negative for Roemer),
    the convention difference means:
    - PINT: M = +d(delay)/d(param)  
    - JUG: M = +d(delay)/d(param)  (match PINT for consistent fitting)
    
    Parameters
    ----------
    params : Dict
        Parameter dictionary with:
        - RAJ, DECJ (radians) - required
        - POSEPOCH (MJD) - optional, defaults to mean TOA time
        - PMRA, PMDEC (mas/yr) - optional, for proper motion correction in PX
    toas_mjd : jnp.ndarray
        TOA times in MJD (TDB), shape (n_toas,)
    ssb_obs_pos : jnp.ndarray
        SSB to observatory position vectors, shape (n_toas, 3), in light-seconds
    fit_params : List[str]
        List of parameters to compute derivatives for
        
    Returns
    -------
    derivatives : Dict[str, jnp.ndarray]
        Dictionary mapping parameter names to derivative arrays.
        Each array has shape (n_toas,).
        Units: d(residual)/d(param) in seconds per fitter-internal unit
        - RAJ: seconds/radian  (fitter stores RAJ in radians)
        - DECJ: seconds/radian  (fitter stores DECJ in radians)
        - PMRA/PMDEC: seconds/(mas/year)
        - PX: seconds/mas
    """
    from jug.io.par_reader import parse_ra, parse_dec
    
    # Ensure inputs are JAX arrays (force float64 for JAX compatibility)
    ssb_obs_pos = jnp.asarray(ssb_obs_pos, dtype=jnp.float64)
    toas_mjd = jnp.asarray(toas_mjd, dtype=jnp.float64)
    
    # Extract pulsar position (in radians)
    # RAJ/DECJ might be strings in HH:MM:SS format, need to parse
    raj_value = params['RAJ']
    if isinstance(raj_value, str):
        psr_ra = parse_ra(raj_value)
    else:
        psr_ra = float(raj_value)
    
    decj_value = params['DECJ']
    if isinstance(decj_value, str):
        psr_dec = parse_dec(decj_value)
    else:
        psr_dec = float(decj_value)
    
    posepoch = float(params.get('POSEPOCH', float(toas_mjd.mean())))
    
    # Get proper motion for PX correction (convert from mas/yr to rad/yr if present)
    # PMRA/PMDEC are in mas/yr in par files
    pmra_mas_yr = params.get('PMRA', 0.0)
    pmdec_mas_yr = params.get('PMDEC', 0.0)
    pmra_rad_yr = float(pmra_mas_yr) * (jnp.pi / 180 / 3600000) if pmra_mas_yr else 0.0
    pmdec_rad_yr = float(pmdec_mas_yr) * (jnp.pi / 180 / 3600000) if pmdec_mas_yr else 0.0
    
    derivatives = {}
    
    for param in fit_params:
        param_upper = param.upper()
        
        if param_upper == 'RAJ':
            # ∂(delay)/∂(RAJ) in seconds/radian
            # Fitter stores RAJ in radians (via parse_ra), so derivative
            # must be in sec/rad to match parameter units.
            deriv_rad = d_delay_d_RAJ(psr_ra, psr_dec, ssb_obs_pos)
            derivatives[param] = deriv_rad

        elif param_upper == 'DECJ':
            # ∂(delay)/∂(DECJ) in seconds/radian
            # Fitter stores DECJ in radians (via parse_dec), so derivative
            # must be in sec/rad to match parameter units.
            deriv_rad = d_delay_d_DECJ(psr_ra, psr_dec, ssb_obs_pos)
            derivatives[param] = deriv_rad
            
        elif param_upper == 'PMRA':
            # ∂(delay)/∂(PMRA) in seconds/(rad/year)
            deriv_rad_yr = d_delay_d_PMRA(psr_ra, psr_dec, ssb_obs_pos, toas_mjd, posepoch)
            # Convert to seconds/(mas/year)
            deriv_mas_yr = deriv_rad_yr / MAS_PER_RAD  # seconds/(mas/year)
            # Match PINT convention: +d(delay)/d(param)
            derivatives[param] = deriv_mas_yr
            
        elif param_upper == 'PMDEC':
            # ∂(delay)/∂(PMDEC) in seconds/(rad/year)
            deriv_rad_yr = d_delay_d_PMDEC(psr_ra, psr_dec, ssb_obs_pos, toas_mjd, posepoch)
            # Convert to seconds/(mas/year)
            deriv_mas_yr = deriv_rad_yr / MAS_PER_RAD  # seconds/(mas/year)
            # Match PINT convention: +d(delay)/d(param)
            derivatives[param] = deriv_mas_yr
            
        elif param_upper == 'PX':
            # ∂(delay)/∂(PX) in seconds/radian
            deriv_rad = d_delay_d_PX(
                psr_ra, psr_dec, ssb_obs_pos,
                toas_mjd=toas_mjd,
                posepoch_mjd=posepoch,
                pmra_rad_yr=pmra_rad_yr,
                pmdec_rad_yr=pmdec_rad_yr,
            )
            # Convert to seconds/mas (PX is stored in mas in par files)
            deriv_mas = deriv_rad / MAS_PER_RAD  # seconds/mas
            # Match PINT convention: +d(delay)/d(param)
            derivatives[param] = deriv_mas

        elif param_upper in ('ELONG', 'ELAT', 'PMELONG', 'PMELAT'):
            # Ecliptic derivatives via Jacobian chain rule.
            # JUG stores ecliptic coords in _ecliptic_* keys from par_reader.
            ecl_lon = float(params.get('_ecliptic_lon_deg', 0))
            ecl_lat = float(params.get('_ecliptic_lat_deg', 0))
            ecl_frame = str(params.get('_ecliptic_frame', 'IERS2010'))
            obl_rad = OBLIQUITY_ARCSEC[ecl_frame] * np.pi / (180.0 * 3600.0)

            if param_upper in ('ELONG', 'ELAT'):
                # Position Jacobian: ∂(RAJ_rad, DECJ_rad)/∂(ELONG_rad, ELAT_rad)
                J = _ecliptic_to_equatorial_jacobian(ecl_lon, ecl_lat, obl_rad)
                d_raj = d_delay_d_RAJ(psr_ra, psr_dec, ssb_obs_pos)
                d_decj = d_delay_d_DECJ(psr_ra, psr_dec, ssb_obs_pos)

                if param_upper == 'ELONG':
                    # d(delay)/d(ELONG_rad) = d(delay)/d(RAJ) * ∂RAJ/∂ELONG + d(delay)/d(DECJ) * ∂DECJ/∂ELONG
                    d_ecl_rad = d_raj * J[0, 0] + d_decj * J[1, 0]
                    # Convert to d(delay)/d(ELONG_deg) since ELONG is stored in degrees
                    derivatives[param] = d_ecl_rad * (np.pi / 180.0)
                else:
                    d_ecl_rad = d_raj * J[0, 1] + d_decj * J[1, 1]
                    derivatives[param] = d_ecl_rad * (np.pi / 180.0)

            else:
                # PM Jacobian: ∂(PMRA, PMDEC)/∂(PMELONG, PMELAT)
                J_pm = _ecliptic_pm_jacobian(ecl_lon, ecl_lat, obl_rad)
                d_pmra = d_delay_d_PMRA(psr_ra, psr_dec, ssb_obs_pos, toas_mjd, posepoch)
                d_pmdec = d_delay_d_PMDEC(psr_ra, psr_dec, ssb_obs_pos, toas_mjd, posepoch)
                # d_pmra is in sec/(rad/yr), d_pmdec is in sec/(rad/yr)

                if param_upper == 'PMELONG':
                    # d(delay)/d(PMELONG_rad_yr) then convert to d(delay)/d(PMELONG_mas_yr)
                    d_ecl_rad_yr = d_pmra * J_pm[0, 0] + d_pmdec * J_pm[1, 0]
                    derivatives[param] = d_ecl_rad_yr / MAS_PER_RAD
                else:
                    d_ecl_rad_yr = d_pmra * J_pm[0, 1] + d_pmdec * J_pm[1, 1]
                    derivatives[param] = d_ecl_rad_yr / MAS_PER_RAD
    
    return derivatives
