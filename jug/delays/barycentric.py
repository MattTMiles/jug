"""Barycentric delay calculations and astrometric functions.

This module provides functions to compute the geometric (Roemer) delay,
Shapiro delay, and related astrometric quantities for pulsar timing.
"""

import os
import time
import traceback
from typing import Dict, Optional, Tuple
import numpy as np
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris

from jug.utils.constants import C_KM_S, AU_KM, SECS_PER_DAY

# Profiling support (guarded by env var)
_PROFILE_ENABLED = os.environ.get('JUG_PROFILE_GEOM', '').strip() == '1'
_call_stats = {
    'compute_ssb_obs_pos_vel': {'count': 0, 'total_time': 0.0, 'call_sites': []}
}


def get_geometry_profile_stats() -> dict:
    """Get profiling statistics for geometry functions.
    
    Only populated when JUG_PROFILE_GEOM=1 environment variable is set.
    
    Returns
    -------
    dict
        Statistics including call counts, total time, and call sites.
    """
    return dict(_call_stats)


def reset_geometry_profile_stats():
    """Reset profiling statistics."""
    global _call_stats
    _call_stats = {
        'compute_ssb_obs_pos_vel': {'count': 0, 'total_time': 0.0, 'call_sites': []}
    }


def compute_ssb_obs_pos_vel(
    tdb_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    timings: Optional[Dict[str, float]] = None,
    use_cache: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute observatory position and velocity relative to Solar System Barycenter.

    Uses Astropy with JPL DE440 ephemeris to compute high-precision
    observatory position and velocity at the SSB.

    Parameters
    ----------
    tdb_mjd : np.ndarray
        Times in TDB (MJD)
    obs_itrf_km : np.ndarray
        Observatory position in ITRF coordinates (km), shape (3,) [X, Y, Z]
    timings : dict, optional
        If provided, internal stage timings are recorded into this dict.
        Keys: 'time_obj_creation', 'earth_ephemeris', 'gcrs_transform', 
              'gcrs_transform_plus', 'velocity_derivation'
    use_cache : bool, default True
        Whether to use disk cache for repeated datasets.

    Returns
    -------
    ssb_obs_pos : np.ndarray
        Observatory position relative to SSB (km), shape (n_times, 3)
    ssb_obs_vel : np.ndarray
        Observatory velocity relative to SSB (km/s), shape (n_times, 3)

    Notes
    -----
    The observatory velocity is computed using numerical differentiation
    with a 1-second timestep, which provides sufficient accuracy for
    pulsar timing applications.

    Examples
    --------
    >>> from jug.utils.constants import OBSERVATORIES
    >>> obs_pos = OBSERVATORIES['meerkat']
    >>> tdb = np.array([58000.0, 58001.0])
    >>> pos, vel = compute_ssb_obs_pos_vel(tdb, obs_pos)
    >>> print(f"Position shape: {pos.shape}")  # (2, 3)
    """
    # Profiling instrumentation
    if _PROFILE_ENABLED:
        func_start = time.perf_counter()
        # Capture call site (limited stack for efficiency)
        stack = traceback.extract_stack(limit=6)
        call_site = ' -> '.join(f"{s.filename.split('/')[-1]}:{s.lineno}" for s in stack[:-1])
        _call_stats['compute_ssb_obs_pos_vel']['call_sites'].append(call_site)
    
    # Ensure arrays are proper dtype
    tdb_mjd = np.asarray(tdb_mjd, dtype=np.float64)
    obs_itrf_km = np.asarray(obs_itrf_km, dtype=np.float64)
    
    # Try disk cache first
    if use_cache:
        from jug.utils.geom_cache import get_geometry_cache
        cache = get_geometry_cache()
        cached = cache.load(tdb_mjd, obs_itrf_km, ephemeris="de440")
        if cached is not None:
            if _PROFILE_ENABLED:
                _call_stats['compute_ssb_obs_pos_vel']['count'] += 1
                _call_stats['compute_ssb_obs_pos_vel']['total_time'] += time.perf_counter() - func_start
            if timings is not None:
                timings['cache_hit'] = True
            return cached
    
    t0 = time.perf_counter() if timings is not None else None
    
    times = Time(tdb_mjd, format='mjd', scale='tdb')
    
    if timings is not None:
        timings['time_obj_creation'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Get Earth position and velocity from DE440
    with solar_system_ephemeris.set('de440'):
        earth_pv = get_body_barycentric_posvel('earth', times)
        ssb_geo_pos = earth_pv[0].xyz.to(u.km).value.T  # Geocenter position
        ssb_geo_vel = earth_pv[1].xyz.to(u.km/u.s).value.T  # Geocenter velocity

    if timings is not None:
        timings['earth_ephemeris'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Convert observatory ITRF position to EarthLocation
    obs_itrf = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km
    )

    # Get observatory position in GCRS (geocentric celestial reference system)
    obs_gcrs = obs_itrf.get_gcrs(obstime=times)
    geo_obs_pos = np.column_stack([
        obs_gcrs.cartesian.x.to(u.km).value,
        obs_gcrs.cartesian.y.to(u.km).value,
        obs_gcrs.cartesian.z.to(u.km).value
    ])

    if timings is not None:
        timings['gcrs_transform'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Compute observatory velocity via numerical derivative
    dt_sec = 1.0  # 1 second timestep
    times_plus = Time(tdb_mjd + dt_sec/SECS_PER_DAY, format='mjd', scale='tdb')
    obs_gcrs_plus = obs_itrf.get_gcrs(obstime=times_plus)
    geo_obs_pos_plus = np.column_stack([
        obs_gcrs_plus.cartesian.x.to(u.km).value,
        obs_gcrs_plus.cartesian.y.to(u.km).value,
        obs_gcrs_plus.cartesian.z.to(u.km).value
    ])
    geo_obs_vel = (geo_obs_pos_plus - geo_obs_pos) / dt_sec  # km/s

    if timings is not None:
        timings['gcrs_transform_plus'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Observatory position and velocity at SSB
    ssb_obs_pos = ssb_geo_pos + geo_obs_pos
    ssb_obs_vel = ssb_geo_vel + geo_obs_vel

    if timings is not None:
        timings['velocity_derivation'] = time.perf_counter() - t0
        timings['cache_hit'] = False
    
    # Ensure float64 output
    ssb_obs_pos = np.asarray(ssb_obs_pos, dtype=np.float64)
    ssb_obs_vel = np.asarray(ssb_obs_vel, dtype=np.float64)
    
    # Save to disk cache
    if use_cache:
        cache.save(tdb_mjd, obs_itrf_km, ssb_obs_pos, ssb_obs_vel, ephemeris="de440")
    
    # Update profiling stats
    if _PROFILE_ENABLED:
        _call_stats['compute_ssb_obs_pos_vel']['count'] += 1
        _call_stats['compute_ssb_obs_pos_vel']['total_time'] += time.perf_counter() - func_start

    return ssb_obs_pos, ssb_obs_vel


def compute_ssb_obs_pos_vel_gcrs_posvel(
    tdb_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    timings: Optional[Dict[str, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SSB position/velocity using EarthLocation.get_gcrs_posvel().
    
    This is an EXPERIMENTAL alternative to compute_ssb_obs_pos_vel that uses
    Astropy's built-in get_gcrs_posvel() method instead of numerical
    differentiation for velocity.
    
    WARNING: This function may NOT produce bit-for-bit identical results
    to the baseline compute_ssb_obs_pos_vel. Use only if equivalence tests pass.
    
    Parameters
    ----------
    tdb_mjd : np.ndarray
        Times in TDB (MJD)
    obs_itrf_km : np.ndarray
        Observatory position in ITRF coordinates (km), shape (3,) [X, Y, Z]
    timings : dict, optional
        If provided, internal stage timings are recorded.

    Returns
    -------
    ssb_obs_pos : np.ndarray
        Observatory position relative to SSB (km), shape (n_times, 3)
    ssb_obs_vel : np.ndarray
        Observatory velocity relative to SSB (km/s), shape (n_times, 3)
    """
    t0 = time.perf_counter() if timings is not None else None
    
    times = Time(tdb_mjd, format='mjd', scale='tdb')
    
    if timings is not None:
        timings['time_obj_creation'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Get Earth position and velocity from DE440
    with solar_system_ephemeris.set('de440'):
        earth_pv = get_body_barycentric_posvel('earth', times)
        ssb_geo_pos = earth_pv[0].xyz.to(u.km).value.T
        ssb_geo_vel = earth_pv[1].xyz.to(u.km/u.s).value.T

    if timings is not None:
        timings['earth_ephemeris'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Convert observatory ITRF position to EarthLocation
    obs_itrf = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km
    )

    # Use get_gcrs_posvel for both position AND velocity
    # Returns (CartesianRepresentation for position, CartesianRepresentation for velocity)
    obs_gcrs_pv = obs_itrf.get_gcrs_posvel(obstime=times)
    
    # Position - obs_gcrs_pv[0] is already CartesianRepresentation
    geo_obs_pos = np.column_stack([
        obs_gcrs_pv[0].x.to(u.km).value,
        obs_gcrs_pv[0].y.to(u.km).value,
        obs_gcrs_pv[0].z.to(u.km).value
    ])
    
    # Velocity - obs_gcrs_pv[1] is CartesianRepresentation with velocity units
    geo_obs_vel = np.column_stack([
        obs_gcrs_pv[1].x.to(u.km/u.s).value,
        obs_gcrs_pv[1].y.to(u.km/u.s).value,
        obs_gcrs_pv[1].z.to(u.km/u.s).value
    ])

    if timings is not None:
        timings['gcrs_posvel'] = time.perf_counter() - t0

    # Observatory position and velocity at SSB
    ssb_obs_pos = ssb_geo_pos + geo_obs_pos
    ssb_obs_vel = ssb_geo_vel + geo_obs_vel

    return ssb_obs_pos, ssb_obs_vel


def compute_pulsar_direction(
    ra_rad: float,
    dec_rad: float,
    pmra_rad_day: float,
    pmdec_rad_day: float,
    posepoch: float,
    t_mjd: np.ndarray
) -> np.ndarray:
    """Compute pulsar direction unit vector with proper motion correction.

    Parameters
    ----------
    ra_rad : float
        Right ascension at POSEPOCH (radians)
    dec_rad : float
        Declination at POSEPOCH (radians)
    pmra_rad_day : float
        Proper motion in RA (radians/day, includes cos(dec) factor)
    pmdec_rad_day : float
        Proper motion in DEC (radians/day)
    posepoch : float
        Reference epoch for astrometric parameters (MJD)
    t_mjd : np.ndarray
        Times at which to compute direction (MJD)

    Returns
    -------
    np.ndarray
        Pulsar direction unit vectors, shape (n_times, 3) [x, y, z]
        in the celestial reference frame

    Notes
    -----
    Proper motion is applied linearly from POSEPOCH. For nearby pulsars
    with large proper motions, this can introduce small errors over long
    baselines, but is sufficient for most pulsar timing applications.

    Examples
    --------
    >>> ra = 1.0  # radians
    >>> dec = -0.5  # radians
    >>> pmra = 1e-10  # very small proper motion
    >>> pmdec = 1e-10
    >>> posepoch = 55000.0
    >>> times = np.array([55000.0, 56000.0])
    >>> L_hat = compute_pulsar_direction(ra, dec, pmra, pmdec, posepoch, times)
    >>> print(f"Direction vectors: {L_hat.shape}")  # (2, 3)
    """
    dt = t_mjd - posepoch
    cos_dec0 = np.cos(dec_rad)

    # Apply proper motion
    ra = ra_rad + pmra_rad_day * dt / cos_dec0
    dec = dec_rad + pmdec_rad_day * dt

    # Convert to unit vector
    cos_dec = np.cos(dec)
    sin_dec = np.sin(dec)
    cos_ra = np.cos(ra)
    sin_ra = np.sin(ra)

    return np.column_stack([
        cos_dec * cos_ra,  # x
        cos_dec * sin_ra,  # y
        sin_dec            # z
    ])


def compute_roemer_delay(
    ssb_obs_pos_km: np.ndarray,
    L_hat: np.ndarray,
    parallax_mas: float = 0.0
) -> np.ndarray:
    """Compute Roemer delay (geometric light travel time) with parallax.

    The Roemer delay is the light travel time from the observatory to
    the Solar System Barycenter in the direction of the pulsar.

    Parameters
    ----------
    ssb_obs_pos_km : np.ndarray
        Observatory position relative to SSB (km), shape (n_times, 3)
    L_hat : np.ndarray
        Pulsar direction unit vectors, shape (n_times, 3)
    parallax_mas : float, optional
        Parallax in milliarcseconds (default: 0.0)
        If non-zero, includes second-order parallax correction

    Returns
    -------
    np.ndarray
        Roemer delay in seconds, shape (n_times,)

    Notes
    -----
    The sign convention is such that positive delay means the signal
    arrives later (pulsar further away).

    The parallax correction is a second-order effect that becomes
    important for nearby pulsars (parallax > 1 mas).

    Examples
    --------
    >>> ssb_pos = np.array([[1e8, 0, 0], [0, 1e8, 0]])  # km
    >>> L_hat = np.array([[1, 0, 0], [0, 1, 0]])  # direction
    >>> delay = compute_roemer_delay(ssb_pos, L_hat, parallax_mas=0.0)
    >>> print(f"Delay: {delay[0]:.3f} seconds")
    """
    # Dot product: projection of position onto pulsar direction
    re_dot_L = np.sum(ssb_obs_pos_km * L_hat, axis=1)

    # Basic Roemer delay: -r·L/c
    roemer_sec = -re_dot_L / C_KM_S

    # Parallax correction (second-order effect)
    if parallax_mas != 0.0:
        # Distance to pulsar in kpc
        distance_kpc = 1.0 / parallax_mas
        # Convert to km
        L_km = distance_kpc * 3.085677581e16

        # Magnitude squared of position vector
        re_sqr = np.sum(ssb_obs_pos_km**2, axis=1)

        # Parallax delay: 0.5 * r^2 / L * (1 - cos^2(theta)) / c
        parallax_sec = 0.5 * (re_sqr / L_km) * (1.0 - re_dot_L**2 / re_sqr) / C_KM_S

        roemer_sec = roemer_sec + parallax_sec

    return roemer_sec


def compute_shapiro_delay(
    obs_body_pos_km: np.ndarray,
    L_hat: np.ndarray,
    T_body: float
) -> np.ndarray:
    """Compute Shapiro delay (gravitational time delay) for a massive body.

    The Shapiro delay is the extra time it takes light to travel through
    the curved spacetime near a massive body (Sun, planets, companion star).

    Parameters
    ----------
    obs_body_pos_km : np.ndarray
        Body position relative to observatory (km), shape (n_times, 3)
    L_hat : np.ndarray
        Pulsar direction unit vectors, shape (n_times, 3)
    T_body : float
        Body's GM/c^3 in seconds (see jug.utils.constants.T_PLANET)

    Returns
    -------
    np.ndarray
        Shapiro delay in seconds, shape (n_times,)

    Notes
    -----
    The formula used is:
        Δt = -2 * (GM/c^3) * ln((r - r·cos(θ)) / AU)

    where r is the distance from observatory to the body, and θ is the
    angle between the pulsar direction and the line to the body.

    The delay is negative (signal arrives earlier) when the pulsar line
    of sight passes close to the body.

    Examples
    --------
    >>> from jug.utils.constants import T_SUN_SEC
    >>> # Sun position relative to observatory
    >>> sun_pos = np.array([[1.5e8, 0, 0]])  # ~1 AU
    >>> L_hat = np.array([[1, 0, 0]])  # looking toward sun
    >>> delay = compute_shapiro_delay(sun_pos, L_hat, T_SUN_SEC)
    >>> print(f"Solar Shapiro delay: {delay[0]:.6f} seconds")
    """
    # Distance from observatory to body
    r = np.sqrt(np.sum(obs_body_pos_km**2, axis=1))

    # Projection onto pulsar direction
    rcostheta = np.sum(obs_body_pos_km * L_hat, axis=1)

    # Shapiro delay: -2 * T * ln((r - r*cos(theta)) / AU)
    return -2.0 * T_body * np.log((r - rcostheta) / AU_KM)


def compute_barycentric_freq(
    freq_topo_mhz: np.ndarray,
    ssb_obs_vel_km_s: np.ndarray,
    L_hat: np.ndarray
) -> np.ndarray:
    """Compute barycentric frequency (Doppler-corrected observing frequency).

    Corrects the topocentric (observed) frequency for the Doppler shift
    due to the observatory's motion relative to the Solar System Barycenter.

    Parameters
    ----------
    freq_topo_mhz : np.ndarray
        Topocentric observing frequencies (MHz), shape (n_times,)
    ssb_obs_vel_km_s : np.ndarray
        Observatory velocity relative to SSB (km/s), shape (n_times, 3)
    L_hat : np.ndarray
        Pulsar direction unit vectors, shape (n_times, 3)

    Returns
    -------
    np.ndarray
        Barycentric frequencies (MHz), shape (n_times,)

    Notes
    -----
    The Doppler formula used is:
        f_bary = f_topo * (1 - v_radial/c)

    where v_radial is the radial velocity (projection of observatory
    velocity onto pulsar direction).

    Examples
    --------
    >>> freq_topo = np.array([1400.0, 1400.0])  # MHz
    >>> vel = np.array([[30, 0, 0], [-30, 0, 0]])  # km/s
    >>> L_hat = np.array([[1, 0, 0], [1, 0, 0]])
    >>> freq_bary = compute_barycentric_freq(freq_topo, vel, L_hat)
    >>> # freq_bary[0] < freq_topo (moving toward pulsar)
    >>> # freq_bary[1] > freq_topo (moving away from pulsar)
    """
    # Radial velocity (positive = moving away from pulsar)
    v_radial = np.sum(ssb_obs_vel_km_s * L_hat, axis=1)

    # Doppler correction: f_bary = f_topo * (1 - v/c)
    return freq_topo_mhz * (1.0 - v_radial / C_KM_S)
