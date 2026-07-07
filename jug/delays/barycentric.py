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

from jug.utils.constants import C_KM_S, AU_KM, SECS_PER_DAY, KPC_TO_KM

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
    use_cache: bool = True,
    ephemeris: str = "de440"
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
    
    # Keep TDB as longdouble until Astropy Time construction. Casting modern
    # MJDs to float64 here shifts Earth/observatory positions by millimeters,
    # leaving ~10 ps annual structure in Roemer-delay parity checks.
    tdb_mjd_ld = np.asarray(tdb_mjd, dtype=np.longdouble)
    tdb_mjd_cache = np.asarray(tdb_mjd_ld, dtype=np.float64)
    obs_itrf_km = np.asarray(obs_itrf_km, dtype=np.float64)
    
    # Try disk cache first
    cache_ephem = ephemeris + "_v3"
    if use_cache:
        from jug.utils.geom_cache import get_geometry_cache
        cache = get_geometry_cache()
        cached = cache.load(tdb_mjd_cache, obs_itrf_km, ephemeris=cache_ephem)
        if cached is not None:
            if _PROFILE_ENABLED:
                _call_stats['compute_ssb_obs_pos_vel']['count'] += 1
                _call_stats['compute_ssb_obs_pos_vel']['total_time'] += time.perf_counter() - func_start
            if timings is not None:
                timings['cache_hit'] = True
            return cached
    
    t0 = time.perf_counter() if timings is not None else None
    
    tdb_mjd_int = np.floor(tdb_mjd_ld)
    times = Time(
        np.asarray(tdb_mjd_int, dtype=np.float64),
        np.asarray(tdb_mjd_ld - tdb_mjd_int, dtype=np.float64),
        format='mjd',
        scale='tdb',
    )
    
    if timings is not None:
        timings['time_obj_creation'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Get Earth position and velocity
    with solar_system_ephemeris.set(ephemeris):
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

    # Get observatory position and velocity in GCRS using astropy's analytical method.
    # This matches PINT's gcrs_posvel_from_itrf / get_gcrs_posvel approach and avoids
    # the ~10 mm/s systematic error that the 1-second finite-difference introduced.
    try:
        gcrs_pv = obs_itrf.get_gcrs_posvel(obstime=times)
    except Exception as exc:
        raise RuntimeError(
            "JUG geometry requires Astropy IERS/EOP data for ITRF→GCRS site motion. "
            'Populate ~/.astropy/cache (e.g. '
            'python -c "from astropy.utils.iers import IERS_A; IERS_A.open()"). '
            f"Original error: {exc}"
        ) from exc
    geo_obs_pos = np.column_stack([
        gcrs_pv[0].x.to(u.km).value,
        gcrs_pv[0].y.to(u.km).value,
        gcrs_pv[0].z.to(u.km).value
    ])
    geo_obs_vel = np.column_stack([
        gcrs_pv[1].x.to(u.km/u.s).value,
        gcrs_pv[1].y.to(u.km/u.s).value,
        gcrs_pv[1].z.to(u.km/u.s).value
    ])

    if timings is not None:
        timings['gcrs_posvel'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Observatory position and velocity at SSB
    ssb_obs_pos = ssb_geo_pos + geo_obs_pos
    ssb_obs_vel = ssb_geo_vel + geo_obs_vel

    if timings is not None:
        timings['cache_hit'] = False
    
    # Ensure float64 output
    ssb_obs_pos = np.asarray(ssb_obs_pos, dtype=np.float64)
    ssb_obs_vel = np.asarray(ssb_obs_vel, dtype=np.float64)
    
    # Save to disk cache
    if use_cache:
        cache.save(tdb_mjd_cache, obs_itrf_km, ssb_obs_pos, ssb_obs_vel, ephemeris=cache_ephem)
    
    # Update profiling stats
    if _PROFILE_ENABLED:
        _call_stats['compute_ssb_obs_pos_vel']['count'] += 1
        _call_stats['compute_ssb_obs_pos_vel']['total_time'] += time.perf_counter() - func_start

    return ssb_obs_pos, ssb_obs_vel


def compute_ssb_obs_pos_vel_gcrs_posvel(
    tdb_mjd: np.ndarray,
    obs_itrf_km: np.ndarray,
    timings: Optional[Dict[str, float]] = None,
    ephemeris: str = "de440"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SSB position/velocity using EarthLocation.get_gcrs_posvel().

    DEPRECATED: compute_ssb_obs_pos_vel now uses get_gcrs_posvel() directly.
    This function is identical in behavior and kept only for reference.
    
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
    
    tdb_mjd_ld = np.asarray(tdb_mjd, dtype=np.longdouble)
    obs_itrf_km = np.asarray(obs_itrf_km, dtype=np.float64)
    tdb_mjd_int = np.floor(tdb_mjd_ld)
    times = Time(
        np.asarray(tdb_mjd_int, dtype=np.float64),
        np.asarray(tdb_mjd_ld - tdb_mjd_int, dtype=np.float64),
        format='mjd',
        scale='tdb',
    )
    
    if timings is not None:
        timings['time_obj_creation'] = time.perf_counter() - t0
        t0 = time.perf_counter()

    # Get Earth position and velocity
    with solar_system_ephemeris.set(ephemeris):
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
    Proper motion is propagated RIGOROUSLY along a great circle on the unit
    sphere (not the linear ra/dec += PM*dt tangent-plane approximation), to
    match PINT/astropy ``apply_space_motion`` (ERFA). The old linear update was
    wrong at O((PM*dt)^2); for nearby high-PM pulsars (e.g. J0437-4715, PM ~141
    mas/yr -> ~1 arcsec over the data span) that second-order error reached
    ~0.4 ns in the Roemer delay (secular + quadratic in time).

    Great-circle propagation:  p(t) = p0*cos(theta) + mhat*sin(theta),
    where p0 is the unit direction at POSEPOCH, theta = |mu|*dt is the total
    angular motion, and mhat is the unit on-sky proper-motion direction.

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
    dt = np.atleast_1d(np.asarray(t_mjd, dtype=np.float64)) - posepoch
    cos_dec0 = np.cos(dec_rad)
    sin_dec0 = np.sin(dec_rad)
    cos_ra0 = np.cos(ra_rad)
    sin_ra0 = np.sin(ra_rad)

    # Direction unit vector at POSEPOCH and the on-sky tangent basis.
    p0 = np.array([cos_dec0 * cos_ra0, cos_dec0 * sin_ra0, sin_dec0])
    e_ra = np.array([-sin_ra0, cos_ra0, 0.0])                       # +RA (on-sky)
    e_dec = np.array([-sin_dec0 * cos_ra0, -sin_dec0 * sin_ra0, cos_dec0])  # +Dec

    # On-sky proper-motion vector (rad/day). pmra_rad_day already includes the
    # cos(dec) factor, so (pmra_rad_day, pmdec_rad_day) are the on-sky rates.
    mu_vec = pmra_rad_day * e_ra + pmdec_rad_day * e_dec
    mu_mag = float(np.hypot(pmra_rad_day, pmdec_rad_day))

    if mu_mag == 0.0:
        return np.broadcast_to(p0, (dt.shape[0], 3)).copy()

    mhat = mu_vec / mu_mag
    theta = mu_mag * dt  # (n_times,)
    return (np.outer(np.cos(theta), p0) + np.outer(np.sin(theta), mhat))


def rotate_equatorial_to_ecliptic(vectors: np.ndarray, obliquity_rad: float) -> np.ndarray:
    """Rotate Cartesian vectors from equatorial to ecliptic coordinates."""
    vectors = np.asarray(vectors)
    cos_obl = np.cos(obliquity_rad)
    sin_obl = np.sin(obliquity_rad)
    return np.column_stack([
        vectors[:, 0],
        vectors[:, 1] * cos_obl + vectors[:, 2] * sin_obl,
        -vectors[:, 1] * sin_obl + vectors[:, 2] * cos_obl,
    ])


def compute_ecliptic_pulsar_direction(
    lon_deg: float,
    lat_deg: float,
    pm_lon_mas_yr: float,
    pm_lat_mas_yr: float,
    posepoch: float,
    t_mjd: np.ndarray,
) -> np.ndarray:
    """Compute native ecliptic pulsar direction with Tempo2-style PM fields."""
    dt_years = (np.asarray(t_mjd) - posepoch) / 365.25
    lon0 = np.deg2rad(lon_deg)
    lat0 = np.deg2rad(lat_deg)
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)

    # PMELONG/PMLAMBDA include cos(latitude), matching PMRA convention.
    cos_lat0 = np.cos(lat0)
    lon = lon0 + pm_lon_mas_yr * mas_to_rad * dt_years / cos_lat0
    lat = lat0 + pm_lat_mas_yr * mas_to_rad * dt_years

    cos_lat = np.cos(lat)
    return np.column_stack([
        cos_lat * np.cos(lon),
        cos_lat * np.sin(lon),
        np.sin(lat),
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

    # Basic Roemer delay: -r*L/c
    roemer_sec = -re_dot_L / C_KM_S

    # Parallax correction (second-order effect)
    if parallax_mas != 0.0:
        # Distance to pulsar in kpc
        distance_kpc = 1.0 / parallax_mas
        # Convert to km
        L_km = distance_kpc * KPC_TO_KM

        # Magnitude squared of position vector
        re_sqr = np.sum(ssb_obs_pos_km**2, axis=1)

        # Parallax delay: 0.5 * r^2 / L * (1 - cos^2(theta)) / c
        # Guard against re_sqr = 0 (observer at SSB) to avoid 0/0 NaN
        with np.errstate(invalid='ignore'):
            parallax_sec = np.where(
                re_sqr > 0,
                0.5 * (re_sqr / L_km) * (1.0 - re_dot_L**2 / re_sqr) / C_KM_S,
                0.0
            )

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
        Deltat = -2 * (GM/c^3) * ln((r - r*cos(theta)) / AU)

    where r is the distance from observatory to the body, and theta is the
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
    L_hat: np.ndarray,
    einstein_rate: np.ndarray | None = None,
) -> np.ndarray:
    """Compute barycentric frequency (Doppler-corrected observing frequency).

    Corrects the topocentric (observed) frequency for the Doppler shift
    due to the observatory's motion relative to the Solar System Barycenter,
    and optionally for Einstein time dilation (DILATEFREQ).

    Parameters
    ----------
    freq_topo_mhz : np.ndarray
        Topocentric observing frequencies (MHz), shape (n_times,)
    ssb_obs_vel_km_s : np.ndarray
        Observatory velocity relative to SSB (km/s), shape (n_times, 3)
    L_hat : np.ndarray
        Pulsar direction unit vectors, shape (n_times, 3)
    einstein_rate : np.ndarray or None, optional
        Einstein rate (dTDB/dTT) per TOA, shape (n_times,).
        If provided (DILATEFREQ=Y), divides freq by this rate.

    Returns
    -------
    np.ndarray
        Barycentric frequencies (MHz), shape (n_times,)
    """
    # Radial velocity (positive = moving away from pulsar)
    v_radial = np.sum(ssb_obs_vel_km_s * L_hat, axis=1)

    # Doppler correction: f_bary = f_topo * (1 - v/c)
    freq_bary = freq_topo_mhz * (1.0 - v_radial / C_KM_S)

    # Einstein rate correction (GR time dilation)
    if einstein_rate is not None:
        freq_bary = freq_bary / einstein_rate

    return freq_bary


def compute_einstein_rate(tdb_mjd: np.ndarray, units: str = 'TDB') -> np.ndarray:
    """Compute the Einstein rate (dTDB/dTT) for each TOA.

    This is used by DILATEFREQ to correct observing frequencies for
    gravitational time dilation in the Solar System.

    Parameters
    ----------
    tdb_mjd : np.ndarray
        TDB MJD values, shape (n_toas,)
    units : str
        Timescale units: 'TDB' or 'TCB'. For TDB, rate ≈ 1 ± 7e-10.
        For TCB, rate includes IFTE_K factor.

    Returns
    -------
    np.ndarray
        Einstein rate per TOA, shape (n_toas,)

    Notes
    -----
    Uses numerical differentiation of the TDB-TT relationship via astropy.
    Matches Tempo2's einsteinRate computation in tt2tdb.C.
    """
    from astropy.time import Time

    dt_days = 0.001  # ~86 seconds, small enough for accurate derivative

    # Use longdouble then jd1/jd2 split to avoid "large_mjd + small_dt" float64
    # precision loss. For MJD~57000, float64 ULP~1.3e-11 days gives rate noise
    # ~9e-9, which is ~27x larger than the Einstein rate signal (~3e-10).
    # With jd1/jd2 split, jd2~0.5 has ULP~1e-16 days → noise~1e-13 (negligible).
    tdb_ld = np.asarray(tdb_mjd, dtype=np.longdouble)
    mjd_int = np.floor(tdb_ld)
    mjd_frac = tdb_ld - mjd_int
    jd1 = (mjd_int + 2400000.0).astype(np.float64)
    jd2 = (mjd_frac + 0.5).astype(np.float64)  # JD frac in [0.5, 1.5)

    t1 = Time(jd1, jd2, format='jd', scale='tdb')
    t2 = Time(jd1, jd2 + dt_days, format='jd', scale='tdb')

    tt1 = t1.tt
    tt2 = t2.tt
    tt_diff = (tt2.jd1 - tt1.jd1) + (tt2.jd2 - tt1.jd2)
    rate = dt_days / tt_diff

    # Tempo2 applies DILATEFREQ in the model timescale.  For TCB par files,
    # dTCB/dTT has the same periodic Einstein term as dTDB/dTT plus the
    # constant Irwin-Fukushima scale factor.
    if str(units).upper() == 'TCB':
        from jug.utils.timescales import IFTE_K
        rate = rate * np.float64(IFTE_K)

    return rate
