
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
from jug.utils.constants import C_M_S

def _pressure_from_height(height_m):
    """Compute surface pressure from observatory height using US Standard Atmosphere.

    Matches PINT's TroposphereDelay.pressure_from_altitude (CRC Handbook Ch14 p19).

    Parameters
    ----------
    height_m : float
        Height above ellipsoid in meters.

    Returns
    -------
    pressure_mbar : float
        Estimated surface pressure in mbar.
    """
    T = 288.15 - 0.0065 * height_m  # temperature lapse (K)
    p_kPa = 101.325 * (288.15 / T) ** (-5.25575)
    return p_kPa * 10.0  # kPa -> mbar


def compute_tropospheric_delay(elevation_deg, height_m, lat_deg, mjd, pressure_mbar=None):
    """
    Compute tropospheric delay (hydrostatic + wet) usually used in pulsar timing (e.g. Tempo2).
    
    References:
    - Zenith Hydrostatic Delay (ZHD): Davis et al. (1985)
    - Mapping Function: Niell Mapping Function (NMF), Niell (1996)
    
    Pressure is derived from observatory height via the US Standard Atmosphere
    (matching PINT) unless explicitly overridden.
    
    NOTE: Not JIT'd because of Python-level np.interp calls for NMF coefficient
    interpolation (scalar operations on the station latitude). The main array
    operations on elevation_deg and mjd use jnp and will be traced when called
    from JIT'd contexts.
    
    Parameters
    ----------
    elevation_deg : array_like
        Elevation of source in degrees.
    height_m : float
        Height of observatory above ellipsoid (meters).
    lat_deg : float
        Geodetic latitude of observatory (degrees).
    mjd : array_like
        Modified Julian Date of observation (for NMF seasonal variation).
    pressure_mbar : float, optional
        Surface pressure (mbar). If None, computed from height_m.
        
    Returns
    -------
    delay_sec : array_like
        Tropospheric delay in seconds.
    """
    
    if pressure_mbar is None:
        pressure_mbar = _pressure_from_height(height_m)

    # Ensure inputs are JAX arrays
    elevation_deg = jnp.asarray(elevation_deg, dtype=jnp.float64)
    mjd = jnp.asarray(mjd, dtype=jnp.float64)
    
    # Constants
    const_a = 0.0022768  # m/mbar (Davis et al. 1985)
    
    # 1. Zenith Hydrostatic Delay (ZHD) - Davis et al. (1985)
    lat_rad = jnp.radians(lat_deg)
    cos_2lat = jnp.cos(2.0 * lat_rad)
    height_km = height_m / 1000.0
    
    denom = 1.0 - 0.00266 * cos_2lat - 0.00028 * height_km
    zhd_m = (const_a * pressure_mbar) / denom
    
    # 2. Niell Mapping Function (NMF) - Niell (1996).
    # Match PINT's TroposphereDelay implementation: compute the Herring map
    # for neighboring latitude coefficient sets, then interpolate the map.
    lats_nmf = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

    a_avg = np.array([1.2769934e-3, 1.2769934e-3, 1.2683230e-3, 1.2465397e-3,
                      1.2196049e-3, 1.2045996e-3, 1.2045996e-3])
    b_avg = np.array([2.9153695e-3, 2.9153695e-3, 2.9152299e-3, 2.9288445e-3,
                      2.9022565e-3, 2.9024912e-3, 2.9024912e-3])
    c_avg = np.array([62.610505e-3, 62.610505e-3, 62.837393e-3, 63.721774e-3,
                      63.824265e-3, 64.258455e-3, 64.258455e-3])

    a_amp = np.array([0.0, 0.0, 1.2709626e-5, 2.6523662e-5,
                      3.4000452e-5, 4.1202191e-5, 4.1202191e-5])
    b_amp = np.array([0.0, 0.0, 2.1414979e-5, 3.0160779e-5,
                      7.2562722e-5, 11.723375e-5, 11.723375e-5])
    c_amp = np.array([0.0, 0.0, 9.0128400e-5, 4.3497037e-5,
                      84.795348e-5, 170.37206e-5, 170.37206e-5])

    abs_lat = abs(lat_deg)
    lat_idx = np.searchsorted(lats_nmf, abs_lat, side='left') - 1
    lat_idx = int(np.clip(lat_idx, 0, len(lats_nmf) - 2))
    lat0, lat1 = lats_nmf[lat_idx], lats_nmf[lat_idx + 1]
    lat_weight = (abs_lat - lat0) / (lat1 - lat0)

    season_offset = 0.5 if lat_deg < 0 else 0.0
    # Niell (1996) references the annual term to ~DOY 28, i.e. cos(2*pi*(DOY-28)/yr).
    # The MJD->phase offset must therefore be NEGATIVE (matches PINT's
    # DOY_OFFSET = -28). A previous +28.0 here had the wrong sign, shifting the
    # seasonal phase by 56 days (~0.003 ns tropo error).
    year_fraction = jnp.mod(2000.0 + (mjd - 51544.5 - 28.0) / 365.25 + season_offset, 1.0)
    cos_t = jnp.cos(2.0 * jnp.pi * year_fraction)

    def herring_map(alt_rad, a, b, c):
        sin_alt = jnp.sin(alt_rad)
        return (1.0 + a / (1.0 + b / (1.0 + c))) / (
            sin_alt + a / (sin_alt + b / (sin_alt + c))
        )

    alt_rad = jnp.radians(elevation_deg)

    maps = []
    for i in (lat_idx, lat_idx + 1):
        a = a_avg[i] + a_amp[i] * cos_t
        b = b_avg[i] + b_amp[i] * cos_t
        c = c_avg[i] + c_amp[i] * cos_t
        maps.append(herring_map(alt_rad, a, b, c))
    mapping_function = maps[0] + lat_weight * (maps[1] - maps[0])

    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    sin_el = jnp.sin(alt_rad)
    correction = (1.0 / sin_el - herring_map(alt_rad, a_ht, b_ht, c_ht)) * height_km

    total_mapping_function = mapping_function + correction

    tropo_delay_m = zhd_m * total_mapping_function
    tropo_delay_sec = tropo_delay_m / C_M_S

    return tropo_delay_sec
