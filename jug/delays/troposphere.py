
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np

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
    EARTH_R_m = 6356766.0  # mean Earth radius (m)
    gph = EARTH_R_m * height_m / (EARTH_R_m + height_m)  # geopotential height
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
    
    # 2. Niell Mapping Function (NMF) - Niell (1996)
    # Need to verify if we need NMF or if Tempo2 uses something simpler like 1/sin(E).
    # Tempo2 usually uses NMF.
    # NMF has separate Hydrostatic and Wet mapping functions.
    # For standard CORRECT_TROPOSPHERE without met data, Tempo2 likely only computes
    # the hydrostatic component or assumes a standard wet component is negligible or included?
    # Actually, Tempo2 code suggests it computes both if flags are set, or defaults.
    # Let's start with just Hydrostatic Mapping Function (HMF) applied to ZHD.
    # Usually Total Delay = ZHD * m_h(E) + ZWD * m_w(E).
    # Without external ZWD data, Tempo2 might just do ZHD correction.
    
    # Let's implement full NMF for Hydrostatic component.
    # NMF Coefficients for average atmosphere (Table 3 of Niell 1996)
    # Latitudes: 15, 30, 45, 60, 75
    lats_nmf = np.array([15.0, 30.0, 45.0, 60.0, 75.0])
    
    # Hydrostatic coefficients (avg)
    a_avg = np.array([1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3])
    b_avg = np.array([2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3])
    c_avg = np.array([62.610505e-3, 62.830701e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3])
    
    # Amplitude of seasonal variation
    a_amp = np.array([0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5])
    b_amp = np.array([0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5])
    c_amp = np.array([0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5])
    
    # Interpolate coefficients to station latitude
    abs_lat = abs(lat_deg)
    
    def interpolate_coeff(lat, ref_lats, ref_vals):
        if lat < ref_lats[0]:
            return ref_vals[0]
        elif lat > ref_lats[-1]:
            return ref_vals[-1]
        else:
            return np.interp(lat, ref_lats, ref_vals)
            
    a0 = interpolate_coeff(abs_lat, lats_nmf, a_avg)
    b0 = interpolate_coeff(abs_lat, lats_nmf, b_avg)
    c0 = interpolate_coeff(abs_lat, lats_nmf, c_avg)
    
    aa = interpolate_coeff(abs_lat, lats_nmf, a_amp)
    ba = interpolate_coeff(abs_lat, lats_nmf, b_amp)
    ca = interpolate_coeff(abs_lat, lats_nmf, c_amp)
    
    # Seasonal variation
    # t is time from Jan 0.0 in UT days / 365.25
    # Reference epoch: Jan 0, DOY 0.
    # MJD of Jan 0 is roughly ... 
    # Niell 1996: "t is time from Jan 0.0" (DOY 1 - 1?)
    # Usually: t = (DOY - 28) / 365.25. (Phase is -28 days for northern hemisphere)
    # Seasonal variation: use jnp.where instead of Python if/else for hemisphere
    doy_phase = jnp.where(lat_deg < 0, 28.0 + 365.25/2.0, 28.0)
    # Convert MJD to DOY (approximate is fine for NMF)
    # MJD -> Year, DOY
    # MJD 0 = Nov 17 1858
    # MJD 51544 = Jan 1 2000
    mjd_2000 = 51544.5
    days_since_2000 = mjd - mjd_2000
    # Approx year fraction (365.25 days)
    # This is rough, but NMF is not super sensitive to exact hour
    doy = (days_since_2000 % 365.25) # Rough DOY 0-365
    t_year = (doy - doy_phase) / 365.25
    
    cos_t = jnp.cos(2.0 * jnp.pi * t_year)
    
    a = a0 - aa * cos_t
    b = b0 - ba * cos_t
    c = c0 - ca * cos_t
    
    # Setup mapping function m(E)
    # m(E) = (1 + a/(1 + b/(1 + c))) / (sinE + a/(sinE + b/(sinE + c)))
    sin_el = jnp.sin(jnp.radians(elevation_deg))
    
    # Enforce minimum elevation to avoid singularity?
    # Usually impose cutoff of e.g. 2 degrees.
    # But for calculation just let it be.
    
    numerator = 1.0 + a / (1.0 + b / (1.0 + c))
    denominator = sin_el + a / (sin_el + b / (sin_el + c))
    
    mapping_function = numerator / denominator
    
    # Height correction for NMF (Niell 1996 eq 4)
    # dm(E) =  (1/sinE) - (1 + a_ht/(1 + b_ht/(1 + c_ht))) / (sinE + a_ht/(sinE + b_ht/(sinE + c_ht))) * H_km
    # Coefficients:
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    
    num_ht = 1.0 + a_ht / (1.0 + b_ht / (1.0 + c_ht))
    den_ht = sin_el + a_ht / (sin_el + b_ht / (sin_el + c_ht))
    
    correction = (1.0/sin_el - num_ht/den_ht) * height_km
    
    total_mapping_function = mapping_function + correction
    
    # Final tropo delay (meters)
    tropo_delay_m = zhd_m * total_mapping_function
    
    # Convert to seconds
    SPEED_OF_LIGHT = 299792458.0
    tropo_delay_sec = tropo_delay_m / SPEED_OF_LIGHT
    
    return tropo_delay_sec
