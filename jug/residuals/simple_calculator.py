"""Simple end-to-end residual calculator for testing JUG.

This module provides a minimal interface to compute pulsar timing residuals
without the full complexity of the complete calculator class.
"""

import math
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as utem_ephemeris
from astropy import units as u

# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file, get_longdouble, parse_ra, parse_dec, validate_par_timescale
from jug.io.tim_reader import parse_tim_file_mjds, compute_tdb_standalone_vectorized
from jug.io.clock import parse_clock_file
from jug.delays.barycentric import (
    compute_ssb_obs_pos_vel,
    compute_pulsar_direction,
    compute_roemer_delay,
    compute_shapiro_delay,
    compute_barycentric_freq
)
from jug.delays.combined import compute_total_delay_jax
from jug.utils.constants import (
    SECS_PER_DAY, T_SUN_SEC, T_PLANET, OBSERVATORIES, K_DM_SEC
)


def compute_residuals_simple(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Path | str | None = None,
    observatory: str = "meerkat",
    subtract_tzr: bool = True,
    verbose: bool = True,
    tzrmjd_scale: str = "AUTO"
) -> dict:
    """Compute pulsar timing residuals from .par and .tim files.

    This is a simplified calculator for testing. For production use,
    consider the full JUGResidualCalculator class (when implemented).

    Parameters
    ----------
    par_file : Path or str
        Path to .par file with timing model parameters
    tim_file : Path or str
        Path to .tim file with TOAs
    clock_dir : Path or str or None, optional
        Directory containing clock files. If None (default), uses the
        data/clock directory in the JUG package installation
    observatory : str, optional
        Observatory name (default: "meerkat")
    subtract_tzr : bool, optional
        Whether to subtract TZR phase from residuals (default: True)
        Set to False for fitting to preserve parameter signals
    verbose : bool, optional
        Whether to print progress messages (default: True)
        Set to False for production/batch processing
    tzrmjd_scale : str, optional
        Timescale interpretation for TZRMJD. Options: "AUTO" (default), "TDB", "UTC".
        - "AUTO": Derive from par file UNITS keyword (recommended). If UNITS=TDB,
          treat TZRMJD as TDB; if UNITS=TCB/TT, fail via validate_par_timescale.
        - "TDB": Force TZRMJD to be treated as TDB (no conversion).
        - "UTC": Force legacy UTC->TDB conversion via clock chain. WARNING: This
          contradicts UNITS=TDB and will produce ~69s offset. Use only for legacy
          par files that genuinely have UTC TZRMJD values.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'residuals_us': Residuals in microseconds (np.ndarray)
        - 'rms_us': RMS of residuals in microseconds (float)
        - 'mean_us': Mean of residuals in microseconds (float)
        - 'n_toas': Number of TOAs (int)
        - 'tdb_mjd': TDB times in MJD (np.ndarray)

    Examples
    --------
    >>> result = compute_residuals_simple(
    ...     "J0437-4715.par",
    ...     "J0437-4715.tim",
    ...     clock_dir="data/clock"
    ... )
    >>> print(f"RMS: {result['rms_us']:.3f} μs")
    >>> print(f"N_TOAs: {result['n_toas']}")
    """
    if verbose:
        if verbose: print("=" * 60)
        if verbose: print("JUG Simple Residual Calculator")
        if verbose: print("=" * 60)

    # Set default clock directory relative to package installation
    if clock_dir is None:
        # Get the directory where this module is located
        module_dir = Path(__file__).parent
        # Navigate to the JUG root directory and then to data/clock
        clock_dir = module_dir.parent.parent / "data" / "clock"

    # Parse files
    if verbose: print(f"\n1. Loading files...")
    params = parse_par_file(par_file)
    
    # Validate par file timescale (fail fast on TCB)
    par_timescale = validate_par_timescale(params, context="compute_residuals_simple")
    if verbose: print(f"   Par file timescale: {par_timescale}")
    
    toas = parse_tim_file_mjds(tim_file)
    if verbose: print(f"   Loaded {len(toas)} TOAs from {Path(tim_file).name}")
    if verbose: print(f"   Loaded timing model from {Path(par_file).name}")

    # Load clock files
    if verbose: print(f"\n2. Loading clock corrections...")
    clock_dir = Path(clock_dir)
    mk_clock = parse_clock_file(clock_dir / "mk2utc.clk")
    gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")
    bipm_clock = parse_clock_file(clock_dir / "tai2tt_bipm2024.clk")
    if verbose: print(f"   Loaded 3 clock files (using BIPM2024)")
    
    # Validate clock file coverage
    from jug.io.clock import check_clock_files
    mjd_utc = np.array([toa.mjd_int + toa.mjd_frac for toa in toas])
    mjd_start = np.min(mjd_utc)
    mjd_end = np.max(mjd_utc)
    
    if verbose: print(f"\n   Validating clock file coverage (MJD {mjd_start:.1f} - {mjd_end:.1f})...")
    clock_ok = check_clock_files(mjd_start, mjd_end, mk_clock, gps_clock, bipm_clock, verbose=verbose)
    if not clock_ok:
        if verbose: print(f"   ⚠️  Clock file validation found issues (see above)")

    # Observatory location
    obs_itrf_km = OBSERVATORIES.get(observatory.lower())
    if obs_itrf_km is None:
        raise ValueError(f"Unknown observatory: {observatory}")

    location = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km
    )

    # Compute TDB
    if verbose: print(f"\n3. Computing TDB (standalone, no PINT)...")
    mjd_ints = [toa.mjd_int for toa in toas]
    mjd_fracs = [toa.mjd_frac for toa in toas]

    tdb_mjd = compute_tdb_standalone_vectorized(
        mjd_ints, mjd_fracs,
        mk_clock, gps_clock, bipm_clock,
        location
    )
    if verbose: print(f"   Computed TDB for {len(tdb_mjd)} TOAs")

    # Astrometry
    if verbose: print(f"\n4. Computing astrometric delays...")
    ra_rad = parse_ra(params['RAJ'])
    dec_rad = parse_dec(params['DECJ'])
    pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
    pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
    posepoch = params.get('POSEPOCH', params['PEPOCH'])
    parallax_mas = params.get('PX', 0.0)

    ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km)
    L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_mjd)

    # Roemer and Shapiro delays
    roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)

    times = Time(tdb_mjd, format='mjd', scale='tdb')
    with solar_system_ephemeris.set('de440'):
        sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
    obs_sun_pos_km = sun_pos - ssb_obs_pos_km
    sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)

    # Planet Shapiro (if enabled)
    planet_shapiro_enabled = str(params.get('PLANET_SHAPIRO', 'N')).upper() in ('Y', 'YES', 'TRUE', '1')
    planet_shapiro_sec = np.zeros(len(tdb_mjd))
    if planet_shapiro_enabled:
        if verbose: print(f"   Computing planetary Shapiro delays...")
        with solar_system_ephemeris.set('de440'):
            for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
                planet_pos = get_body_barycentric_posvel(planet, times)[0].xyz.to(u.km).value.T
                obs_planet_km = planet_pos - ssb_obs_pos_km
                planet_shapiro_sec += compute_shapiro_delay(obs_planet_km, L_hat, T_PLANET[planet])

    roemer_shapiro = roemer_sec + sun_shapiro_sec + planet_shapiro_sec

    # Barycentric frequency
    freq_mhz = np.array([toa.freq_mhz for toa in toas])
    freq_bary_mhz = compute_barycentric_freq(freq_mhz, ssb_obs_vel_km_s, L_hat)

    # Prepare JAX arrays
    if verbose: print(f"\n5. Running JAX delay kernel...")
    tdb_jax = jnp.array(tdb_mjd, dtype=jnp.float64)
    freq_bary_jax = jnp.array(freq_bary_mhz, dtype=jnp.float64)
    obs_sun_jax = jnp.array(obs_sun_pos_km, dtype=jnp.float64)
    L_hat_jax = jnp.array(L_hat, dtype=jnp.float64)
    roemer_shapiro_jax = jnp.array(roemer_shapiro, dtype=jnp.float64)

    # DM coefficients
    dm_coeffs = []
    k = 0
    while True:
        key = 'DM' if k == 0 else f'DM{k}'
        if key in params:
            dm_coeffs.append(float(params[key]))
            k += 1
        else:
            break
    dm_coeffs = np.array(dm_coeffs if dm_coeffs else [0.0])
    dm_coeffs_jax = jnp.array(dm_coeffs, dtype=jnp.float64)
    dm_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(dm_coeffs))], dtype=jnp.float64)
    dm_epoch_jax = jnp.array(float(params.get('DMEPOCH', params['PEPOCH'])), dtype=jnp.float64)

    # FD parameters
    fd_coeffs = []
    fd_idx = 1
    while f'FD{fd_idx}' in params:
        fd_coeffs.append(float(params[f'FD{fd_idx}']))
        fd_idx += 1
    fd_coeffs_jax = jnp.array(fd_coeffs, dtype=jnp.float64) if fd_coeffs else jnp.array([0.0], dtype=jnp.float64)
    has_fd_jax = jnp.array(len(fd_coeffs) > 0)
    ne_sw_jax = jnp.array(float(params.get('NE_SW', 0.0)))

    # Binary parameters - detect model and route appropriately
    has_binary = 'PB' in params or 'FB0' in params
    binary_model = params.get('BINARY', 'NONE').upper() if has_binary else 'NONE'

    # Check for DDK early and fail explicitly (DDK not implemented)
    # Uses centralized helper for consistent behavior across all code paths
    if binary_model == 'DDK':
        from jug.utils.binary_model_overrides import resolve_binary_model
        binary_model = resolve_binary_model(binary_model, warn=True)

    # Map model name to ID
    # 0: None, 1: ELL1/H, 2: DD/DDH/DDGR, 3: T2, 4: BT*
    model_id = 0
    if has_binary:
        if binary_model in ('ELL1', 'ELL1H'):
            model_id = 1
        elif binary_model in ('DD', 'DDH', 'DDGR'):
            model_id = 2
        elif binary_model == 'T2':
            model_id = 3
        elif binary_model in ('BT', 'BTX'):
            model_id = 4
        else:
            # Fallback for unknown models (treat as DD if looks like DD?)
            # For now default to 0 with warning or error?
            # Assuming valid model
            pass

    if verbose: print(f"\n5. Detecting binary model: {binary_model} (ID: {model_id})")

    # Initialize ALL binary parameters to 0.0 (JAX superset)
    pb_val = float(params.get('PB', 0.0))
    
    # If PB is missing but FB0 exists, derive PB (for T0 conversion)
    if pb_val == 0.0 and 'FB0' in params:
        fb0 = float(params['FB0'])
        if fb0 != 0.0:
            pb_val = (1.0 / fb0) / 86400.0 # Convert Hz^-1 (sec) to Days
            
    a1_val = float(params.get('A1', 0.0))
    
    # Raw parameter extraction
    t0_val = float(params.get('T0', 0.0))
    tasc_val = float(params.get('TASC', 0.0))
    
    ecc_val = float(params.get('ECC', 0.0))
    om_val = float(params.get('OM', 0.0))
    
    eps1_val = float(params.get('EPS1', 0.0))
    eps2_val = float(params.get('EPS2', 0.0))

    # Parameter Conversion Logic for Universal Kernel
    # If using T2/DD (ID > 1) but provided with ELL1 parameters (EPS1/EPS2) and no ECC/OM,
    # convert ELL1 -> Keplerian.
    if model_id > 1:
        has_ell1_params = 'EPS1' in params or 'EPS2' in params
        has_kepler_params = 'ECC' in params and 'OM' in params
        
        if has_ell1_params and not has_kepler_params:
            if verbose: print("   Converting ELL1 parameters (EPS1/EPS2) to Keplerian (ECC/OM/T0) for T2/DD model parameterization")
            # ELL1 definition: EPS1 = e * sin(omega), EPS2 = e * cos(omega)
            # e = sqrt(EPS1^2 + EPS2^2)
            # omega = atan2(EPS1, EPS2)
            
            ecc_derived = np.sqrt(eps1_val**2 + eps2_val**2)
            om_rad = np.arctan2(eps1_val, eps2_val)
            om_deg = np.degrees(om_rad) % 360.0
            
            # T0 conversion: T0 = TASC + (omega/2pi) * PB
            # Note: This is an approximation for small e, but likely sufficient or matches Tempo2's T2 conversion
            # Exact T0 is time of periastron passage.
            # At TASC, u (mean anomaly + omega) = 0 ?? No.
            # TASC is time of ascending node. True anomaly nu = -omega.
            # Standard conversion: T0 = TASC + PB/2pi * (E - e*sinE) ? No that's M.
            # For small e, M ~ nu. M_asc = -omega.
            # T_asc = T0 + M_asc / n = T0 - omega / n = T0 - omega / (2pi/PB)
            # So T0 = T_asc + omega * PB / 2pi.
            
            t0_derived = tasc_val + (om_deg / 360.0) * pb_val
            
            # Use derived values
            ecc_val = ecc_derived
            om_val = om_deg
            t0_val = t0_derived
            
    # Normalize T0/TASC for the kernel if missing
    if t0_val == 0.0 and tasc_val != 0.0:
        t0_val = tasc_val # Fallback if no conversion happened but we need something
    if tasc_val == 0.0 and t0_val != 0.0:
        tasc_val = t0_val # Fallback

    
    gamma_val = float(params.get('GAMMA', 0.0))
    pbdot_val = float(params.get('PBDOT', 0.0))
    xdot_val = float(params.get('XDOT', 0.0))
    omdot_val = float(params.get('OMDOT', 0.0))
    edot_val = float(params.get('EDOT', 0.0))
    
    m2_val = float(params.get('M2', 0.0))
    sini_val = 0.0
    sini_param = params.get('SINI', 0.0)
    if isinstance(sini_param, str) and sini_param.upper() == 'KIN':
        sini_val = float(params.get('KIN', 0.0)) # This is technically sin(KIN) needed? 
        # Wait, wrapper logic in t2 handles "SINI" as float. 
        # If parameters pass SINI='KIN', we need to resolve it.
        # But 'SINI' usually holds the sine value. 
        # Let's assume params dictionary resolves to values or we parse it.
        pass 
    try:
        sini_val = float(sini_param)
    except ValueError:
        pass # Handle KIN reference logic if needed, but for now strict float
        
    kin_val = float(params.get('KIN', 0.0))
    kom_val = float(params.get('KOM', 0.0))
    
    h3_val = float(params.get('H3', 0.0))
    h4_val = float(params.get('H4', 0.0))
    stig_val = float(params.get('STIG', 0.0))
    
    # Shapiro M2/SINI vs H3/STIG
    r_shap_val = 0.0
    s_shap_val = 0.0
    if 'H3' in params and 'STIG' in params:
        r_shap_val = h3_val
        s_shap_val = stig_val
    elif 'M2' in params:
        r_shap_val = T_SUN_SEC * m2_val
        s_shap_val = sini_val

    # FB Parameters
    use_fb = 'FB0' in params and 'PB' not in params
    if use_fb:
        fb_coeffs = []
        fb_idx = 0
        while f'FB{fb_idx}' in params:
            fb_coeffs.append(float(params[f'FB{fb_idx}']))
            fb_idx += 1
        fb_coeffs_jax = jnp.array(fb_coeffs, dtype=jnp.float64)
        fb_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(fb_coeffs))], dtype=jnp.float64)
        fb_epoch_jax = jnp.array(float(params.get('TASC', params.get('T0', params['PEPOCH']))))
        use_fb_jax = jnp.array(True)
        # Dummy PB to avoid div/0 in non-FB branches if they run (switch mostly handles this)
        pb_val = 1.0 
    else:
        fb_coeffs_jax = jnp.array([0.0], dtype=jnp.float64)
        fb_factorials_jax = jnp.array([1.0], dtype=jnp.float64)
        fb_epoch_jax = jnp.array(0.0)
        use_fb_jax = jnp.array(False)

    # JAX Arrays for Scalars
    has_binary_jax = jnp.array(has_binary)
    binary_model_id_jax = jnp.array(model_id, dtype=jnp.int32)
    
    pb_jax = jnp.array(pb_val)
    a1_jax = jnp.array(a1_val)
    tasc_jax = jnp.array(tasc_val)
    t0_jax = jnp.array(t0_val)
    
    ecc_jax = jnp.array(ecc_val)
    om_jax = jnp.array(om_val)
    eps1_jax = jnp.array(eps1_val)
    eps2_jax = jnp.array(eps2_val)
    
    gamma_jax = jnp.array(gamma_val)
    pbdot_jax = jnp.array(pbdot_val)
    xdot_jax = jnp.array(xdot_val)
    omdot_jax = jnp.array(omdot_val)
    edot_jax = jnp.array(edot_val)
    
    m2_jax = jnp.array(m2_val)
    sini_jax = jnp.array(sini_val)
    kin_jax = jnp.array(kin_val)
    kom_jax = jnp.array(kom_val)
    h3_jax = jnp.array(h3_val)
    h4_jax = jnp.array(h4_val)
    stig_jax = jnp.array(stig_val)
    
    r_shap_jax = jnp.array(r_shap_val)
    s_shap_jax = jnp.array(s_shap_val)

    # DDK Kopeikin parameters
    # Observer position in light-seconds (convert from km)
    SPEED_OF_LIGHT_KM_S = 299792.458
    obs_pos_ls_jax = jnp.array(ssb_obs_pos_km / SPEED_OF_LIGHT_KM_S, dtype=jnp.float64)

    # Parallax in milliarcseconds
    px_jax = jnp.array(parallax_mas)

    # Pulsar coordinates for Kopeikin projections
    # ra_rad and dec_rad are already defined earlier
    sin_ra_jax = jnp.array(np.sin(ra_rad))
    cos_ra_jax = jnp.array(np.cos(ra_rad))
    sin_dec_jax = jnp.array(np.sin(dec_rad))
    cos_dec_jax = jnp.array(np.cos(dec_rad))

    # K96 proper motion parameters (Kopeikin 1996)
    # K96 flag: default True for DDK, can be disabled via par file
    k96_flag = True
    if 'K96' in params:
        k96_param = params['K96']
        if isinstance(k96_param, bool):
            k96_flag = k96_param
        elif isinstance(k96_param, str):
            k96_flag = k96_param.upper() not in ('N', 'NO', 'FALSE', '0', 'F')
        else:
            k96_flag = bool(k96_param)
    k96_jax = jnp.array(k96_flag)

    # Convert proper motion from mas/yr to radians/second
    # PMRA in par files is typically μ_α * cos(δ) in mas/yr
    # For K96, we need μ_long = PMRA / cos(DEC) (actual angular velocity in RA)
    # and μ_lat = PMDEC
    MAS_PER_YR_TO_RAD_PER_SEC = (np.pi / 180.0 / 3600.0 / 1000.0) / (365.25 * 86400.0)

    pmra_mas_yr = float(params.get('PMRA', 0.0))  # This is μ_α * cos(δ)
    pmdec_mas_yr = float(params.get('PMDEC', 0.0))

    # For K96, we need μ_long = PMRA / cos(DEC)
    # But PINT uses PMRA directly (which is already μ_α * cos(δ))
    # Looking at PINT's DDK_model.py, it uses PMLONG_DDK and PMLAT_DDK
    # which are set equal to PMRA and PMDEC by default
    # So we should use PMRA and PMDEC directly
    pmra_rad_per_sec = pmra_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC
    pmdec_rad_per_sec = pmdec_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC

    pmra_rad_per_sec_jax = jnp.array(pmra_rad_per_sec)
    pmdec_rad_per_sec_jax = jnp.array(pmdec_rad_per_sec)

    if verbose and model_id == 5:
        print(f"   DDK model with Kopeikin corrections:")
        print(f"     KIN={kin_val:.3f}°, KOM={kom_val:.3f}°, PX={parallax_mas:.3f} mas")
        print(f"     K96={k96_flag}, PMRA={pmra_mas_yr:.3f} mas/yr, PMDEC={pmdec_mas_yr:.3f} mas/yr")

    # === Tropospheric Delay (compute BEFORE kernel for PINT-compatible pre-binary time) ===
    # Check CORRECT_TROPOSPHERE flag (usually 'Y' or 'T')
    # Default is False unless specified
    correct_troposphere = False
    if 'CORRECT_TROPOSPHERE' in params:
        flag = str(params['CORRECT_TROPOSPHERE']).upper().strip()
        if flag in ['Y', 'T', '1', 'TRUE']:
            correct_troposphere = True
    
    tropo_delay_sec = np.zeros(len(toas), dtype=np.float64)
    if correct_troposphere:
        if verbose: print(f"   Calculating tropospheric delay (Davis ZHD + Niell MF)...")
        from jug.delays.troposphere import compute_tropospheric_delay
        
        loc_height_m = location.height.to(u.m).value
        loc_lat_deg = location.lat.deg
        
        # Source coordinate (J2000)
        source_coord = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, frame='icrs')
        
        # Observation times (UTC)
        mjd_utc_arr = np.array([t.mjd_int + t.mjd_frac for t in toas])
        obs_times = Time(mjd_utc_arr, format='mjd', scale='utc')
        
        # Transform to AltAz
        topocentric_frame = AltAz(obstime=obs_times, location=location)
        source_altaz = source_coord.transform_to(topocentric_frame)
        elevation_deg = source_altaz.alt.deg
        
        # Calculate delay
        tropo_delay_sec = compute_tropospheric_delay(
            elevation_deg=elevation_deg,
            height_m=loc_height_m,
            lat_deg=loc_lat_deg,
            mjd=mjd_utc_arr
        )
        
        if verbose: print(f"   Tropospheric delay: mean={np.mean(tropo_delay_sec)*1e6:.3f} μs, range=[{np.min(tropo_delay_sec)*1e6:.3f}, {np.max(tropo_delay_sec)*1e6:.3f}] μs")
    
    tropo_jax = jnp.array(tropo_delay_sec, dtype=jnp.float64)

    # Compute total delay (DM + SW + FD + binary)
    # Note: Troposphere is passed to kernel for PINT-compatible pre-binary time calculation,
    # but is also added to total delay separately (kernel only uses it for binary time).
    if verbose: print(f"\n6. Running JAX delay kernel...")
    total_delay_jax = compute_total_delay_jax(
        tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
        dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
        ne_sw_jax, fd_coeffs_jax, has_fd_jax,
        roemer_shapiro_jax, has_binary_jax, binary_model_id_jax,
        pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax, pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax,
        ecc_jax, om_jax, t0_jax, omdot_jax, edot_jax, m2_jax, sini_jax, kin_jax, kom_jax, h3_jax, h4_jax, stig_jax,
        fb_coeffs_jax, fb_factorials_jax, fb_epoch_jax, use_fb_jax,
        # DDK Kopeikin parameters (Kopeikin 1995)
        obs_pos_ls_jax, px_jax, sin_ra_jax, cos_ra_jax, sin_dec_jax, cos_dec_jax,
        # K96 proper motion parameters (Kopeikin 1996)
        k96_jax, pmra_rad_per_sec_jax, pmdec_rad_per_sec_jax,
        # Tropospheric delay (for PINT-compatible pre-binary time)
        tropo_jax
    ).block_until_ready()
    
    # Add external binary delay if we used dispatcher
    total_delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble)

    # Add tropospheric delay to total (kernel uses it only for binary time, not in sum)
    if correct_troposphere:
        total_delay_sec += tropo_delay_sec

    # Compute DM and SW delays separately for pre-binary time (needed by fitter)
    # These replicate the kernel formulas in NumPy for use outside the kernel
    dm_epoch = float(params.get('DMEPOCH', params['PEPOCH']))
    dt_years = (np.array(tdb_mjd) - dm_epoch) / 365.25
    dm_eff = sum(dm_coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(dm_coeffs)))
    dm_delay_sec = K_DM_SEC * dm_eff / (freq_bary_mhz ** 2)
    
    # Solar wind delay
    ne_sw = float(params.get('NE_SW', 0.0))
    if ne_sw > 0:
        AU_KM = 1.495978707e8
        AU_PC = 4.84813681e-6
        r_km = np.sqrt(np.sum(obs_sun_pos_km**2, axis=1))
        r_au = r_km / AU_KM
        sun_dir = obs_sun_pos_km / r_km[:, np.newaxis]
        cos_elong = np.sum(sun_dir * L_hat, axis=1)
        elong = np.arccos(np.clip(cos_elong, -1.0, 1.0))
        rho = np.pi - elong
        sin_rho = np.maximum(np.sin(rho), 1e-10)
        geometry_pc = AU_PC * rho / (r_au * sin_rho)
        dm_sw = ne_sw * geometry_pc
        sw_delay_sec = K_DM_SEC * dm_sw / (freq_bary_mhz ** 2)
    else:
        sw_delay_sec = np.zeros(len(tdb_mjd))

    # Compute residuals
    if verbose: print(f"\n7. Computing phase residuals...")
    delay_sec = total_delay_sec

    # Spin parameters (high precision)
    F0 = get_longdouble(params, 'F0')
    F1 = get_longdouble(params, 'F1', default=0.0)
    F2 = get_longdouble(params, 'F2', default=0.0)
    PEPOCH = get_longdouble(params, 'PEPOCH')

    # Pre-compute coefficients
    F1_half = F1 / np.longdouble(2.0)
    F2_sixth = F2 / np.longdouble(6.0)
    PEPOCH_sec = PEPOCH * np.longdouble(SECS_PER_DAY)

    # Time at emission (TDB - all delays)
    tdb_mjd_ld = np.array(tdb_mjd, dtype=np.longdouble)
    tdb_sec = tdb_mjd_ld * np.longdouble(SECS_PER_DAY)
    dt_sec = tdb_sec - PEPOCH_sec - delay_sec

    # Compute phase using Horner's method
    phase = dt_sec * (F0 + dt_sec * (F1_half + dt_sec * F2_sixth))

    # TZR phase offset (if specified)
    tzr_phase = np.longdouble(0.0)
    if 'TZRMJD' in params:
        if verbose: print(f"\n   Computing TZR phase at TZRMJD...")
        TZRMJD_raw = get_longdouble(params, 'TZRMJD')
        
        # Handle TZRSITE - use the TZR site for TZR-specific calculations
        tzr_site = params.get('TZRSITE', observatory).lower()
        tzr_obs_itrf_km = OBSERVATORIES.get(tzr_site)
        if tzr_obs_itrf_km is None:
            if verbose: print(f"   ⚠️  Unknown TZRSITE '{tzr_site}', falling back to '{observatory}'")
            tzr_obs_itrf_km = obs_itrf_km
            tzr_location = location
        else:
            tzr_location = EarthLocation.from_geocentric(
                tzr_obs_itrf_km[0] * u.km,
                tzr_obs_itrf_km[1] * u.km,
                tzr_obs_itrf_km[2] * u.km
            )
        
        # Resolve TZRMJD timescale - unified with par timescale
        tzrmjd_scale_upper = tzrmjd_scale.upper()
        
        # AUTO: derive from par timescale (single source of truth)
        if tzrmjd_scale_upper == "AUTO":
            # par_timescale is already validated (TCB/TT would have failed by now)
            tzrmjd_scale_resolved = par_timescale  # Will be "TDB"
            if verbose: print(f"   TZRMJD scale: AUTO -> {tzrmjd_scale_resolved} (from par UNITS)")
        elif tzrmjd_scale_upper in ("TDB", "UTC"):
            tzrmjd_scale_resolved = tzrmjd_scale_upper
            if verbose: print(f"   TZRMJD scale: {tzrmjd_scale_resolved} (explicit override)")
        else:
            raise ValueError(f"Invalid tzrmjd_scale '{tzrmjd_scale}'. Must be 'AUTO', 'TDB', or 'UTC'.")
        
        # Apply the resolved timescale
        if tzrmjd_scale_resolved == "TDB":
            # TZRMJD is already in TDB (standard for PINT/Tempo2 TDB par files)
            TZRMJD_TDB = TZRMJD_raw
            delta_tzr_sec = 0.0
            if verbose: print(f"   TZRMJD treated as TDB (no conversion)")
        elif tzrmjd_scale_resolved == "UTC":
            # Legacy behavior: convert UTC to TDB via clock chain
            # Warn loudly if this contradicts UNITS=TDB
            if par_timescale == "TDB":
                print(f"   ⚠️  WARNING: tzrmjd_scale='UTC' contradicts par file UNITS=TDB!")
                print(f"       This will apply a ~69s UTC->TDB conversion to TZRMJD.")
                print(f"       If your par file has UNITS=TDB, TZRMJD should already be in TDB.")
                print(f"       Use tzrmjd_scale='AUTO' (default) or 'TDB' unless you have a")
                print(f"       legacy par file with genuinely UTC TZRMJD values.")
            
            TZRMJD_TDB_ld = compute_tdb_standalone_vectorized(
                [int(TZRMJD_raw)], [float(TZRMJD_raw - int(TZRMJD_raw))],
                mk_clock, gps_clock, bipm_clock, tzr_location
            )[0]
            TZRMJD_TDB = np.longdouble(TZRMJD_TDB_ld)
            delta_tzr_sec = float(TZRMJD_TDB - TZRMJD_raw) * 86400.0
            if verbose: print(f"   TZRMJD converted from UTC to TDB (delta = {delta_tzr_sec:.3f} s)")
            
            # Additional warning if the shift is large
            if abs(delta_tzr_sec) > 1e-3:
                print(f"   ⚠️  WARNING: Large TZRMJD shift detected ({delta_tzr_sec:.3f} s)!")
                print(f"       This confirms TZRMJD was treated as UTC.")
        
        # Compute all delays at TZRMJD to get the TZR delay
        tzr_tdb_arr = np.array([float(TZRMJD_TDB)])
        
        # Astrometry at TZR (using TZR site location)
        tzr_ssb_obs_pos, tzr_ssb_obs_vel = compute_ssb_obs_pos_vel(tzr_tdb_arr, tzr_obs_itrf_km)
        tzr_L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tzr_tdb_arr)
        tzr_roemer = compute_roemer_delay(tzr_ssb_obs_pos, tzr_L_hat, parallax_mas)[0]
        
        # Sun Shapiro at TZR
        tzr_times = Time(tzr_tdb_arr, format='mjd', scale='tdb')
        with solar_system_ephemeris.set('de440'):
            tzr_sun_pos = get_body_barycentric_posvel('sun', tzr_times)[0].xyz.to(u.km).value.T
        tzr_obs_sun = tzr_sun_pos - tzr_ssb_obs_pos
        tzr_sun_shapiro = compute_shapiro_delay(tzr_obs_sun, tzr_L_hat, T_SUN_SEC)[0]

        # Planet Shapiro at TZR
        tzr_planet_shapiro = 0.0
        if planet_shapiro_enabled:
            with solar_system_ephemeris.set('de440'):
                for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
                    tzr_planet_pos = get_body_barycentric_posvel(planet, tzr_times)[0].xyz.to(u.km).value.T
                    tzr_obs_planet = tzr_planet_pos - tzr_ssb_obs_pos
                    tzr_planet_shapiro += compute_shapiro_delay(tzr_obs_planet, tzr_L_hat, T_PLANET[planet])[0]
        
        tzr_roemer_shapiro = tzr_roemer + tzr_sun_shapiro + tzr_planet_shapiro
        
        # Barycentric frequency at TZR
        if 'TZRFRQ' in params:
            tzr_freq = float(params['TZRFRQ'])
        else:
            tzr_freq = 1400.0  # Default
        tzr_freq_bary = compute_barycentric_freq(np.array([tzr_freq]), tzr_ssb_obs_vel, tzr_L_hat)[0]
        
        # Prepare JAX arrays for TZR
        tzr_tdb_jax = jnp.array([float(TZRMJD_TDB)])
        tzr_freq_bary_jax = jnp.array([tzr_freq_bary])
        tzr_obs_sun_jax = jnp.array(tzr_obs_sun)
        tzr_L_hat_jax = jnp.array(tzr_L_hat)
        tzr_roemer_shapiro_jax = jnp.array([tzr_roemer_shapiro])

        # TZR observer position in light-seconds for DDK
        tzr_obs_pos_ls_jax = jnp.array(tzr_ssb_obs_pos / SPEED_OF_LIGHT_KM_S, dtype=jnp.float64)

        # Compute TZR delay
        tzr_total_delay_jax = compute_total_delay_jax(
            tzr_tdb_jax, tzr_freq_bary_jax, tzr_obs_sun_jax, tzr_L_hat_jax,
            dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
            ne_sw_jax, fd_coeffs_jax, has_fd_jax,
            tzr_roemer_shapiro_jax, has_binary_jax, binary_model_id_jax,
            pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax, pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax,
            ecc_jax, om_jax, t0_jax, omdot_jax, edot_jax, m2_jax, sini_jax, kin_jax, kom_jax, h3_jax, h4_jax, stig_jax,
            fb_coeffs_jax, fb_factorials_jax, fb_epoch_jax, use_fb_jax,
            # DDK Kopeikin parameters
            tzr_obs_pos_ls_jax, px_jax, sin_ra_jax, cos_ra_jax, sin_dec_jax, cos_dec_jax,
            # K96 proper motion parameters
            k96_jax, pmra_rad_per_sec_jax, pmdec_rad_per_sec_jax
        ).block_until_ready()
        
        tzr_delay = np.longdouble(float(tzr_total_delay_jax[0]))
        
        # Debug: Compute individual TZR delay components (outside JAX for debugging)
        # DM delay
        dm_epoch = float(params.get('DMEPOCH', params['PEPOCH']))
        dt_years = (float(TZRMJD_TDB) - dm_epoch) / 365.25
        dm_eff = sum(dm_coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(dm_coeffs)))
        # Note: Use module-level K_DM_SEC constant (already imported)
        tzr_dm_delay = K_DM_SEC * dm_eff / (tzr_freq_bary ** 2)
        
        # Solar wind
        ne_sw = float(params.get('NE_SW', 0.0))
        if ne_sw > 0:
            r_km = np.sqrt(np.sum(tzr_obs_sun**2))
            r_au = r_km / 1.495978707e8
            sun_dir = tzr_obs_sun / r_km
            cos_elong = np.sum(sun_dir * tzr_L_hat[0])
            elong = np.arccos(np.clip(cos_elong, -1.0, 1.0))
            rho = np.pi - elong
            sin_rho = max(np.sin(rho), 1e-10)
            AU_PC = 4.84813681e-6
            geometry_pc = AU_PC * rho / (r_au * sin_rho)
            dm_sw = ne_sw * geometry_pc
            tzr_sw_delay = K_DM_SEC * dm_sw / (tzr_freq_bary ** 2)
        else:
            tzr_sw_delay = 0.0
        
        # FD delay  
        if len(fd_coeffs_jax) > 0 and has_fd_jax:
            log_freq = np.log(tzr_freq_bary / 1000.0)
            # FD formula: sum(FD_i * log(f/1GHz)^i), need trailing 0 for polyval
            tzr_fd_delay = np.polyval(list(fd_coeffs_jax)[::-1] + [0], log_freq)
        else:
            tzr_fd_delay = 0.0
        
        # Binary delay - would need to replicate the whole ELL1 calculation, skip for now
        has_binary = 'PB' in params
        if has_binary:
            tzr_binary_delay = float(tzr_delay) - tzr_roemer_shapiro - tzr_dm_delay - tzr_sw_delay - tzr_fd_delay
        else:
            tzr_binary_delay = 0.0
        
        if verbose: print(f"   TZR delay breakdown:")
        if verbose: print(f"     Roemer+Shapiro: {tzr_roemer_shapiro:.9f} s")
        if verbose: print(f"     DM:             {tzr_dm_delay:.9f} s")
        if verbose: print(f"     Solar wind:     {tzr_sw_delay:.9f} s")
        if verbose: print(f"     FD:             {tzr_fd_delay:.9f} s")
        if verbose: print(f"     Binary:         {tzr_binary_delay:.9f} s")
        if verbose: print(f"     TOTAL:          {float(tzr_delay):.9f} s")
        
        # Compute phase at TZR
        tzr_dt_sec = TZRMJD_TDB * np.longdouble(SECS_PER_DAY) - PEPOCH_sec - tzr_delay
        tzr_phase = F0 * tzr_dt_sec + F1_half * tzr_dt_sec**2 + F2_sixth * tzr_dt_sec**3
        
        if verbose: print(f"   TZRMJD (raw):  {float(TZRMJD_raw):.15f}")
        if verbose: print(f"   TZRMJD (used): {float(TZRMJD_TDB):.15f} (scale={tzrmjd_scale_resolved})")
        if verbose: print(f"   delta_tzr:     {delta_tzr_sec:.6f} s")
        if verbose: print(f"   TZR delay: {float(tzr_delay):.9f} s")
        if verbose: print(f"   TZR phase: {float(tzr_phase):.6f} cycles")

    # Wrap phase to [-0.5, 0.5] cycles (PINT's "nearest" pulse approach)
    # This is done by keeping only the fractional part
    if subtract_tzr:
        # Legacy mode: subtract TZR then wrap
        frac_phase = np.mod(phase - tzr_phase + 0.5, 1.0) - 0.5
    else:
        # PINT-style: wrap to nearest integer pulse (discard integer part)
        # This is what PINT does by default in track_mode="nearest"
        phase_wrapped = phase - np.round(phase)  # Fractional part only
        frac_phase = phase_wrapped

    # Convert to microseconds
    residuals_us = np.asarray(frac_phase / F0 * 1e6, dtype=np.float64)

    # Subtract weighted mean (PINT's default behavior)
    # NOTE: For fitting, we should NOT subtract mean here!
    # The fitter needs the raw wrapped residuals to see parameter errors
    if subtract_tzr:
        # Legacy/display mode: subtract mean for nice plots
        errors_us = np.array([toa.error_us for toa in toas])
        weights = 1.0 / (errors_us ** 2)
        weighted_mean = np.sum(residuals_us * weights) / np.sum(weights)
        residuals_us = residuals_us - weighted_mean
    else:
        # Fitting mode: keep raw wrapped residuals
        # The fitter will handle mean subtraction if needed
        errors_us = np.array([toa.error_us for toa in toas])
        weights = 1.0 / (errors_us ** 2)

    # Compute weighted RMS (chi-squared reduced)
    weighted_rms = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    
    # Also compute unweighted for comparison
    unweighted_rms = np.std(residuals_us)

    # Results
    if verbose: print(f"\n" + "=" * 60)
    if verbose: print(f"Results:")
    if verbose: print(f"  Weighted RMS: {weighted_rms:.3f} μs")
    if verbose: print(f"  Unweighted RMS: {unweighted_rms:.3f} μs")
    if verbose: print(f"  Mean: {np.mean(residuals_us):.3f} μs")
    if verbose: print(f"  Min: {np.min(residuals_us):.3f} μs")
    if verbose: print(f"  Max: {np.max(residuals_us):.3f} μs")
    if verbose: print(f"  N_TOAs: {len(residuals_us)}")
    if verbose: print("=" * 60)

    # Convert ssb_obs_pos from km to light-seconds for astrometry derivatives
    SPEED_OF_LIGHT_KM_S = 299792.458
    ssb_obs_pos_ls = ssb_obs_pos_km / SPEED_OF_LIGHT_KM_S
    
    # Compute pre-binary delay: roemer_shapiro + DM + SW + tropo (NOT FD)
    # This is the PINT-compatible time for binary model evaluation
    prebinary_delay_sec = roemer_shapiro + dm_delay_sec + sw_delay_sec + tropo_delay_sec
    
    return {
        'residuals_us': residuals_us,
        'rms_us': float(weighted_rms),  # Use weighted RMS as primary
        'weighted_rms_us': float(weighted_rms),
        'unweighted_rms_us': float(unweighted_rms),
        'mean_us': float(np.mean(residuals_us)),
        'n_toas': len(residuals_us),
        'tdb_mjd': np.array(tdb_mjd, dtype=np.float64),
        'errors_us': errors_us,
        # Add computed delays for JAX fitting
        'total_delay_sec': np.array(total_delay_sec, dtype=np.float64),
        'freq_bary_mhz': np.array(freq_bary_mhz, dtype=np.float64),
        'tzr_phase': float(tzr_phase),
        # Add emission time (computed with longdouble, converted to float64)
        'dt_sec': np.array(dt_sec, dtype=np.float64),
        # Roemer+Shapiro delay for computing barycentric times (legacy, for backward compat)
        'roemer_shapiro_sec': np.array(roemer_shapiro, dtype=np.float64),
        # Pre-binary delay: roemer_shapiro + DM + SW + tropo (PINT-compatible binary evaluation time)
        'prebinary_delay_sec': np.array(prebinary_delay_sec, dtype=np.float64),
        # Individual delay components (for diagnostics)
        'dm_delay_sec': np.array(dm_delay_sec, dtype=np.float64),
        'sw_delay_sec': np.array(sw_delay_sec, dtype=np.float64),
        'tropo_delay_sec': np.array(tropo_delay_sec, dtype=np.float64),
        # SSB to observatory position in light-seconds (needed for astrometry derivatives)
        'ssb_obs_pos_ls': np.array(ssb_obs_pos_ls, dtype=np.float64),
    }
