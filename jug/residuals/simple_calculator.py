"""Simple end-to-end residual calculator for testing JUG.

This module provides a minimal interface to compute pulsar timing residuals
without the full complexity of the complete calculator class.
"""

import math
from pathlib import Path
import numpy as np
from astropy.coordinates import EarthLocation, get_body_barycentric_posvel, solar_system_ephemeris
from astropy.time import Time
from astropy import units as u

# Enable JAX 64-bit precision
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file, get_longdouble, parse_ra, parse_dec
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
    SECS_PER_DAY, T_SUN_SEC, T_PLANET, OBSERVATORIES
)


def compute_residuals_simple(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Path | str = "data/clock",
    observatory: str = "meerkat"
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
    clock_dir : Path or str, optional
        Directory containing clock files (default: "data/clock")
    observatory : str, optional
        Observatory name (default: "meerkat")

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
    print("=" * 60)
    print("JUG Simple Residual Calculator")
    print("=" * 60)

    # Parse files
    print(f"\n1. Loading files...")
    params = parse_par_file(par_file)
    toas = parse_tim_file_mjds(tim_file)
    print(f"   Loaded {len(toas)} TOAs from {Path(tim_file).name}")
    print(f"   Loaded timing model from {Path(par_file).name}")

    # Load clock files
    print(f"\n2. Loading clock corrections...")
    clock_dir = Path(clock_dir)
    mk_clock = parse_clock_file(clock_dir / "mk2utc.clk")
    gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")
    bipm_clock = parse_clock_file(clock_dir / "tai2tt_bipm2024.clk")
    print(f"   Loaded 3 clock files (using BIPM2024)")

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
    print(f"\n3. Computing TDB (standalone, no PINT)...")
    mjd_ints = [toa.mjd_int for toa in toas]
    mjd_fracs = [toa.mjd_frac for toa in toas]

    tdb_mjd = compute_tdb_standalone_vectorized(
        mjd_ints, mjd_fracs,
        mk_clock, gps_clock, bipm_clock,
        location
    )
    print(f"   Computed TDB for {len(tdb_mjd)} TOAs")

    # Astrometry
    print(f"\n4. Computing astrometric delays...")
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
        print(f"   Computing planetary Shapiro delays...")
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
    print(f"\n5. Running JAX delay kernel...")
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

    # Binary parameters
    has_binary = 'PB' in params
    has_binary_jax = jnp.array(has_binary)
    if has_binary:
        pb_jax = jnp.array(float(params['PB']))
        a1_jax = jnp.array(float(params['A1']))
        tasc_jax = jnp.array(float(params['TASC']))
        eps1_jax = jnp.array(float(params.get('EPS1', 0.0)))
        eps2_jax = jnp.array(float(params.get('EPS2', 0.0)))
        pbdot_jax = jnp.array(float(params.get('PBDOT', 0.0)))
        xdot_jax = jnp.array(float(params.get('XDOT', 0.0)))
        gamma_jax = jnp.array(float(params.get('GAMMA', 0.0)))

        # Shapiro delay parameters
        # Handle both H3/STIG (orthometric) and M2/SINI (mass/inclination) parameterizations
        if 'H3' in params and 'STIG' in params:
            # Orthometric parameters (H3, STIG)
            H3 = float(params['H3'])
            STIG = float(params['STIG'])
            r_shap_jax = jnp.array(H3)
            s_shap_jax = jnp.array(STIG)
        elif 'M2' in params and 'SINI' in params:
            # Convert M2/SINI to r/s
            # r = TSUN * M2 (where TSUN = G*Msun/c^3 = 4.925490947e-6 s)
            # s = SINI
            M2 = float(params['M2'])  # solar masses
            SINI = float(params['SINI'])
            r_shap_jax = jnp.array(T_SUN_SEC * M2)
            s_shap_jax = jnp.array(SINI)
        else:
            # No Shapiro delay parameters
            r_shap_jax = jnp.array(0.0)
            s_shap_jax = jnp.array(0.0)
    else:
        # Default binary params (unused)
        pb_jax = a1_jax = tasc_jax = eps1_jax = eps2_jax = jnp.array(0.0)
        pbdot_jax = xdot_jax = gamma_jax = r_shap_jax = s_shap_jax = jnp.array(0.0)

    # Compute total delay
    total_delay_jax = compute_total_delay_jax(
        tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
        dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
        ne_sw_jax, fd_coeffs_jax, has_fd_jax,
        roemer_shapiro_jax, has_binary_jax,
        pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax,
        pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax
    ).block_until_ready()

    # Compute residuals
    print(f"\n6. Computing phase residuals...")
    delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble)

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
        print(f"\n   Computing TZR phase at TZRMJD...")
        TZRMJD_UTC = get_longdouble(params, 'TZRMJD')
        
        # Convert TZRMJD from UTC to TDB
        TZRMJD_TDB_ld = compute_tdb_standalone_vectorized(
            [int(TZRMJD_UTC)], [float(TZRMJD_UTC - int(TZRMJD_UTC))],
            mk_clock, gps_clock, bipm_clock, location
        )[0]
        TZRMJD_TDB = np.longdouble(TZRMJD_TDB_ld)
        
        # Compute all delays at TZRMJD to get the TZR delay
        tzr_tdb_arr = np.array([float(TZRMJD_TDB)])
        
        # Astrometry at TZR
        tzr_ssb_obs_pos, tzr_ssb_obs_vel = compute_ssb_obs_pos_vel(tzr_tdb_arr, obs_itrf_km)
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
        
        # Compute TZR delay
        tzr_total_delay_jax = compute_total_delay_jax(
            tzr_tdb_jax, tzr_freq_bary_jax, tzr_obs_sun_jax, tzr_L_hat_jax,
            dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
            ne_sw_jax, fd_coeffs_jax, has_fd_jax,
            tzr_roemer_shapiro_jax, has_binary_jax,
            pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax,
            pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax
        ).block_until_ready()
        
        tzr_delay = np.longdouble(float(tzr_total_delay_jax[0]))
        
        # Debug: Compute individual TZR delay components (outside JAX for debugging)
        # DM delay
        dm_epoch = float(params.get('DMEPOCH', params['PEPOCH']))
        dt_years = (float(TZRMJD_TDB) - dm_epoch) / 365.25
        dm_eff = sum(dm_coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(dm_coeffs)))
        K_DM_SEC = 1.0 / 2.41e-4
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
            tzr_fd_delay = np.polyval(list(fd_coeffs_jax)[::-1], log_freq)
        else:
            tzr_fd_delay = 0.0
        
        # Binary delay - would need to replicate the whole ELL1 calculation, skip for now
        has_binary = 'PB' in params
        if has_binary:
            tzr_binary_delay = float(tzr_delay) - tzr_roemer_shapiro - tzr_dm_delay - tzr_sw_delay - tzr_fd_delay
        else:
            tzr_binary_delay = 0.0
        
        print(f"   TZR delay breakdown:")
        print(f"     Roemer+Shapiro: {tzr_roemer_shapiro:.9f} s")
        print(f"     DM:             {tzr_dm_delay:.9f} s")
        print(f"     Solar wind:     {tzr_sw_delay:.9f} s")
        print(f"     FD:             {tzr_fd_delay:.9f} s")
        print(f"     Binary:         {tzr_binary_delay:.9f} s")
        print(f"     TOTAL:          {float(tzr_delay):.9f} s")
        
        # Compute phase at TZR
        tzr_dt_sec = TZRMJD_TDB * np.longdouble(SECS_PER_DAY) - PEPOCH_sec - tzr_delay
        tzr_phase = F0 * tzr_dt_sec + F1_half * tzr_dt_sec**2 + F2_sixth * tzr_dt_sec**3
        
        print(f"   TZRMJD: {TZRMJD_UTC:.6f} UTC -> {TZRMJD_TDB:.6f} TDB")
        print(f"   TZR delay: {float(tzr_delay):.9f} s")
        print(f"   TZR phase: {float(tzr_phase):.6f} cycles")

    # Wrap phase to [-0.5, 0.5] cycles
    frac_phase = np.mod(phase - tzr_phase + 0.5, 1.0) - 0.5

    # Convert to microseconds
    residuals_us = np.asarray(frac_phase / F0 * 1e6, dtype=np.float64)

    # Subtract weighted mean (matching PINT behavior)
    errors_us = np.array([toa.error_us for toa in toas])
    weights = 1.0 / (errors_us ** 2)
    weighted_mean = np.sum(residuals_us * weights) / np.sum(weights)
    residuals_us = residuals_us - weighted_mean

    # Compute weighted RMS (chi-squared reduced)
    weighted_rms = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))
    
    # Also compute unweighted for comparison
    unweighted_rms = np.std(residuals_us)

    # Results
    print(f"\n" + "=" * 60)
    print(f"Results:")
    print(f"  Weighted RMS: {weighted_rms:.3f} μs")
    print(f"  Unweighted RMS: {unweighted_rms:.3f} μs")
    print(f"  Mean: {np.mean(residuals_us):.3f} μs")
    print(f"  Min: {np.min(residuals_us):.3f} μs")
    print(f"  Max: {np.max(residuals_us):.3f} μs")
    print(f"  N_TOAs: {len(residuals_us)}")
    print("=" * 60)

    return {
        'residuals_us': residuals_us,
        'rms_us': float(weighted_rms),  # Use weighted RMS as primary
        'weighted_rms_us': float(weighted_rms),
        'unweighted_rms_us': float(unweighted_rms),
        'mean_us': float(np.mean(residuals_us)),
        'n_toas': len(residuals_us),
        'tdb_mjd': np.array(tdb_mjd, dtype=np.float64),
        'errors_us': errors_us
    }
