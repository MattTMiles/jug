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

# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()
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
    clock_dir: Path | str | None = None,
    observatory: str = "meerkat",
    subtract_tzr: bool = True,
    verbose: bool = True
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
    has_binary = 'PB' in params
    binary_model = params.get('BINARY', 'NONE').upper() if has_binary else 'NONE'
    
    if verbose: print(f"\n5. Detecting binary model...")
    if has_binary:
        if verbose: print(f"   Binary model: {binary_model}")
    else:
        if verbose: print(f"   No binary companion")
    
    # Check if we're using ELL1 (inline) or need to dispatch to BT/DD/T2
    use_ell1_inline = binary_model in ('ELL1', 'ELL1H', 'NONE') and has_binary
    use_dispatch = binary_model in ('BT', 'BTX', 'DD', 'DDH', 'DDGR', 'DDK', 'T2')
    
    if use_dispatch:
        # For BT/DD/T2 models, we need to compute delays differently
        # These models use ECC/OM/T0 instead of TASC/EPS1/EPS2
        from jug.delays.binary_dispatch import dispatch_binary_delay
        
        if verbose: print(f"   Using dispatched binary model: {binary_model}")
        
        # Build parameter dict for dispatcher
        binary_params = {
            'PB': float(params['PB']),
            'A1': float(params['A1']),
            'ECC': float(params.get('ECC', 0.0)),
            'OM': float(params.get('OM', 0.0)),
            'T0': float(params.get('T0', params.get('PEPOCH', 0.0))),
            'GAMMA': float(params.get('GAMMA', 0.0)),
            'PBDOT': float(params.get('PBDOT', 0.0)),
            'XDOT': float(params.get('XDOT', 0.0)),
            'OMDOT': float(params.get('OMDOT', 0.0)),
            'EDOT': float(params.get('EDOT', 0.0))
        }
        
        # Handle SINI - may be numeric or reference to KIN
        sini_value = params.get('SINI', 0.0)
        if isinstance(sini_value, str) and sini_value.upper() == 'KIN':
            # SINI references KIN parameter
            binary_params['SINI'] = float(params.get('KIN', 0.0))
        else:
            binary_params['SINI'] = float(sini_value)
        
        # Handle M2
        binary_params['M2'] = float(params.get('M2', 0.0))
        
        # Handle H3/H4/STIG for DDH (orthometric Shapiro delay parameters)
        binary_params['H3'] = float(params.get('H3', 0.0)) if 'H3' in params else None
        binary_params['H4'] = float(params.get('H4', 0.0)) if 'H4' in params else None
        binary_params['STIG'] = float(params.get('STIG', 0.0)) if 'STIG' in params else None
        
        # For T2, also need KIN/KOM
        if binary_model == 'T2':
            binary_params['KIN'] = float(params.get('KIN', 0.0))
            binary_params['KOM'] = float(params.get('KOM', 0.0))
        
        # Compute binary delays using dispatcher
        # Binary delays need topocentric time = TDB - (Roemer+Shapiro+DM+SW+FD)
        # We need to iterate: use TDB as first guess, compute other delays, then recompute binary
        if verbose: print(f"   Computing {len(tdb_mjd)} binary delays (iterative)...")
        
        # Call dispatcher with array (will handle vectorization internally or via loop)
        from jug.delays.binary_dd import dd_binary_delay_vectorized
        from jug.delays.binary_t2 import t2_binary_delay_vectorized
        
        # Iteration 1: Use TDB as first approximation
        tdb_array_f64 = jnp.array(np.array(tdb_mjd, dtype=np.float64))
        
        if binary_model in ('DD', 'DDH', 'DDGR', 'DDK'):
            # Use vectorized DD function directly
            binary_delay_sec = np.array(dd_binary_delay_vectorized(
                tdb_array_f64,
                pb_days=binary_params['PB'],
                a1_lt_sec=binary_params['A1'],
                ecc=binary_params['ECC'],
                omega_deg=binary_params['OM'],
                t0_mjd=binary_params['T0'],
                gamma_sec=binary_params['GAMMA'],
                pbdot=binary_params['PBDOT'],
                xdot=binary_params['XDOT'],
                omdot_deg_yr=binary_params['OMDOT'],
                edot=binary_params['EDOT'],
                m2_msun=binary_params['M2'],
                sini=binary_params['SINI'],
                h3_sec=binary_params['H3'],
                h4_sec=binary_params['H4'],
                stig=binary_params['STIG']
            ))
            # DEBUG: Log binary delay statistics after iteration 1
            if verbose: print(f"   DEBUG Iter1: Binary delays computed - range [{np.min(binary_delay_sec):.3f}, {np.max(binary_delay_sec):.3f}] s, mean={np.mean(binary_delay_sec):.3f} s, std={np.std(binary_delay_sec):.3f} s")
        elif binary_model == 'T2':
            # Use vectorized T2 function directly
            binary_delay_sec = np.array(t2_binary_delay_vectorized(
                tdb_array_f64,
                pb=binary_params['PB'],
                a1=binary_params['A1'],
                ecc=binary_params['ECC'],
                om=binary_params['OM'],
                t0=binary_params['T0'],
                gamma=binary_params['GAMMA'],
                pbdot=binary_params['PBDOT'],
                xdot=binary_params['XDOT'],
                edot=binary_params['EDOT'],
                omdot=binary_params['OMDOT'],
                m2=binary_params['M2'],
                sini=binary_params['SINI'],
                kin=binary_params.get('KIN', 0.0),
                kom=binary_params.get('KOM', 0.0)
            ))
            # DEBUG: Log binary delay statistics after T2 iteration 1
            if verbose: print(f"   DEBUG Iter1: Binary delays computed - range [{np.min(binary_delay_sec):.3f}, {np.max(binary_delay_sec):.3f}] s, mean={np.mean(binary_delay_sec):.3f} s, std={np.std(binary_delay_sec):.3f} s")
        elif binary_model in ('BT', 'BTX'):
            # Use vectorized BT function
            from jug.delays.binary_bt import bt_binary_delay_vectorized
            binary_delay_sec = np.array(bt_binary_delay_vectorized(
                tdb_array_f64,
                pb=binary_params['PB'],
                a1=binary_params['A1'],
                ecc=binary_params['ECC'],
                om=binary_params['OM'],
                t0=binary_params['T0'],
                gamma=binary_params['GAMMA'],
                pbdot=binary_params['PBDOT'],
                m2=binary_params['M2'],
                sini=binary_params['SINI'],
                omdot=binary_params['OMDOT'],
                xdot=binary_params['XDOT']
            ))
            # DEBUG: Log binary delay statistics after BT iteration 1
            if verbose: print(f"   DEBUG Iter1: Binary delays computed - range [{np.min(binary_delay_sec):.3f}, {np.max(binary_delay_sec):.3f}] s, mean={np.mean(binary_delay_sec):.3f} s, std={np.std(binary_delay_sec):.3f} s")
        else:
            raise ValueError(f"Unsupported binary model: {binary_model}")
        
        # Now we need to iterate: compute Roemer+Shapiro+DM+SW+FD first,
        # then recompute binary at topocentric time
        # For now, store first approximation - we'll refine after getting other delays
        binary_delay_approx1 = binary_delay_sec.copy()
        
        # Set up for JAX kernel with binary delays pre-computed
        has_binary_jax = jnp.array(False)  # Tell JAX kernel we handled binary externally
        binary_delay_external = binary_delay_sec  # We'll add this back later
        
        # Dummy ELL1 params for JAX kernel
        pb_jax = a1_jax = tasc_jax = eps1_jax = eps2_jax = jnp.array(0.0)
        pbdot_jax = xdot_jax = gamma_jax = r_shap_jax = s_shap_jax = jnp.array(0.0)
        
    elif use_ell1_inline or has_binary:
        # ELL1/ELL1H: use inline computation (current code path)
        if verbose: print(f"   Using inline ELL1 computation")
        
        has_binary_jax = jnp.array(has_binary)
        binary_delay_external = None  # No external binary delay
        
        if has_binary:
            pb_jax = jnp.array(float(params['PB']))
            a1_jax = jnp.array(float(params['A1']))
            tasc_jax = jnp.array(float(params.get('TASC', 0.0)))
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
    else:
        # No binary
        has_binary_jax = jnp.array(False)
        binary_delay_external = None
        pb_jax = a1_jax = tasc_jax = eps1_jax = eps2_jax = jnp.array(0.0)
        pbdot_jax = xdot_jax = gamma_jax = r_shap_jax = s_shap_jax = jnp.array(0.0)

    # Compute total delay (DM + SW + FD + inline binary if ELL1)
    if verbose: print(f"\n6. Running JAX delay kernel...")
    total_delay_jax = compute_total_delay_jax(
        tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
        dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
        ne_sw_jax, fd_coeffs_jax, has_fd_jax,
        roemer_shapiro_jax, has_binary_jax,
        pb_jax, a1_jax, tasc_jax, eps1_jax, eps2_jax,
        pbdot_jax, xdot_jax, gamma_jax, r_shap_jax, s_shap_jax
    ).block_until_ready()
    
    # Add external binary delay if we used dispatcher
    if binary_delay_external is not None:
        if verbose: print(f"   Refining binary delays (iteration 2)...")
        # Iteration 2: Now we have Roemer+Shapiro+DM+SW+FD, compute topocentric time
        # t_topo = TDB - (other_delays) / SECS_PER_DAY
        other_delays_sec = np.asarray(total_delay_jax, dtype=np.float64)
        t_topo_mjd = tdb_mjd - other_delays_sec / SECS_PER_DAY
        
        # Recompute binary delay at topocentric time
        t_topo_f64 = jnp.array(np.array(t_topo_mjd, dtype=np.float64))
        
        if binary_model in ('DD', 'DDH', 'DDGR', 'DDK'):
            binary_delay_sec = np.array(dd_binary_delay_vectorized(
                t_topo_f64,
                pb_days=binary_params['PB'],
                a1_lt_sec=binary_params['A1'],
                ecc=binary_params['ECC'],
                omega_deg=binary_params['OM'],
                t0_mjd=binary_params['T0'],
                gamma_sec=binary_params['GAMMA'],
                pbdot=binary_params['PBDOT'],
                xdot=binary_params['XDOT'],
                omdot_deg_yr=binary_params['OMDOT'],
                edot=binary_params['EDOT'],
                m2_msun=binary_params['M2'],
                sini=binary_params['SINI'],
                h3_sec=binary_params['H3'],
                h4_sec=binary_params['H4'],
                stig=binary_params['STIG']
            ))
        elif binary_model == 'T2':
            binary_delay_sec = np.array(t2_binary_delay_vectorized(
                t_topo_f64,
                pb=binary_params['PB'],
                a1=binary_params['A1'],
                ecc=binary_params['ECC'],
                om=binary_params['OM'],
                t0=binary_params['T0'],
                gamma=binary_params['GAMMA'],
                pbdot=binary_params['PBDOT'],
                xdot=binary_params['XDOT'],
                edot=binary_params['EDOT'],
                omdot=binary_params['OMDOT'],
                m2=binary_params['M2'],
                sini=binary_params['SINI'],
                kin=binary_params.get('KIN', 0.0),
                kom=binary_params.get('KOM', 0.0)
            ))
        # BT doesn't need iteration 2 (we'll skip for now)
        
        if verbose: print(f"   Binary delay change: {np.mean(np.abs(binary_delay_sec - binary_delay_approx1))*1e6:.3f} μs")
        
        # DEBUG: Log binary delay statistics after iteration 2
        if verbose: print(f"   DEBUG Iter2: Binary delays refined - range [{np.min(binary_delay_sec):.3f}, {np.max(binary_delay_sec):.3f}] s, mean={np.mean(binary_delay_sec):.3f} s, std={np.std(binary_delay_sec):.3f} s")
        
        total_delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble) + binary_delay_sec
        
        # DEBUG: Log total delay after adding binary
        if verbose: print(f"   DEBUG: Total delay after adding binary - range [{np.min(total_delay_sec):.3f}, {np.max(total_delay_sec):.3f}] s, mean={np.mean(total_delay_sec):.3f} s")
        if verbose: print(f"   DEBUG: total_delay_jax before adding binary - range [{np.min(total_delay_jax):.3f}, {np.max(total_delay_jax):.3f}] s")
    else:
        total_delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble)

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
        
        if verbose: print(f"   TZRMJD: {TZRMJD_UTC:.6f} UTC -> {TZRMJD_TDB:.6f} TDB")
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
        # Roemer+Shapiro delay for computing barycentric times (needed for binary fitting)
        'roemer_shapiro_sec': np.array(roemer_shapiro, dtype=np.float64),
        # SSB to observatory position in light-seconds (needed for astrometry derivatives)
        'ssb_obs_pos_ls': np.array(ssb_obs_pos_ls, dtype=np.float64),
    }
