"""Core residual calculator for JUG.

This module computes pulsar timing residuals from .par and .tim files,
including all delay corrections (barycentric, DM, solar wind, binary,
tropospheric, FD). It is the primary computation engine used by the
session, API, and fitter modules.
"""

import math
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric_posvel
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
from astropy import units as u

# Ensure JAX is configured for x64 precision
from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

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
    SECS_PER_DAY, SECS_PER_YEAR, T_SUN_SEC, T_PLANET, OBSERVATORIES, K_DM_SEC,
    C_KM_S, MAS_PER_RAD, AU_KM, AU_PC
)

# Old JPL ephemerides moved to a_old_versions/ on NAIF server
_OLD_EPHEM_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels"
    "/spk/planets/a_old_versions/{name}.bsp"
)
_OLD_EPHEMERIDES = {'de200', 'de405', 'de410', 'de414', 'de421', 'de423', 'de430'}

# Ephemerides available from JPL SSD server (not NAIF)
_SSD_EPHEMERIDES = {
    'de436': 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de436.bsp',
    'de441': 'https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de441.bsp',
}

# Current recommended default
_DEFAULT_EPHEMERIS = 'de440'

# Observatory-to-clock-file mapping
_OBS_CLOCK_FILES = {
    'meerkat': 'mk2utc.clk',
    'ao': 'ao2gps.clk', 'arecibo': 'ao2gps.clk', '3': 'ao2gps.clk',
    'gbt': 'gbt2gps.clk', '1': 'gbt2gps.clk', 'gb': 'gbt2gps.clk',
    'parkes': 'pks2gps.clk', 'pks': 'pks2gps.clk', 'pk': 'pks2gps.clk', '7': 'pks2gps.clk',
    'jb': 'jb2gps.clk', 'jodrell': 'jb2gps.clk', '8': 'jb2gps.clk',
    'ef': 'eff2gps.clk', 'eff': 'eff2gps.clk', 'effelsberg': 'eff2gps.clk', 'g': 'eff2gps.clk',
    'effix': 'effix2gps.clk',
    'leap': ['leap2effix.clk', 'effix2gps.clk'],
    'nc': 'ncy2gps.clk', 'ncy': 'ncy2gps.clk', 'nancay': 'ncy2gps.clk', 'f': 'ncy2gps.clk',
    'ncyobs': ['ncyobs2obspm.clk', 'obspm2gps.clk'],
    'wsrt': 'wsrt2gps.clk', 'we': 'wsrt2gps.clk', 'i': 'wsrt2gps.clk',
    'vla': 'vla2gps.clk', 'vl': 'vla2gps.clk',
    'jbroach': ['jbroach2jb.clk', 'jb2gps.clk'],
    'jbdfb': ['jbdfb2jb.clk', 'jb2gps.clk'],
    'jbmk2roach': 'jb2gps.clk',
    'gmrt': 'gmrt2gps.clk',
}


# Parameters recognized by JUG (from PARAMETER_REGISTRY + metadata/directives)
_KNOWN_PAR_KEYWORDS = None  # Lazily initialized

def _get_known_par_keywords():
    """Return the set of all par file keywords JUG recognizes."""
    global _KNOWN_PAR_KEYWORDS
    if _KNOWN_PAR_KEYWORDS is not None:
        return _KNOWN_PAR_KEYWORDS
    from jug.model.parameter_spec import PARAMETER_REGISTRY, _ALIAS_MAP
    known = set(PARAMETER_REGISTRY.keys()) | set(_ALIAS_MAP.keys())
    # Metadata/directives that are valid but not timing parameters
    known |= {
        'PSRJ', 'PSRB', 'PSR', 'PSRNAME',
        'EPHEM', 'EPHVER', 'CLK', 'UNITS', 'TIMEEPH', 'T2CMETHOD',
        'MODE', 'NITS', 'NTOA', 'TRES', 'CHI2R',
        'START', 'FINISH', 'TRACK',
        'TZRMJD', 'TZRFRQ', 'TZRSITE',
        'BINARY', 'INFO', 'PLANET_SHAPIRO',
        'CORRECT_TROPOSPHERE', 'K96',
        'RNAMP', 'RNIDX', 'TNRedAmp', 'TNRedGam', 'TNRedC',
        'TNDMAMP', 'TNDMGam', 'TNDMC',
        'TNSUBTRACTDM', 'TNSUBTRACTPOLY',
        'DMMODEL', 'CONSTRAIN', 'NCOEFF',
        'RM', 'SWM', 'SOLARN0',
    }
    # DMX ranges: DMX_nnnn, DMXR1_nnnn, DMXR2_nnnn, DMXF1_nnnn, DMXF2_nnnn, DMXEP_nnnn
    _KNOWN_PAR_KEYWORDS = known
    return known


def _is_known_param(key):
    """Check if a par file keyword is recognized by JUG."""
    known = _get_known_par_keywords()
    if key in known:
        return True
    # Dynamic parameter patterns: DMX_nnnn, DMXR1_nnnn, JUMP1, FBn, FDn, etc.
    import re
    if re.match(r'^(DMX|DMXR1|DMXR2|DMXF1|DMXF2|DMXEP)_\d+$', key):
        return True
    if re.match(r'^(JUMP|CM)\d+$', key):
        return True
    return False


def _warn_unrecognized_params(params, verbose=True):
    """Warn about par file parameters not recognized by JUG."""
    unknown = []
    for key in params:
        if key.startswith('_'):  # Internal metadata keys
            continue
        if not _is_known_param(key):
            unknown.append(key)
    if unknown:
        # Bold yellow via ANSI escape codes
        BOLD_YELLOW = '\033[1;33m'
        RESET = '\033[0m'
        msg = (f"{BOLD_YELLOW}[!] Unrecognized par file parameters "
               f"(ignored by JUG): {', '.join(sorted(unknown))}{RESET}")
        print(msg)
    return unknown


def _resolve_ephemeris(name: str) -> str:
    """Resolve ephemeris name to a value astropy can use.

    For old JPL ephemerides that have been moved on the NAIF server,
    downloads from the archive URL and returns the cached file path.
    
    For ephemerides available on JPL SSD server (e.g., DE436, DE441),
    downloads from SSD and returns the cached file path.
    
    Falls back to DE440 if download fails.
    """
    import re
    import sys
    
    # Check if ephemeris is available from SSD server
    if name.lower() in _SSD_EPHEMERIDES:
        from astropy.utils.data import download_file, is_url_in_cache
        url = _SSD_EPHEMERIDES[name.lower()]
        try:
            if not is_url_in_cache(url):
                print(f"Downloading {name.upper()} from JPL SSD server...", file=sys.stderr)
                path = download_file(url, cache=True)
                print(f"[x] {name.upper()} downloaded successfully.", file=sys.stderr)
            else:
                path = download_file(url, cache=True)
            return path
        except Exception as e:
            print(f"\n{'='*70}", file=sys.stderr)
            print(f"WARNING: Could not download {name.upper()} from JPL SSD server.", file=sys.stderr)
            print(f"         Error: {e}", file=sys.stderr)
            print(f"         Falling back to {_DEFAULT_EPHEMERIS.upper()}.", file=sys.stderr)
            print(f"{'='*70}\n", file=sys.stderr)
            return _DEFAULT_EPHEMERIS
    
    if not re.match(r'de\d{3}s?$', name.lower()):
        return name
    if name.lower() not in _OLD_EPHEMERIDES:
        return name
    # Try standard astropy path first
    try:
        solar_system_ephemeris.validate(name)
        return name
    except Exception:
        pass
    # Download from old versions URL
    from astropy.utils.data import download_file
    url = _OLD_EPHEM_URL.format(name=name.lower())
    try:
        path = download_file(url, cache=True)
        return path
    except Exception:
        # If old ephemeris download fails, fall back to default with warning
        print(f"\n{'='*70}", file=sys.stderr)
        print(f"WARNING: Could not download ephemeris {name.upper()}.", file=sys.stderr)
        print(f"         Falling back to {_DEFAULT_EPHEMERIS.upper()}.", file=sys.stderr)
        print(f"{'='*70}\n", file=sys.stderr)
        return _DEFAULT_EPHEMERIS


def compute_phase_residuals(dt_sec_ld, params, weights, subtract_mean=True,
                            tzr_phase=None, tdb_sec_ld=None, jump_phase=None):
    """Compute phase residuals from emission-time offsets (canonical implementation).

    This is the single shared function used by both the evaluate-only and fitter
    codepaths to guarantee identical phase computation, wrapping, and conversion.

    Parameters
    ----------
    dt_sec_ld : np.ndarray (longdouble)
        Time since PEPOCH minus all delays, in seconds.
        Must be longdouble to preserve phase precision for large |dt|.
    params : dict
        Timing model parameters (needs F0, F1, F2).
    weights : np.ndarray (float64)
        1/sigma^2 weights for weighted mean subtraction.
    subtract_mean : bool
        Whether to subtract weighted mean from residuals.
    tzr_phase : float or longdouble, optional
        Phase at the TZR reference point. If provided, subtracted from each
        TOA's phase before wrapping to ensure correct pulse numbering.
    tdb_sec_ld : np.ndarray (longdouble), optional
        TDB times in seconds (longdouble). Required for glitch computation.
        If None, glitch contributions are not computed.

    Returns
    -------
    residuals_us : np.ndarray (float64)
        Residuals in microseconds.
    residuals_sec : np.ndarray (float64)
        Residuals in seconds.
    """
    F0 = get_longdouble(params, 'F0')
    F1 = get_longdouble(params, 'F1', default=0.0)
    F2 = get_longdouble(params, 'F2', default=0.0)
    F1_half = F1 / np.longdouble(2.0)
    F2_sixth = F2 / np.longdouble(6.0)

    dt = np.asarray(dt_sec_ld, dtype=np.longdouble)

    # Phase using Horner's method (longdouble precision)
    phase = dt * (F0 + dt * (F1_half + dt * F2_sixth))

    # Glitch contributions
    # Glitch phase is computed at TDB (not emission time) following PINT/Tempo2 convention.
    PEPOCH = get_longdouble(params, 'PEPOCH')
    PEPOCH_sec = PEPOCH * np.longdouble(SECS_PER_DAY)
    glitch_idx = 1
    while f'GLEP_{glitch_idx}' in params:
        glep = get_longdouble(params, f'GLEP_{glitch_idx}')
        glep_sec = glep * np.longdouble(SECS_PER_DAY)
        glph = get_longdouble(params, f'GLPH_{glitch_idx}', default=0.0)
        glf0 = get_longdouble(params, f'GLF0_{glitch_idx}', default=0.0)
        glf1 = get_longdouble(params, f'GLF1_{glitch_idx}', default=0.0)
        glf0d = get_longdouble(params, f'GLF0D_{glitch_idx}', default=0.0)
        gltd = get_longdouble(params, f'GLTD_{glitch_idx}', default=0.0)

        # dt_glitch is time since PEPOCH (matching PINT's convention)
        # The glitch activates for t > GLEP
        dt_glitch = dt  # emission time relative to PEPOCH
        glep_dt = (glep_sec - PEPOCH_sec)  # GLEP offset from PEPOCH
        active = dt_glitch > glep_dt
        dt_since_glep = np.where(active, dt_glitch - glep_dt, np.longdouble(0.0))

        glitch_phase = (glph
                       + glf0 * dt_since_glep
                       + np.longdouble(0.5) * glf1 * dt_since_glep**2)

        # Exponential recovery term
        if gltd != 0.0 and glf0d != 0.0:
            gltd_sec = gltd * np.longdouble(SECS_PER_DAY)
            glitch_phase += glf0d * gltd_sec * (
                np.longdouble(1.0) - np.exp(-dt_since_glep / gltd_sec)
            )

        phase += np.where(active, glitch_phase, np.longdouble(0.0))
        glitch_idx += 1

    # Add JUMP phase offsets (applied as phase shifts, not delay subtractions)
    if jump_phase is not None:
        phase = phase + np.asarray(jump_phase, dtype=np.longdouble)

    # Subtract TZR phase before wrapping for correct pulse numbering
    if tzr_phase is not None:
        phase = phase - np.longdouble(tzr_phase)

    # Wrap to nearest integer pulse
    frac_phase = phase - np.round(phase)

    # Convert to float64 seconds
    residuals_sec = np.asarray(frac_phase / F0, dtype=np.float64)

    if subtract_mean:
        wm = np.sum(residuals_sec * weights) / np.sum(weights)
        residuals_sec = residuals_sec - wm

    residuals_us = residuals_sec * 1e6
    return residuals_us, residuals_sec


def _load_obs_clock(clock_dir, obs_code, verbose=False):
    """Load observatory clock, combining multi-hop chains if needed.

    When _OBS_CLOCK_FILES maps an observatory to a list of files,
    all corrections are summed on a merged MJD grid that captures
    step functions and fine structure from every file in the chain.
    """
    entry = _OBS_CLOCK_FILES.get(obs_code)
    if entry is None:
        if verbose:
            print(f"   No clock file for '{obs_code}' (assuming zero correction)")
        return {'mjd': np.array([0.0, 100000.0]), 'offset': np.array([0.0, 0.0])}

    files = [entry] if isinstance(entry, str) else list(entry)

    # Load first file
    first_file = files[0]
    if not (clock_dir / first_file).exists():
        if verbose:
            print(f"   Clock file {first_file} not found for '{obs_code}' (assuming zero)")
        return {'mjd': np.array([0.0, 100000.0]), 'offset': np.array([0.0, 0.0])}

    if len(files) == 1:
        clk = parse_clock_file(clock_dir / first_file)
        if verbose:
            print(f"   Loaded clock for {obs_code}: {first_file}")
        return clk

    # Multi-hop: load all files, merge MJD grids, sum corrections
    from jug.io.clock import interpolate_clock_vectorized
    clks = []
    for f in files:
        if (clock_dir / f).exists():
            clks.append(parse_clock_file(clock_dir / f))
        else:
            if verbose:
                print(f"   [!] Clock file {f} not found in chain for '{obs_code}'")
            clks.append({'mjd': np.array([0.0, 100000.0]),
                         'offset': np.array([0.0, 0.0])})

    # Merge all MJD grids (preserving duplicate MJDs for step functions)
    mjd_grid = np.sort(np.unique(np.concatenate([c['mjd'] for c in clks])))
    combined_offset = np.zeros_like(mjd_grid)
    for clk in clks:
        combined_offset += interpolate_clock_vectorized(clk, mjd_grid)

    if verbose:
        print(f"   Loaded clock chain for {obs_code}: {' → '.join(files)}")
    return {'mjd': mjd_grid, 'offset': combined_offset}


def _load_clock_corrections(observatory, all_obs_codes, clock_dir, params,
                            mjd_utc, verbose):
    """Load observatory, GPS, and BIPM clock files; validate coverage.

    Returns
    -------
    dict with keys: obs_clock, obs_clocks, gps_clock, bipm_clock,
        bipm_version, clock_ok, clock_issues.
    """
    clock_dir = Path(clock_dir)

    # Primary observatory clock (may be multi-hop chain)
    obs_clock = _load_obs_clock(clock_dir, observatory.lower(), verbose=verbose)

    # Pre-load clocks for all observatories (multi-obs support)
    obs_clocks = {observatory.lower(): obs_clock}
    if len(all_obs_codes) > 1:
        for obs_code in all_obs_codes:
            if obs_code == observatory.lower():
                continue
            obs_clocks[obs_code] = _load_obs_clock(clock_dir, obs_code, verbose=verbose)

    gps_clock = parse_clock_file(clock_dir / "gps2utc.clk")

    # BIPM version from CLK parameter
    import re
    clk_param = str(params.get('CLK', '')).strip()
    bipm_version = 'bipm2024'
    if clk_param:
        m = re.search(r'BIPM(\d{4})', clk_param, re.IGNORECASE)
        if m:
            bipm_version = f'bipm{m.group(1)}'
    bipm_file = f"tai2tt_{bipm_version}.clk"
    if not (clock_dir / bipm_file).exists():
        if verbose: print(f"   BIPM clock file {bipm_file} not found, falling back to bipm2024")
        bipm_file = "tai2tt_bipm2024.clk"
        bipm_version = 'bipm2024'
    bipm_clock = parse_clock_file(clock_dir / bipm_file)
    if verbose: print(f"   Loaded GPS and {bipm_version.upper()} clock files")

    # Validate coverage
    from jug.io.clock import check_clock_files, check_iers_coverage
    mjd_start = np.min(mjd_utc)
    mjd_end = np.max(mjd_utc)
    if verbose: print(f"\n   Validating clock file coverage (MJD {mjd_start:.1f} - {mjd_end:.1f})...")
    clock_ok, clock_issues = check_clock_files(
        mjd_start, mjd_end, obs_clock, gps_clock, bipm_clock,
        verbose=verbose, clock_dir=str(clock_dir)
    )
    check_iers_coverage(mjd_start, mjd_end, verbose=verbose)

    return {
        'obs_clock': obs_clock, 'obs_clocks': obs_clocks,
        'gps_clock': gps_clock, 'bipm_clock': bipm_clock,
        'bipm_version': bipm_version,
        'clock_ok': clock_ok, 'clock_issues': clock_issues,
    }


def _extract_binary_params(params, verbose):
    """Extract and normalize binary model parameters from the par dict.

    Handles model detection (ELL1/DD/T2/BT/DDK), T2 dispatch,
    ELL1-to-Keplerian conversion, FB mode, and Shapiro parameterization.

    Returns
    -------
    dict with keys: model_id, has_binary, binary_model, plus all scalar
    float values (*_val) and JAX arrays (*_jax) needed by the delay kernel.
    """
    has_binary = 'PB' in params or 'FB0' in params
    binary_model = params.get('BINARY', 'NONE').upper() if has_binary else 'NONE'

    # Map model name to ID:  0=None, 1=ELL1/H, 2=DD/DDH/DDGR, 3=T2, 4=BT*, 5=DDK
    model_id = 0
    if has_binary:
        if binary_model in ('ELL1', 'ELL1H'):
            model_id = 1
        elif binary_model in ('DD', 'DDH', 'DDGR'):
            model_id = 2
        elif binary_model == 'T2':
            has_tasc = 'TASC' in params and float(params.get('TASC', 0.0)) != 0.0
            has_eps = 'EPS1' in params or 'EPS2' in params
            has_kin_kom = 'KIN' in params or 'KOM' in params
            if has_tasc or has_eps:
                model_id = 1  # ELL1
            elif has_kin_kom:
                model_id = 5  # DDK
                if 'KIN' in params:
                    params['KIN'] = 180.0 - float(params['KIN'])
                if 'KOM' in params:
                    params['KOM'] = 90.0 - float(params['KOM'])
            else:
                model_id = 2  # DD
        elif binary_model in ('BT', 'BTX'):
            model_id = 4
        elif binary_model == 'DDK':
            model_id = 5

    if verbose: print(f"\n5. Detecting binary model: {binary_model} (ID: {model_id})")

    # --- Scalar parameter extraction ---
    pb_val = float(params.get('PB', 0.0))
    if pb_val == 0.0 and 'FB0' in params:
        fb0 = float(params['FB0'])
        if fb0 != 0.0:
            pb_val = (1.0 / fb0) / SECS_PER_DAY

    a1_val = float(params.get('A1', 0.0))
    t0_val = float(params.get('T0', 0.0))
    tasc_val = float(params.get('TASC', 0.0))
    ecc_val = float(params.get('ECC', 0.0))
    om_val = float(params.get('OM', 0.0))
    eps1_val = float(params.get('EPS1', 0.0))
    eps2_val = float(params.get('EPS2', 0.0))

    # ELL1-to-Keplerian conversion for DD/T2/DDK models
    if model_id > 1:
        has_ell1_params = 'EPS1' in params or 'EPS2' in params
        has_kepler_params = 'ECC' in params and 'OM' in params
        if has_ell1_params and not has_kepler_params:
            if verbose: print("   Converting ELL1 parameters (EPS1/EPS2) to Keplerian (ECC/OM/T0)")
            ecc_val = np.sqrt(eps1_val**2 + eps2_val**2)
            om_val = np.degrees(np.arctan2(eps1_val, eps2_val)) % 360.0
            t0_val = tasc_val + (om_val / 360.0) * pb_val

    # Normalize T0/TASC fallbacks
    if t0_val == 0.0 and tasc_val != 0.0:
        t0_val = tasc_val
    if tasc_val == 0.0 and t0_val != 0.0:
        tasc_val = t0_val

    gamma_val = float(params.get('GAMMA', 0.0))
    pbdot_val = float(params.get('PBDOT', 0.0))
    xdot_val = float(params.get('XDOT', params.get('A1DOT', 0.0)))
    omdot_val = float(params.get('OMDOT', 0.0))
    edot_val = float(params.get('EDOT', 0.0))
    m2_val = float(params.get('M2', 0.0))

    sini_param = params.get('SINI', 0.0)
    if isinstance(sini_param, str) and sini_param.upper() == 'KIN':
        sini_val = float(jnp.sin(jnp.deg2rad(float(params.get('KIN', 0.0)))))
    else:
        sini_val = float(sini_param)

    kin_val = float(params.get('KIN', 0.0))
    kom_val = float(params.get('KOM', 0.0))
    h3_val = float(params.get('H3', 0.0))
    h4_val = float(params.get('H4', 0.0))
    stig_val = float(params.get('STIG', 0.0))

    # Shapiro parameterization: H3/STIG, H3/H4, H3-only, or M2/SINI
    r_shap_val = 0.0
    s_shap_val = 0.0
    if 'H3' in params and 'STIG' in params and stig_val != 0.0:
        pass  # kernel uses h3/stig directly
    elif 'H3' in params and 'H4' in params and h4_val != 0.0:
        pass  # kernel uses h3/h4 directly
    elif 'H3' in params and h3_val != 0.0 and stig_val == 0.0 and h4_val == 0.0:
        pass  # kernel h3-only branch
    elif 'M2' in params:
        r_shap_val = T_SUN_SEC * m2_val
        s_shap_val = sini_val

    # FB mode
    has_fb0 = 'FB0' in params
    has_higher_fb = any(f'FB{i}' in params for i in range(1, 13))
    use_fb = has_fb0 or (has_higher_fb and 'PB' in params)
    if use_fb:
        if not has_fb0 and 'PB' in params:
            pb_sec = float(params['PB']) * SECS_PER_DAY
            params['FB0'] = 1.0 / pb_sec
        fb_coeffs = []
        fb_idx = 0
        while f'FB{fb_idx}' in params:
            fb_coeffs.append(float(params[f'FB{fb_idx}']))
            fb_idx += 1
        fb_coeffs_jax = jnp.array(fb_coeffs, dtype=jnp.float64)
        fb_factorials_jax = jnp.array([float(math.factorial(i)) for i in range(len(fb_coeffs))], dtype=jnp.float64)
        fb_epoch_jax = jnp.array(float(params.get('TASC', params.get('T0', params['PEPOCH']))))
        use_fb_jax = jnp.array(True)
        if pb_val == 0.0:
            pb_val = 1.0
    else:
        fb_coeffs_jax = jnp.array([0.0], dtype=jnp.float64)
        fb_factorials_jax = jnp.array([1.0], dtype=jnp.float64)
        fb_epoch_jax = jnp.array(0.0)
        use_fb_jax = jnp.array(False)

    # Bundle all JAX arrays
    bp = {
        'model_id': model_id, 'has_binary': has_binary, 'binary_model': binary_model,
        # Scalar values (needed by TZR debug and orbital phase)
        'pb_val': pb_val, 'a1_val': a1_val, 't0_val': t0_val, 'tasc_val': tasc_val,
        'ecc_val': ecc_val, 'om_val': om_val, 'sini_val': sini_val,
        # JAX scalars
        'has_binary_jax': jnp.array(has_binary),
        'binary_model_id_jax': jnp.array(model_id, dtype=jnp.int32),
        'pb_jax': jnp.array(pb_val), 'a1_jax': jnp.array(a1_val),
        'tasc_jax': jnp.array(tasc_val), 't0_jax': jnp.array(t0_val),
        'ecc_jax': jnp.array(ecc_val), 'om_jax': jnp.array(om_val),
        'eps1_jax': jnp.array(eps1_val), 'eps2_jax': jnp.array(eps2_val),
        'gamma_jax': jnp.array(gamma_val), 'pbdot_jax': jnp.array(pbdot_val),
        'xdot_jax': jnp.array(xdot_val), 'omdot_jax': jnp.array(omdot_val),
        'edot_jax': jnp.array(edot_val),
        'm2_jax': jnp.array(m2_val), 'sini_jax': jnp.array(sini_val),
        'kin_jax': jnp.array(kin_val), 'kom_jax': jnp.array(kom_val),
        'h3_jax': jnp.array(h3_val), 'h4_jax': jnp.array(h4_val),
        'stig_jax': jnp.array(stig_val),
        'r_shap_jax': jnp.array(r_shap_val), 's_shap_jax': jnp.array(s_shap_val),
        # FB arrays
        'fb_coeffs_jax': fb_coeffs_jax, 'fb_factorials_jax': fb_factorials_jax,
        'fb_epoch_jax': fb_epoch_jax, 'use_fb_jax': use_fb_jax,
    }
    return bp


def _prepare_ddk_kopeikin(params, model_id, is_ecliptic, ssb_obs_pos_km,
                          ra_rad, dec_rad, parallax_mas, verbose):
    """Prepare DDK Kopeikin correction parameters as JAX arrays.

    Returns
    -------
    dict with keys: obs_pos_ls_jax, px_jax, sin_ra_jax, cos_ra_jax,
        sin_dec_jax, cos_dec_jax, k96_jax, pmra_rad_per_sec_jax,
        pmdec_rad_per_sec_jax, ecl_obl_cos, ecl_obl_sin.
    """
    SPEED_OF_LIGHT_KM_S = C_KM_S
    ecl_obl_cos = 1.0
    ecl_obl_sin = 0.0

    if is_ecliptic and model_id == 5:
        from jug.io.par_reader import OBLIQUITY_ARCSEC
        ecl_frame = str(params.get('_ecliptic_frame', 'IERS2010')).upper()
        _obl_rad = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010']) * np.pi / (180.0 * 3600.0)
        ecl_obl_cos = np.cos(_obl_rad)
        ecl_obl_sin = np.sin(_obl_rad)
        _x = ssb_obs_pos_km[:, 0]
        _y = ssb_obs_pos_km[:, 1] * ecl_obl_cos + ssb_obs_pos_km[:, 2] * ecl_obl_sin
        _z = -ssb_obs_pos_km[:, 1] * ecl_obl_sin + ssb_obs_pos_km[:, 2] * ecl_obl_cos
        obs_pos_for_ddk = np.column_stack([_x, _y, _z])
    else:
        obs_pos_for_ddk = ssb_obs_pos_km

    obs_pos_ls_jax = jnp.array(obs_pos_for_ddk / SPEED_OF_LIGHT_KM_S, dtype=jnp.float64)
    px_jax = jnp.array(parallax_mas)

    # Pulsar coordinates (ecliptic lon/lat for ecliptic pulsars, RA/DEC otherwise)
    if is_ecliptic and model_id == 5:
        _ecl_lon_rad = np.radians(params['_ecliptic_lon_deg'])
        _ecl_lat_rad = np.radians(params['_ecliptic_lat_deg'])
        sin_ra_jax = jnp.array(np.sin(_ecl_lon_rad))
        cos_ra_jax = jnp.array(np.cos(_ecl_lon_rad))
        sin_dec_jax = jnp.array(np.sin(_ecl_lat_rad))
        cos_dec_jax = jnp.array(np.cos(_ecl_lat_rad))
    else:
        sin_ra_jax = jnp.array(np.sin(ra_rad))
        cos_ra_jax = jnp.array(np.cos(ra_rad))
        sin_dec_jax = jnp.array(np.sin(dec_rad))
        cos_dec_jax = jnp.array(np.cos(dec_rad))

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
    k96_jax = jnp.array(k96_flag)

    # Proper motion in radians/second
    MAS_PER_YR_TO_RAD_PER_SEC = 1.0 / (MAS_PER_RAD * SECS_PER_YEAR)
    if is_ecliptic and model_id == 5:
        pmra_mas_yr = float(params.get('_ecliptic_pm_lon', 0.0))
        pmdec_mas_yr = float(params.get('_ecliptic_pm_lat', 0.0))
    else:
        pmra_mas_yr = float(params.get('PMRA', 0.0))
        pmdec_mas_yr = float(params.get('PMDEC', 0.0))

    pmra_rad_per_sec_jax = jnp.array(pmra_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC)
    pmdec_rad_per_sec_jax = jnp.array(pmdec_mas_yr * MAS_PER_YR_TO_RAD_PER_SEC)

    if verbose and model_id == 5:
        _pm_label = "PMELONG/PMELAT" if is_ecliptic else "PMRA/PMDEC"
        print(f"   DDK model with Kopeikin corrections (frame: {'ecliptic' if is_ecliptic else 'equatorial'}):")
        print(f"     KIN={float(params.get('KIN',0)):.3f}deg, KOM={float(params.get('KOM',0)):.3f}deg, PX={parallax_mas:.3f} mas")
        print(f"     K96={k96_flag}, {_pm_label}=({pmra_mas_yr:.3f}, {pmdec_mas_yr:.3f}) mas/yr")

    return {
        'obs_pos_ls_jax': obs_pos_ls_jax, 'px_jax': px_jax,
        'sin_ra_jax': sin_ra_jax, 'cos_ra_jax': cos_ra_jax,
        'sin_dec_jax': sin_dec_jax, 'cos_dec_jax': cos_dec_jax,
        'k96_jax': k96_jax,
        'pmra_rad_per_sec_jax': pmra_rad_per_sec_jax,
        'pmdec_rad_per_sec_jax': pmdec_rad_per_sec_jax,
        'ecl_obl_cos': ecl_obl_cos, 'ecl_obl_sin': ecl_obl_sin,
    }


def _call_delay_kernel(tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
                       dm_jax, bp, ddk, roemer_shapiro_jax,
                       tropo_jax, dmx_jax):
    """Call the JAX combined delay kernel with all parameters.

    Parameters
    ----------
    dm_jax : dict  with keys dm_coeffs_jax, dm_factorials_jax, dm_epoch_jax,
        ne_sw_jax, fd_coeffs_jax, has_fd_jax.
    bp : dict from _extract_binary_params.
    ddk : dict from _prepare_ddk_kopeikin.
    """
    return compute_total_delay_jax(
        tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
        dm_jax['dm_coeffs_jax'], dm_jax['dm_factorials_jax'], dm_jax['dm_epoch_jax'],
        dm_jax['ne_sw_jax'], dm_jax['fd_coeffs_jax'], dm_jax['has_fd_jax'],
        roemer_shapiro_jax, bp['has_binary_jax'], bp['binary_model_id_jax'],
        bp['pb_jax'], bp['a1_jax'], bp['tasc_jax'], bp['eps1_jax'], bp['eps2_jax'],
        bp['pbdot_jax'], bp['xdot_jax'], bp['gamma_jax'], bp['r_shap_jax'], bp['s_shap_jax'],
        bp['ecc_jax'], bp['om_jax'], bp['t0_jax'], bp['omdot_jax'], bp['edot_jax'],
        bp['m2_jax'], bp['sini_jax'], bp['kin_jax'], bp['kom_jax'],
        bp['h3_jax'], bp['h4_jax'], bp['stig_jax'],
        bp['fb_coeffs_jax'], bp['fb_factorials_jax'], bp['fb_epoch_jax'], bp['use_fb_jax'],
        ddk['obs_pos_ls_jax'], ddk['px_jax'],
        ddk['sin_ra_jax'], ddk['cos_ra_jax'], ddk['sin_dec_jax'], ddk['cos_dec_jax'],
        ddk['k96_jax'], ddk['pmra_rad_per_sec_jax'], ddk['pmdec_rad_per_sec_jax'],
        tropo_jax, dmx_jax,
    ).block_until_ready()


def _compute_tzr_phase(params, bp, dm_jax, ddk,
                       obs_clock, gps_clock, bipm_clock,
                       observatory, location, obs_itrf_km,
                       ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day,
                       posepoch, parallax_mas, ephem,
                       F0, F1_half, F2_sixth, PEPOCH_sec,
                       is_ecliptic, ssb_obs_pos_km, fd_coeffs,
                       planet_shapiro_enabled,
                       tzrmjd_scale, verbose):
    """Compute phase at the TZR reference point.

    Returns
    -------
    tzr_phase : np.longdouble
        Phase at TZRMJD, to be subtracted from TOA phases.
    tzrmjd_scale_resolved : str
        Timescale used for TZRMJD ('UTC' or 'TDB').
    """
    TZRMJD_raw = get_longdouble(params, 'TZRMJD')
    SPEED_OF_LIGHT_KM_S = C_KM_S

    # Resolve TZRSITE
    tzr_site_raw = params.get('TZRSITE', observatory)
    tzr_site = str(tzr_site_raw).lower() if tzr_site_raw is not None else observatory.lower()
    tzr_is_ssb = tzr_site in ('ssb', '@', 'coe')
    tzr_obs_itrf_km = OBSERVATORIES.get(tzr_site)

    if tzr_is_ssb:
        tzr_obs_itrf_km = np.array([0.0, 0.0, 0.0])
        tzr_location = EarthLocation.from_geocentric(0 * u.km, 0 * u.km, 0 * u.km)
        if verbose: print(f"   TZRSITE=ssb: barycentric reference TOA (no clock/Roemer correction)")
    elif tzr_obs_itrf_km is None:
        if verbose: print(f"   [!]  Unknown TZRSITE '{tzr_site}', falling back to '{observatory}'")
        tzr_obs_itrf_km = obs_itrf_km
        tzr_location = location
    else:
        tzr_location = EarthLocation.from_geocentric(
            tzr_obs_itrf_km[0] * u.km, tzr_obs_itrf_km[1] * u.km, tzr_obs_itrf_km[2] * u.km
        )

    # Convert TZRMJD to TDB
    tzrmjd_scale_resolved = "UTC"
    tzrmjd_scale_upper = tzrmjd_scale.upper()

    if tzr_is_ssb:
        TZRMJD_TDB = TZRMJD_raw
        delta_tzr_sec = 0.0
        if verbose: print(f"   TZRMJD treated as TDB (TZRSITE=ssb, barycentric)")
    elif tzrmjd_scale_upper in ("AUTO", "UTC"):
        tzrmjd_scale_resolved = "UTC"
        if verbose:
            label = "AUTO -> UTC" if tzrmjd_scale_upper == "AUTO" else "UTC (explicit override)"
            print(f"   TZRMJD scale: {label} (site arrival time, converting to TDB)")
        TZRMJD_TDB_ld = compute_tdb_standalone_vectorized(
            [int(TZRMJD_raw)], [float(TZRMJD_raw - int(TZRMJD_raw))],
            obs_clock, gps_clock, bipm_clock, tzr_location
        )[0]
        TZRMJD_TDB = np.longdouble(TZRMJD_TDB_ld)
        delta_tzr_sec = float(TZRMJD_TDB - TZRMJD_raw) * SECS_PER_DAY
        if verbose: print(f"   TZRMJD converted from UTC to TDB (delta = {delta_tzr_sec:.3f} s)")
    elif tzrmjd_scale_upper == "TDB":
        TZRMJD_TDB = TZRMJD_raw
        delta_tzr_sec = 0.0
        tzrmjd_scale_resolved = "TDB"
        if verbose: print(f"   TZRMJD treated as TDB (no conversion)")
    else:
        raise ValueError(f"Invalid tzrmjd_scale '{tzrmjd_scale}'. Must be 'AUTO', 'TDB', or 'UTC'.")

    # Astrometry at TZR
    tzr_tdb_arr = np.array([float(TZRMJD_TDB)])
    if tzr_is_ssb:
        tzr_ssb_obs_pos = np.zeros((1, 3))
        tzr_ssb_obs_vel = np.zeros((1, 3))
    else:
        tzr_ssb_obs_pos, tzr_ssb_obs_vel = compute_ssb_obs_pos_vel(tzr_tdb_arr, tzr_obs_itrf_km, ephemeris=ephem)

    tzr_L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tzr_tdb_arr)
    tzr_roemer = compute_roemer_delay(tzr_ssb_obs_pos, tzr_L_hat, parallax_mas)[0]

    # Sun Shapiro at TZR
    tzr_times = Time(tzr_tdb_arr, format='mjd', scale='tdb')
    with solar_system_ephemeris.set(ephem):
        tzr_sun_pos = get_body_barycentric_posvel('sun', tzr_times)[0].xyz.to(u.km).value.T
    tzr_obs_sun = tzr_sun_pos - tzr_ssb_obs_pos
    tzr_sun_shapiro = compute_shapiro_delay(tzr_obs_sun, tzr_L_hat, T_SUN_SEC)[0]

    # Planet Shapiro at TZR
    tzr_planet_shapiro = 0.0
    if planet_shapiro_enabled:
        with solar_system_ephemeris.set(ephem):
            for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
                tzr_planet_pos = get_body_barycentric_posvel(planet, tzr_times)[0].xyz.to(u.km).value.T
                tzr_obs_planet = tzr_planet_pos - tzr_ssb_obs_pos
                tzr_planet_shapiro += compute_shapiro_delay(tzr_obs_planet, tzr_L_hat, T_PLANET[planet])[0]

    tzr_roemer_shapiro = tzr_roemer + tzr_sun_shapiro + tzr_planet_shapiro

    # Barycentric frequency at TZR
    if 'TZRFRQ' in params:
        tzr_freq_raw = float(params['TZRFRQ'])
        tzr_freq = tzr_freq_raw if np.isfinite(tzr_freq_raw) else 1e12
    else:
        tzr_freq = 1400.0
    tzr_freq_bary = compute_barycentric_freq(np.array([tzr_freq]), tzr_ssb_obs_vel, tzr_L_hat)[0]

    # TZR DDK observer position
    model_id = bp['model_id']
    if is_ecliptic and model_id == 5:
        _x = tzr_ssb_obs_pos[:, 0]
        _y = tzr_ssb_obs_pos[:, 1] * ddk['ecl_obl_cos'] + tzr_ssb_obs_pos[:, 2] * ddk['ecl_obl_sin']
        _z = -tzr_ssb_obs_pos[:, 1] * ddk['ecl_obl_sin'] + tzr_ssb_obs_pos[:, 2] * ddk['ecl_obl_cos']
        tzr_obs_pos_for_ddk = np.column_stack([_x, _y, _z])
    else:
        tzr_obs_pos_for_ddk = tzr_ssb_obs_pos

    # TZR DMX
    from jug.model.dmx import parse_dmx_ranges, build_dmx_design_matrix
    tzr_dmx_ranges = parse_dmx_ranges(params)
    tzr_dmx_delay = 0.0
    if tzr_dmx_ranges:
        tzr_dmx_matrix, _ = build_dmx_design_matrix(
            np.array([float(TZRMJD_raw)], dtype=np.float64),
            np.array([tzr_freq_bary]), tzr_dmx_ranges
        )
        tzr_dmx_values = np.array([r.value for r in tzr_dmx_ranges])
        tzr_dmx_delay = float((tzr_dmx_matrix @ tzr_dmx_values)[0])

    # Build TZR-specific DDK dict with TZR observer position
    tzr_ddk = dict(ddk)
    tzr_ddk['obs_pos_ls_jax'] = jnp.array(tzr_obs_pos_for_ddk / SPEED_OF_LIGHT_KM_S, dtype=jnp.float64)

    # Call delay kernel at TZR
    tzr_total_delay_jax = _call_delay_kernel(
        jnp.array([float(TZRMJD_TDB)]), jnp.array([tzr_freq_bary]),
        jnp.array(tzr_obs_sun), jnp.array(tzr_L_hat),
        dm_jax, bp, tzr_ddk, jnp.array([tzr_roemer_shapiro]),
        None, jnp.array([tzr_dmx_delay], dtype=jnp.float64),
    )

    tzr_delay = np.longdouble(float(tzr_total_delay_jax[0]))
    if tzr_dmx_delay != 0.0:
        tzr_delay += np.longdouble(tzr_dmx_delay)
        if verbose: print(f"   TZR DMX delay: {tzr_dmx_delay:.9f} s")

    # Verbose: TZR delay breakdown
    if verbose:
        dm_coeffs = []
        k = 0
        while True:
            key = 'DM' if k == 0 else f'DM{k}'
            if key in params:
                dm_coeffs.append(float(params[key]))
                k += 1
            else:
                break
        dm_coeffs = dm_coeffs or [0.0]
        dm_epoch = float(params.get('DMEPOCH', params['PEPOCH']))
        dt_years = (float(TZRMJD_TDB) - dm_epoch) / 365.25
        dm_eff = sum(dm_coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(dm_coeffs)))
        tzr_dm_delay = K_DM_SEC * dm_eff / (tzr_freq_bary ** 2)

        ne_sw = float(params.get('NE_SW', 0.0))
        if ne_sw != 0:
            r_km = np.sqrt(np.sum(tzr_obs_sun**2))
            r_au = r_km / AU_KM
            sun_dir = tzr_obs_sun / r_km
            cos_elong = np.sum(sun_dir * tzr_L_hat[0])
            elong = np.arccos(np.clip(cos_elong, -1.0, 1.0))
            rho = np.pi - elong
            sin_rho = max(np.sin(rho), 1e-10)
            geometry_pc = AU_PC * rho / (r_au * sin_rho)
            dm_sw = ne_sw * geometry_pc
            tzr_sw_delay = K_DM_SEC * dm_sw / (tzr_freq_bary ** 2)
        else:
            tzr_sw_delay = 0.0

        if fd_coeffs and len(fd_coeffs) > 0:
            log_freq = np.log(tzr_freq_bary / 1000.0)
            tzr_fd_delay = np.polyval(list(fd_coeffs)[::-1] + [0], log_freq)
        else:
            tzr_fd_delay = 0.0

        has_binary = bp['has_binary']
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

    if verbose:
        print(f"   TZRMJD (raw):  {float(TZRMJD_raw):.15f}")
        print(f"   TZRMJD (used): {float(TZRMJD_TDB):.15f} (scale={tzrmjd_scale_resolved})")
        print(f"   delta_tzr:     {delta_tzr_sec:.6f} s")
        print(f"   TZR delay: {float(tzr_delay):.9f} s")
        print(f"   TZR phase: {float(tzr_phase):.6f} cycles")

    return tzr_phase


def compute_residuals_simple(
    par_file: Path | str,
    tim_file: Path | str,
    clock_dir: Path | str | None = None,
    observatory: str = "auto",
    subtract_tzr: bool = True,
    verbose: bool = True,
    tzrmjd_scale: str = "AUTO",
) -> dict:
    """Compute pulsar timing residuals from .par and .tim files.

    Orchestrates the full residual computation pipeline:
    1. Parse par/tim files, detect observatories
    2. Load and validate clock corrections
    3. Compute TDB via clock chain (UTC -> GPS -> TT -> TDB)
    4. Compute astrometric delays (Roemer, Shapiro, parallax)
    5. Run JAX delay kernel (DM, solar wind, FD, binary)
    6. Apply JUMPs, DMX, exponential dips, troposphere
    7. Compute phase residuals relative to TZR reference

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
        Observatory name (default: "auto" -- auto-detect from .tim file)
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
    >>> print(f"RMS: {result['rms_us']:.3f} mus")
    >>> print(f"N_TOAs: {result['n_toas']}")
    """
    if verbose:
        print("=" * 60)
        print("JUG Simple Residual Calculator")
        print("=" * 60)

    # Set default clock directory relative to package installation
    if clock_dir is None:
        # Get the directory where this module is located
        module_dir = Path(__file__).parent
        # Navigate to the JUG root directory and then to data/clock
        clock_dir = module_dir.parent.parent / "data" / "clock"

    # Parse files
    if verbose: print(f"\n1. Loading files...")
    params = parse_par_file(par_file)
    
    # Validate par file timescale and convert TCB to TDB if needed
    par_timescale = validate_par_timescale(params, context="compute_residuals_simple", verbose=verbose)
    if verbose: print(f"   Par file timescale: {par_timescale}")
    
    toas = parse_tim_file_mjds(tim_file)
    if verbose: print(f"   Loaded {len(toas)} TOAs from {Path(tim_file).name}")
    if verbose: print(f"   Loaded timing model from {Path(par_file).name}")

    # Warn about unrecognized parameters
    _warn_unrecognized_params(params, verbose=verbose)

    # Observatory location - auto-detect from TOAs if not explicitly set.
    # Collect all unique observatories in the dataset for multi-obs support.
    if observatory == "auto" and toas:
        observatory = toas[0].observatory
        if verbose: print(f"   Auto-detected primary observatory: {observatory}")
    obs_itrf_km = OBSERVATORIES.get(observatory.lower())
    if obs_itrf_km is None:
        raise ValueError(f"Unknown observatory: {observatory}. "
                         f"Known: {', '.join(sorted(set(OBSERVATORIES.keys())))}")

    # Detect all unique observatories in the TOA list
    all_obs_codes = sorted(set(toa.observatory.lower() for toa in toas))
    is_multi_obs = len(all_obs_codes) > 1
    if is_multi_obs and verbose:
        print(f"   Multi-observatory dataset: {all_obs_codes}")

    # Load clock files
    if verbose: print(f"\n2. Loading clock corrections...")
    mjd_utc = np.array([toa.mjd_int + toa.mjd_frac for toa in toas])
    clk = _load_clock_corrections(observatory, all_obs_codes, clock_dir, params, mjd_utc, verbose)
    obs_clock = clk['obs_clock']
    obs_clocks = clk['obs_clocks']
    gps_clock = clk['gps_clock']
    bipm_clock = clk['bipm_clock']
    clock_issues = clk['clock_issues']

    location = EarthLocation.from_geocentric(
        obs_itrf_km[0] * u.km,
        obs_itrf_km[1] * u.km,
        obs_itrf_km[2] * u.km
    )

    # Compute TDB
    if verbose: print(f"\n3. Computing TDB (standalone, no PINT)...")
    mjd_ints = [toa.mjd_int for toa in toas]
    mjd_fracs = [toa.mjd_frac for toa in toas]

    # Extract -to flags (TIME statement offsets, in seconds)
    time_offsets = np.array([float(toa.flags.get('to', 0.0)) for toa in toas])
    n_to = np.sum(time_offsets != 0.0)
    if n_to > 0 and verbose:
        print(f"   Applying -to time offsets to {n_to} TOAs")

    if not is_multi_obs:
        tdb_mjd = compute_tdb_standalone_vectorized(
            mjd_ints, mjd_fracs,
            obs_clock, gps_clock, bipm_clock,
            location, time_offsets=time_offsets
        )
    else:
        # Multi-observatory: compute TDB per observatory, then reassemble
        tdb_mjd = np.zeros(len(toas), dtype=np.longdouble)
        for obs_code in all_obs_codes:
            idxs = [i for i, toa in enumerate(toas) if toa.observatory.lower() == obs_code]
            obs_loc_km = OBSERVATORIES.get(obs_code)
            if obs_loc_km is None:
                if verbose: print(f"   [!]  Unknown observatory '{obs_code}', using primary")
                obs_loc_km = obs_itrf_km
            obs_loc = EarthLocation.from_geocentric(
                obs_loc_km[0] * u.km, obs_loc_km[1] * u.km, obs_loc_km[2] * u.km
            )
            clk = obs_clocks.get(obs_code, obs_clock)
            tdb_mjd[idxs] = compute_tdb_standalone_vectorized(
                [mjd_ints[i] for i in idxs], [mjd_fracs[i] for i in idxs],
                clk, gps_clock, bipm_clock, obs_loc,
                time_offsets=time_offsets[idxs]
            )
        if verbose: print(f"   Computed TDB per observatory: {all_obs_codes}")
    if verbose: print(f"   Computed TDB for {len(tdb_mjd)} TOAs")

    # Astrometry
    if verbose: print(f"\n4. Computing astrometric delays...")
    ephem = _resolve_ephemeris(str(params.get('EPHEM', 'de440')).lower())
    ra_rad = parse_ra(params['RAJ'])
    dec_rad = parse_dec(params['DECJ'])
    pmra_rad_day = params.get('PMRA', 0.0) * (np.pi / 180 / 3600000) / 365.25
    pmdec_rad_day = params.get('PMDEC', 0.0) * (np.pi / 180 / 3600000) / 365.25
    posepoch = params.get('POSEPOCH', params['PEPOCH'])
    parallax_mas = params.get('PX', 0.0)

    if not is_multi_obs:
        ssb_obs_pos_km, ssb_obs_vel_km_s = compute_ssb_obs_pos_vel(tdb_mjd, obs_itrf_km, ephemeris=ephem)
    else:
        # Multi-observatory: compute SSB-obs position per observatory
        ssb_obs_pos_km = np.zeros((len(toas), 3))
        ssb_obs_vel_km_s = np.zeros((len(toas), 3))
        for obs_code in all_obs_codes:
            idxs = [i for i, toa in enumerate(toas) if toa.observatory.lower() == obs_code]
            obs_loc_km = OBSERVATORIES.get(obs_code, obs_itrf_km)
            pos, vel = compute_ssb_obs_pos_vel(tdb_mjd[idxs], obs_loc_km, ephemeris=ephem)
            ssb_obs_pos_km[idxs] = pos
            ssb_obs_vel_km_s[idxs] = vel
    L_hat = compute_pulsar_direction(ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day, posepoch, tdb_mjd)

    # Roemer and Shapiro delays
    roemer_sec = compute_roemer_delay(ssb_obs_pos_km, L_hat, parallax_mas)

    times = Time(tdb_mjd, format='mjd', scale='tdb')
    with solar_system_ephemeris.set(ephem):
        sun_pos = get_body_barycentric_posvel('sun', times)[0].xyz.to(u.km).value.T
    obs_sun_pos_km = sun_pos - ssb_obs_pos_km
    sun_shapiro_sec = compute_shapiro_delay(obs_sun_pos_km, L_hat, T_SUN_SEC)

    # Planet Shapiro (if enabled)
    planet_shapiro_enabled = str(params.get('PLANET_SHAPIRO', 'N')).upper() in ('Y', 'YES', 'TRUE', '1')
    planet_shapiro_sec = np.zeros(len(tdb_mjd))
    if planet_shapiro_enabled:
        if verbose: print(f"   Computing planetary Shapiro delays...")
        with solar_system_ephemeris.set(ephem):
            for planet in ['jupiter', 'saturn', 'uranus', 'neptune', 'venus']:
                planet_pos = get_body_barycentric_posvel(planet, times)[0].xyz.to(u.km).value.T
                obs_planet_km = planet_pos - ssb_obs_pos_km
                planet_shapiro_sec += compute_shapiro_delay(obs_planet_km, L_hat, T_PLANET[planet])

    roemer_shapiro = roemer_sec + sun_shapiro_sec + planet_shapiro_sec

    # Barycentric frequency (used for DM/SW/FD delays by both Tempo2 and PINT)
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

    # Bundle DM/FD JAX arrays for kernel calls
    dm_jax = {
        'dm_coeffs_jax': dm_coeffs_jax, 'dm_factorials_jax': dm_factorials_jax,
        'dm_epoch_jax': dm_epoch_jax, 'ne_sw_jax': ne_sw_jax,
        'fd_coeffs_jax': fd_coeffs_jax, 'has_fd_jax': has_fd_jax,
    }

    # Binary parameters - detect model and extract all values
    bp = _extract_binary_params(params, verbose)
    model_id = bp['model_id']
    has_binary = bp['has_binary']

    # DDK Kopeikin parameters
    is_ecliptic = bool(params.get('_ecliptic_coords', False))
    ddk = _prepare_ddk_kopeikin(params, model_id, is_ecliptic, ssb_obs_pos_km,
                                ra_rad, dec_rad, parallax_mas, verbose)

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
        
        # Source coordinate (J2000)
        source_coord = SkyCoord(ra=ra_rad*u.rad, dec=dec_rad*u.rad, frame='icrs')
        
        mjd_utc_arr = np.array([t.mjd_int + t.mjd_frac for t in toas])
        
        # Multi-observatory: compute tropospheric delay per observatory using correct location
        for obs_code in all_obs_codes:
            idxs = [i for i, toa in enumerate(toas) if toa.observatory.lower() == obs_code]
            obs_loc_km = OBSERVATORIES.get(obs_code, obs_itrf_km)
            obs_loc = EarthLocation.from_geocentric(
                obs_loc_km[0] * u.km, obs_loc_km[1] * u.km, obs_loc_km[2] * u.km
            )
            loc_height_m = obs_loc.height.to(u.m).value
            loc_lat_deg = obs_loc.lat.deg
            
            mjd_obs = mjd_utc_arr[idxs]
            obs_times = Time(mjd_obs, format='mjd', scale='utc')
            topocentric_frame = AltAz(obstime=obs_times, location=obs_loc)
            source_altaz = source_coord.transform_to(topocentric_frame)
            elevation_deg = source_altaz.alt.deg
            
            tropo_obs = np.asarray(compute_tropospheric_delay(
                elevation_deg=elevation_deg,
                height_m=loc_height_m,
                lat_deg=loc_lat_deg,
                mjd=mjd_obs
            ), dtype=np.float64)
            tropo_delay_sec[idxs] = tropo_obs
        
        if verbose: print(f"   Tropospheric delay: mean={np.mean(tropo_delay_sec)*1e6:.3f} mus, range=[{np.min(tropo_delay_sec)*1e6:.3f}, {np.max(tropo_delay_sec)*1e6:.3f}] mus")
    
    tropo_jax = jnp.array(tropo_delay_sec, dtype=jnp.float64)

    # Compute DMX delay BEFORE the kernel call so it can be included in
    # the pre-binary time (PINT evaluates DMX before the binary model)
    from jug.model.dmx import parse_dmx_ranges, build_dmx_design_matrix
    dmx_ranges = parse_dmx_ranges(params)
    dmx_delay_sec = np.zeros(len(tdb_mjd), dtype=np.float64)
    if dmx_ranges:
        # Use site arrival MJDs for DMX range matching (consistent with PINT/Tempo2)
        dmx_matrix, dmx_labels = build_dmx_design_matrix(np.array(mjd_utc, dtype=np.float64), freq_bary_mhz, dmx_ranges)
        dmx_values = np.array([r.value for r in dmx_ranges])
        dmx_delay_sec = np.asarray(dmx_matrix @ dmx_values, dtype=np.float64)
        if verbose: print(f"   Computed {len(dmx_ranges)} DMX ranges for pre-binary time")
    dmx_jax = jnp.array(dmx_delay_sec, dtype=jnp.float64)

    # Compute total delay (DM + SW + FD + binary)
    if verbose: print(f"\n6. Running JAX delay kernel...")
    total_delay_jax = _call_delay_kernel(
        tdb_jax, freq_bary_jax, obs_sun_jax, L_hat_jax,
        dm_jax, bp, ddk, roemer_shapiro_jax, tropo_jax, dmx_jax,
    )
    total_delay_sec = np.asarray(total_delay_jax, dtype=np.longdouble)

    # Add tropospheric delay to total (kernel uses it only for binary time, not in sum)
    if correct_troposphere:
        total_delay_sec += np.asarray(tropo_delay_sec, dtype=np.float64)

    # Compute DM and SW delays separately for pre-binary time (needed by fitter)
    # These replicate the kernel formulas in NumPy for use outside the kernel
    dm_epoch = float(params.get('DMEPOCH', params['PEPOCH']))
    dt_years = (np.array(tdb_mjd) - dm_epoch) / 365.25
    dm_eff = sum(dm_coeffs[i] * (dt_years ** i) / math.factorial(i) for i in range(len(dm_coeffs)))
    dm_delay_sec = K_DM_SEC * dm_eff / (freq_bary_mhz ** 2)

    # Add DMX contribution to total delay and DM delay
    # DMX was already computed before the kernel call (for pre-binary time);
    # now add it to total_delay_sec (kernel doesn't include it in its sum)
    if dmx_ranges:
        total_delay_sec += np.asarray(dmx_delay_sec, dtype=np.float64)
        dm_delay_sec = dm_delay_sec + dmx_delay_sec
        if verbose: print(f"   Applied {len(dmx_ranges)} DMX ranges to DM delay")

    # Exponential dip model (Tempo2 EXPEP/EXPPH/EXPTAU/EXPINDEX)
    # Adds frequency-dependent exponential decay delays for DM events.
    # Formula: delay += EXPPH * (freq_SSB/1.4GHz)^EXPINDEX * exp(-(t-EXPEP)/EXPTAU)
    # Only applied for t > EXPEP. EXPINDEX defaults to -2 if not set.
    exp_idx = 1
    while f'EXPEP_{exp_idx}' in params:
        expep = float(params[f'EXPEP_{exp_idx}'])
        expph = float(params.get(f'EXPPH_{exp_idx}', 0.0))
        exptau = float(params.get(f'EXPTAU_{exp_idx}', 1.0))
        expindex = float(params.get(f'EXPINDEX_{exp_idx}', -2.0))
        
        dt_exp = np.array(tdb_mjd, dtype=np.float64) - expep
        active = dt_exp > 0
        if np.any(active):
            freq_norm = np.array(freq_bary_mhz, dtype=np.float64) / 1400.0
            exp_delay = np.zeros(len(tdb_mjd), dtype=np.float64)
            exp_delay[active] = expph * (freq_norm[active] ** expindex) * np.exp(-dt_exp[active] / exptau)
            total_delay_sec -= exp_delay
            if verbose:
                print(f"   Applied EXP dip {exp_idx}: epoch={expep:.1f}, amp={expph:.3e} s, tau={exptau:.1f} d")
        exp_idx += 1

    # Apply JUMPs as phase offsets (not delay subtractions).
    # Tempo2 treats JUMPs as phase shifts: delta_phase = F0 * JUMP_value.
    # Subtracting JUMP from total_delay creates an F1*dt*JUMP cross-term in the
    # spindown polynomial that corrupts residuals for large JUMPs (>1 s).
    # Phase offsets avoid this entirely and match Tempo2 to <0.1 mus.
    jump_phase = np.zeros(len(toas), dtype=np.longdouble)
    jump_lines = params.get('_jump_lines', [])
    if jump_lines:
        F0_jump = get_longdouble(params, 'F0')
        from jug.fitting.derivatives_jump import parse_jump_from_par_line, create_jump_mask_from_flags
        n_jumps_applied = 0
        for jline in jump_lines:
            jinfo = parse_jump_from_par_line(jline)
            if jinfo['type'] == 'flag':
                mask = create_jump_mask_from_flags(
                    [t.flags for t in toas],
                    jinfo['flag_name'], jinfo['flag_value']
                )
                if np.any(mask):
                    jump_phase[mask] += F0_jump * np.longdouble(jinfo['value'])
                    n_jumps_applied += 1
                    if verbose:
                        print(f"   JUMP {jinfo['flag_name']}={jinfo['flag_value']}: "
                              f"{jinfo['value']*1e6:.3f} mus applied to {np.sum(mask)} TOAs")
            elif jinfo['type'] == 'mjd':
                toas_mjd_arr = np.array([t.mjd_int + t.mjd_frac for t in toas])
                mask = (toas_mjd_arr >= jinfo['mjd_start']) & (toas_mjd_arr <= jinfo['mjd_end'])
                if np.any(mask):
                    jump_phase[mask] += F0_jump * np.longdouble(jinfo['value'])
                    n_jumps_applied += 1
                    if verbose:
                        print(f"   JUMP MJD {jinfo['mjd_start']}-{jinfo['mjd_end']}: "
                              f"{jinfo['value']*1e6:.3f} mus applied to {np.sum(mask)} TOAs")
        if verbose and n_jumps_applied:
            print(f"   Applied {n_jumps_applied} JUMPs as phase offsets")

    # Apply -padd (phase offset, cycles) and -radd (time offset, seconds → cycles)
    # flags from TIM files, matching TEMPO2's formResiduals.C behaviour.
    n_padd = 0
    for i, toa in enumerate(toas):
        if 'padd' in toa.flags:
            padd_val = np.longdouble(toa.flags['padd'])
            jump_phase[i] += padd_val
            n_padd += 1
        if 'radd' in toa.flags:
            radd_val = np.longdouble(toa.flags['radd'])
            jump_phase[i] += radd_val * F0_jump
            n_padd += 1
    if verbose and n_padd:
        print(f"   Applied -padd/-radd phase offsets to {n_padd} TOAs")

    # NOTE: DMJUMP is NOT applied in prefit residuals.
    # Neither Tempo2 nor PINT apply DMJUMP as a delay in prefit;
    # it is only used as design-matrix columns in the fitter.

    # Solar wind geometry (always computed for caching; cost is negligible)
    ne_sw = float(params.get('NE_SW', 0.0))
    r_km = np.sqrt(np.sum(obs_sun_pos_km**2, axis=1))
    r_au = r_km / AU_KM
    sun_dir = obs_sun_pos_km / r_km[:, np.newaxis]
    cos_elong = np.sum(sun_dir * L_hat, axis=1)
    elong = np.arccos(np.clip(cos_elong, -1.0, 1.0))
    rho = np.pi - elong
    sin_rho = np.maximum(np.sin(rho), 1e-10)
    sw_geometry_pc = AU_PC * rho / (r_au * sin_rho)
    if ne_sw != 0:
        dm_sw = ne_sw * sw_geometry_pc
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

    # Time at emission (TDB - all delays) -- longdouble for phase precision
    tdb_mjd_ld = np.array(tdb_mjd, dtype=np.longdouble)
    tdb_sec = tdb_mjd_ld * np.longdouble(SECS_PER_DAY)
    dt_sec = tdb_sec - PEPOCH_sec - delay_sec

    # Phase computation is done by the shared function below (after TZR block)

    # TZR phase offset (if specified)
    tzr_phase = np.longdouble(0.0)
    if 'TZRMJD' in params:
        if verbose: print(f"\n   Computing TZR phase at TZRMJD...")
        tzr_phase = _compute_tzr_phase(
            params, bp, dm_jax, ddk,
            obs_clock, gps_clock, bipm_clock,
            observatory, location, obs_itrf_km,
            ra_rad, dec_rad, pmra_rad_day, pmdec_rad_day,
            posepoch, parallax_mas, ephem,
            F0, F1_half, F2_sixth, PEPOCH_sec,
            is_ecliptic, ssb_obs_pos_km, fd_coeffs,
            planet_shapiro_enabled,
            tzrmjd_scale, verbose,
        )

    # Phase computation + wrapping + conversion via shared canonical function.
    # Both evaluate-only and fitter paths call compute_phase_residuals() to guarantee
    # identical arithmetic (longdouble precision, Horner's method, np.round wrapping).
    errors_us = np.array([toa.error_us for toa in toas])
    weights = 1.0 / (errors_us ** 2)
    residuals_us, _ = compute_phase_residuals(
        dt_sec, params, weights, subtract_mean=True,
        tzr_phase=tzr_phase if subtract_tzr else None,
        jump_phase=jump_phase
    )

    # Compute weighted RMS using raw errors
    weighted_rms = np.sqrt(np.sum(weights * residuals_us**2) / np.sum(weights))

    # Compute weighted RMS with EFAC/EQUAD-scaled errors (PINT-compatible)
    noise_lines = params.get('_noise_lines', [])
    if noise_lines:
        from jug.noise.white import apply_white_noise, parse_noise_lines
        noise_entries = parse_noise_lines(noise_lines)
        toa_flags = [toa.flags for toa in toas]
        scaled_errors_us = apply_white_noise(errors_us, toa_flags, noise_entries)
        weights_scaled = 1.0 / (scaled_errors_us ** 2)
        wm_scaled = np.sum(residuals_us * weights_scaled) / np.sum(weights_scaled)
        weighted_rms_scaled = np.sqrt(np.sum(weights_scaled * (residuals_us - wm_scaled)**2) / np.sum(weights_scaled))
    else:
        scaled_errors_us = errors_us
        weighted_rms_scaled = weighted_rms

    # Also compute unweighted for comparison
    unweighted_rms = np.std(residuals_us)

    # Results
    if verbose: print(f"\n" + "=" * 60)
    if verbose: print(f"Results:")
    if verbose: print(f"  Weighted RMS: {weighted_rms:.3f} mus (raw errors)")
    if verbose: print(f"  Weighted RMS: {weighted_rms_scaled:.3f} mus (EFAC/EQUAD scaled)")
    if verbose: print(f"  Unweighted RMS: {unweighted_rms:.3f} mus")
    if verbose: print(f"  Mean: {np.mean(residuals_us):.3f} mus")
    if verbose: print(f"  Min: {np.min(residuals_us):.3f} mus")
    if verbose: print(f"  Max: {np.max(residuals_us):.3f} mus")
    if verbose: print(f"  N_TOAs: {len(residuals_us)}")
    if verbose: print("=" * 60)

    # Convert ssb_obs_pos from km to light-seconds for astrometry derivatives
    SPEED_OF_LIGHT_KM_S = C_KM_S
    ssb_obs_pos_ls = ssb_obs_pos_km / SPEED_OF_LIGHT_KM_S
    
    # Compute pre-binary delay: roemer_shapiro + DM + SW + tropo (NOT FD)
    # This is the PINT-compatible time for binary model evaluation
    prebinary_delay_sec = roemer_shapiro + dm_delay_sec + sw_delay_sec + tropo_delay_sec
    
    # Compute orbital phase (if binary)
    orbital_phase = None
    if has_binary:
        # Use PB from params (already derived if FB0 used)
        pb = float(params.get('PB', 0.0))
        if pb == 0.0 and 'FB0' in params:
             fb0 = float(params['FB0'])
             if fb0 != 0.0:
                 pb = (1.0 / fb0) / SECS_PER_DAY
        
        # Use T0 or TASC (already extracted/normalized above as t0_val/tasc_val)
        # Use the variable t0_val which holds T0 (or TASC if T0 missing)
        # For ELL1, TASC is the ascending node. Phase 0 is usually defined at TASC for ELL1?
        # Standard convention: Phase 0 is at T0 (periastron) or TASC (ascending node).
        # We use whatever is the reference epoch.
        ref_epoch = bp['t0_val'] if bp['t0_val'] != 0.0 else bp['tasc_val']
        
        if pb != 0.0 and ref_epoch != 0.0:
            # Phase = (t - T0) / PB
            # t is Barycentric time. Use tdb_mjd.
            # wrap to [0, 1)
            try:
                phases = (tdb_mjd - ref_epoch) / pb
                orbital_phase = phases - np.floor(phases)
            except Exception:
                orbital_phase = None

    return {
        'residuals_us': residuals_us,
        'rms_us': float(weighted_rms),  # Use weighted RMS as primary
        'weighted_rms_us': float(weighted_rms),
        'weighted_rms_scaled_us': float(weighted_rms_scaled),
        'unweighted_rms_us': float(unweighted_rms),
        'mean_us': float(np.mean(residuals_us)),
        'n_toas': len(residuals_us),
        'tdb_mjd': np.array(tdb_mjd, dtype=np.float64),
        'errors_us': errors_us,
        # Add computed delays for JAX fitting
        'total_delay_sec': np.array(total_delay_sec, dtype=np.float64),
        'freq_bary_mhz': np.array(freq_bary_mhz, dtype=np.float64),
        'tzr_phase': float(tzr_phase),
        # JUMP phase offsets (longdouble, for fitter to use)
        'jump_phase': np.array(jump_phase, dtype=np.longdouble),
        # Emission time offset from PEPOCH (longdouble for phase precision)
        'dt_sec_ld': np.array(dt_sec, dtype=np.longdouble),
        # Also float64 for backward compatibility
        'dt_sec': np.array(dt_sec, dtype=np.float64),
        # Roemer+Shapiro delay for computing barycentric times (legacy, for backward compat)
        'roemer_shapiro_sec': np.array(roemer_shapiro, dtype=np.float64),
        # Pre-binary delay: roemer_shapiro + DM + SW + tropo (PINT-compatible binary evaluation time)
        'prebinary_delay_sec': np.array(prebinary_delay_sec, dtype=np.float64),
        # Individual delay components (for diagnostics)
        'dm_delay_sec': np.array(dm_delay_sec, dtype=np.float64),
        'sw_delay_sec': np.array(sw_delay_sec, dtype=np.float64),
        'sw_geometry_pc': np.array(sw_geometry_pc, dtype=np.float64) if sw_geometry_pc is not None else None,
        'tropo_delay_sec': np.array(tropo_delay_sec, dtype=np.float64),
        # SSB to observatory position in light-seconds (needed for astrometry derivatives)
        'ssb_obs_pos_ls': np.array(ssb_obs_pos_ls, dtype=np.float64),
        'orbital_phase': orbital_phase,
        'toa_flags': [toa.flags for toa in toas],
        # Clock / EOP issues collected during loading (for GUI warnings)
        'clock_issues': clock_issues,
    }
