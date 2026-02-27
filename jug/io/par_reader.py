"""Parser for pulsar timing .par files.

This module handles parsing of pulsar timing model parameters from .par files
(Tempo2, PINT, and NANOGrav conventions) with special handling for high-precision
parameters that require np.longdouble to maintain microsecond-level timing accuracy.

TCB par files are automatically detected and converted to TDB internally.

Ecliptic coordinate support
---------------------------
Par files may specify pulsar positions in ecliptic coordinates (LAMBDA/BETA
or ELONG/ELAT) instead of equatorial (RAJ/DECJ). This module converts ecliptic
coordinates to equatorial at parse time so all downstream code can uniformly
use RAJ/DECJ/PMRA/PMDEC. The original ecliptic values and the ECL frame used
are preserved under private keys for round-trip fidelity.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from jug.utils.constants import HIGH_PRECISION_PARAMS


# Obliquity of the ecliptic in arcseconds, matching PINT/Tempo2 conventions.
# Source: IERS Technical Notes and IAU resolutions.
OBLIQUITY_ARCSEC = {
    'IAU1976': 84381.448,
    'IERS1992': 84381.412,
    'DE403': 84381.412,
    'IERS2003': 84381.4059,
    'IERS2010': 84381.406000,
    'IAU2005': 84381.406000,
    'DEFAULT': 84381.406000,
}


def _parse_float(value_str: str) -> float:
    """Parse a float string, handling Fortran-style D exponent notation.

    Parameters
    ----------
    value_str : str
        Numeric string, possibly with 'D' or 'd' exponent (e.g., '1.23D-04').

    Returns
    -------
    float
        The parsed value.

    Raises
    ------
    ValueError
        If the string cannot be parsed as a float.
    """
    try:
        return float(value_str)
    except ValueError:
        # Handle Fortran D-notation: replace 'D' or 'd' with 'E'
        converted = value_str.replace('D', 'E').replace('d', 'e')
        return float(converted)


def parse_par_file(path: Path | str) -> Dict[str, Any]:
    """Parse a pulsar timing .par file with high precision for timing-critical parameters.

    Parameters
    ----------
    path : Path or str
        Path to .par file

    Returns
    -------
    dict
        Dictionary of parameters with values. Includes special '_high_precision'
        key containing original string representations of high-precision parameters.

    Notes
    -----
    High-precision parameters (F0, F1, F2, PEPOCH, etc.) are stored both as floats
    and as their original string representations for later conversion to np.longdouble
    when needed. This preserves maximum precision.

    Examples
    --------
    >>> params = parse_par_file("J0437-4715.par")
    >>> f0 = get_longdouble(params, 'F0')
    >>> ra_rad = parse_ra(params['RAJ'])
    """
    params = {}
    params_str = {}
    fit_flags = {}  # Track which parameters have fit flag = 1
    noise_lines = []  # Collect EFAC/EQUAD/ECORR lines (non-standard multi-token format)
    jump_lines = []   # Collect JUMP lines (multi-token format: JUMP -flag val value)

    # Keywords that use multi-token format: T2EFAC -f <val> <num>, etc.
    _NOISE_KEYWORDS = {'T2EFAC', 'T2EQUAD', 'EFAC', 'EQUAD', 'ECORR', 'TNECORR', 'DMEFAC', 'DMJUMP'}

    path = Path(path)
    par_filename = path.name.lower()
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            # '#' is standard; 'C ' is a Tempo/Tempo2 comment convention
            if not line or line.startswith('#') or line.upper().startswith('C '):
                continue

            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].upper()

                # Noise parameters have multi-token format -- store raw line
                if key in _NOISE_KEYWORDS:
                    noise_lines.append(line)
                    continue

                # JUMP parameters have multi-token format -- store raw line
                if key == 'JUMP':
                    jump_lines.append(line)
                    continue

                value_str = parts[1]

                # Store high-precision parameters as strings
                if key in HIGH_PRECISION_PARAMS:
                    params_str[key] = value_str
                    try:
                        params[key] = _parse_float(value_str)
                    except ValueError:
                        params[key] = value_str
                else:
                    try:
                        params[key] = _parse_float(value_str)
                    except ValueError:
                        params[key] = value_str

                # Extract fit flag: parts[2] is '1' or '0' if present
                if len(parts) >= 3 and parts[2] in ('0', '1'):
                    if parts[2] == '1':
                        fit_flags[key] = True

    # Store high-precision string values
    params['_high_precision'] = params_str

    # Store fit flags (parameters with fit flag = 1 in par file)
    params['_fit_flags'] = fit_flags
    
    # Store raw noise lines for later parsing by jug.noise.white
    if noise_lines:
        params['_noise_lines'] = noise_lines

    # Store raw JUMP lines for later parsing
    if jump_lines:
        params['_jump_lines'] = jump_lines
        # Extract JUMP values as JUMP1, JUMP2, ... for the fitter
        for idx, jl in enumerate(jump_lines):
            jparts = jl.strip().split()
            # Flag-based: JUMP -flag val offset [fit] [unc]
            # MJD-based:  JUMP MJD start end offset [fit] [unc]
            try:
                if jparts[1].upper() == 'MJD':
                    val = float(jparts[4])
                    fit_idx = 5
                else:
                    val = float(jparts[3])
                    fit_idx = 4
            except (IndexError, ValueError):
                val = 0.0
                fit_idx = -1
            jump_key = f'JUMP{idx + 1}'
            params[jump_key] = val
            if fit_idx > 0 and len(jparts) > fit_idx and jparts[fit_idx] == '1':
                fit_flags[jump_key] = True
    
    # Determine and store par file timescale
    # UNITS keyword is the authoritative source (PINT/Tempo2 convention)
    if 'UNITS' in params:
        units_val = str(params['UNITS']).upper()
        if units_val in ('TDB', 'TCB', 'TT'):
            params['_par_timescale'] = units_val
        else:
            # Unknown UNITS value - default to TDB with warning
            import warnings
            warnings.warn(f"Unknown UNITS value '{params['UNITS']}' in par file, assuming TDB")
            params['_par_timescale'] = 'TDB'
    else:
        # No UNITS keyword — follow TEMPO2 convention:
        #   EPHVER < 5 (or TEMPO1 keyword present) → TDB
        #   EPHVER >= 5 (or EPHVER absent, defaulting to 5) → TCB/SI
        ephver = int(float(params.get('EPHVER', 5)))
        is_tempo1 = 'TEMPO1' in params
        if is_tempo1 or ephver < 5:
            params['_par_timescale'] = 'TDB'
        else:
            params['_par_timescale'] = 'TCB'
    
    # Convert ecliptic coordinates to equatorial if needed
    _convert_ecliptic_to_equatorial(params)

    return params


def _convert_ecliptic_to_equatorial(params: Dict[str, Any]) -> None:
    """Convert ecliptic coordinates (LAMBDA/BETA or ELONG/ELAT) to equatorial.

    Modifies ``params`` in place, replacing ecliptic position/proper-motion
    parameters with their equatorial equivalents (RAJ, DECJ, PMRA, PMDEC).
    If the par file already uses equatorial coordinates, this is a no-op.

    The conversion uses a pure rotation about the x-axis by the obliquity of
    the ecliptic, matching the ``PulsarEcliptic`` frame in PINT/Tempo2.

    The obliquity is determined by the ``ECL`` keyword in the par file
    (default: IERS2010, epsilon = 84381.406" = 23deg26'21.406").

    Parameters
    ----------
    params : dict
        Parameter dictionary from parse_par_file(), modified in place.

    Notes
    -----
    PINT convention for proper motions:
    - PMLAMBDA = dlambda_*cos(beta)  [mas/yr]  (ecliptic)
    - PMRA     = dalpha*cos(delta)  [mas/yr]  (equatorial)
    Both include the cos(lat) factor.
    """
    # Determine which ecliptic keywords are present (LAMBDA/BETA or ELONG/ELAT)
    has_lambda = 'LAMBDA' in params
    has_elong = 'ELONG' in params

    if not has_lambda and not has_elong:
        return  # Equatorial -- nothing to do

    # If RAJ/DECJ are already present, prefer them (ecliptic is informational)
    if 'RAJ' in params:
        return

    # Read ecliptic longitude/latitude in degrees
    if has_lambda:
        ecl_lon_deg = float(params['LAMBDA'])
        ecl_lat_deg = float(params['BETA'])
        pm_lon_key = 'PMLAMBDA'
        pm_lat_key = 'PMBETA'
    else:
        ecl_lon_deg = float(params['ELONG'])
        ecl_lat_deg = float(params['ELAT'])
        pm_lon_key = 'PMELONG'
        pm_lat_key = 'PMELAT'

    # Determine obliquity from ECL keyword (default IERS2010)
    ecl_frame = str(params.get('ECL', 'IERS2010')).upper()
    if ecl_frame not in OBLIQUITY_ARCSEC:
        import warnings
        warnings.warn(
            f"Unknown ECL frame '{ecl_frame}' in par file, "
            f"using IERS2010 obliquity (84381.406 arcsec)"
        )
        ecl_frame = 'IERS2010'

    obl_rad = OBLIQUITY_ARCSEC[ecl_frame] * np.pi / (180.0 * 3600.0)

    # --- Position conversion: ecliptic -> equatorial ---
    lon_rad = np.radians(ecl_lon_deg)
    lat_rad = np.radians(ecl_lat_deg)

    cos_lon = np.cos(lon_rad)
    sin_lon = np.sin(lon_rad)
    cos_lat = np.cos(lat_rad)
    sin_lat = np.sin(lat_rad)
    cos_obl = np.cos(obl_rad)
    sin_obl = np.sin(obl_rad)

    # Cartesian in ecliptic frame
    x = cos_lon * cos_lat
    y = sin_lon * cos_lat
    z = sin_lat

    # Rotate about x-axis by -epsilon (ecliptic -> equatorial)
    x_eq = x
    y_eq = y * cos_obl - z * sin_obl
    z_eq = y * sin_obl + z * cos_obl

    ra_rad = np.arctan2(y_eq, x_eq) % (2 * np.pi)
    dec_rad = np.arctan2(z_eq, np.sqrt(x_eq**2 + y_eq**2))

    # Store equatorial position as sexagesimal strings (expected by downstream)
    params['RAJ'] = format_ra(ra_rad)
    params['DECJ'] = format_dec(dec_rad)

    # --- Proper motion conversion: ecliptic -> equatorial ---
    pm_lon = params.get(pm_lon_key, 0.0)  # mas/yr, includes cos(lat)
    pm_lat = params.get(pm_lat_key, 0.0)  # mas/yr

    if pm_lon != 0.0 or pm_lat != 0.0:
        # The PM vector in ecliptic spherical coords (cos-lat already included):
        #   v_lon = pm_lon  (= dlambda_*cos(beta))
        #   v_lat = pm_lat  (= dbeta)
        # Convert to Cartesian velocity:
        #   dr/dlambda_ = (-sin(lambda_)cos(beta), cos(lambda_)cos(beta), 0) / cos(beta) [for cos-lat PM]
        #   dr/dbeta = (-cos(lambda_)sin(beta), -sin(lambda_)sin(beta), cos(beta))

        # Jacobian columns for (dlambda_*cos(beta), dbeta) -> Cartesian
        # For dlambda_*cos(beta): divide by cos(beta) to get dlambda_, multiply by dr/dlambda_
        # dr/dlambda_ = (-sin(lambda_)cos(beta), cos(lambda_)cos(beta), 0)
        # So (dlambda_*cos(beta)) * (dr/dlambda_ / cos(beta)) = dlambda_*cos(beta) * (-sin(lambda_), cos(lambda_), 0)
        dx = -sin_lon * pm_lon - cos_lon * sin_lat * pm_lat
        dy = cos_lon * pm_lon - sin_lon * sin_lat * pm_lat
        dz = cos_lat * pm_lat

        # Rotate to equatorial
        dx_eq = dx
        dy_eq = dy * cos_obl - dz * sin_obl
        dz_eq = dy * sin_obl + dz * cos_obl

        # Project back to spherical: pm_ra_cosdec and pm_dec
        cos_ra = np.cos(ra_rad)
        sin_ra = np.sin(ra_rad)
        cos_dec = np.cos(dec_rad)
        sin_dec = np.sin(dec_rad)

        # pm_ra*cos(dec) = -sin(ra)*dx_eq + cos(ra)*dy_eq
        # pm_dec = -cos(ra)*sin(dec)*dx_eq - sin(ra)*sin(dec)*dy_eq + cos(dec)*dz_eq
        pmra = -sin_ra * dx_eq + cos_ra * dy_eq
        pmdec = -cos_ra * sin_dec * dx_eq - sin_ra * sin_dec * dy_eq + cos_dec * dz_eq

        params['PMRA'] = pmra
        params['PMDEC'] = pmdec

    # Preserve original ecliptic values and flag for round-trip support
    params['_ecliptic_coords'] = True
    params['_ecliptic_frame'] = ecl_frame
    params['_ecliptic_lon_deg'] = ecl_lon_deg
    params['_ecliptic_lat_deg'] = ecl_lat_deg
    params['_ecliptic_pm_lon'] = pm_lon
    params['_ecliptic_pm_lat'] = pm_lat


def validate_par_timescale(params: Dict[str, Any], context: str = "JUG", 
                           verbose: bool = False) -> str:
    """Validate and convert par file timescale to TDB if needed.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary from parse_par_file() (modified in place if TCB)
    context : str
        Description of where this validation is happening (for error messages)
    verbose : bool
        If True, print conversion details
        
    Returns
    -------
    str
        The validated/converted timescale (always 'TDB')
        
    Raises
    ------
    NotImplementedError
        If the timescale is TT (not yet supported)
    
    Notes
    -----
    JUG internally works in TDB (Barycentric Dynamical Time). If a TCB
    (Barycentric Coordinate Time) par file is provided, this function
    automatically converts all parameters to TDB following the Irwin &
    Fukushima (1999) convention:
    
    1. Epoch parameters (PEPOCH, T0, TASC, etc.) are transformed from TCB to TDB
    2. Frequency/DM/orbital parameters are scaled by IFTE_K factors
    3. The conversion is applied in-place and logged if verbose=True
    
    The conversion matches PINT and Tempo2 behavior.
    """
    from jug.utils.timescales import convert_par_params_to_tdb, parse_timescale
    
    timescale = parse_timescale(params)
    
    if timescale == 'TDB':
        return timescale
    elif timescale == 'TCB':
        # Convert TCB to TDB in place
        _, log = convert_par_params_to_tdb(params, verbose=verbose)
        
        if verbose:
            print(f"{context}: Detected TCB par file, converting to internal TDB representation")
            for msg in log:
                print(f"  {msg}")
        
        return 'TDB'
    elif timescale == 'TT':
        raise NotImplementedError(
            f"{context}: TT (Terrestrial Time) par files are not yet supported.\n\n"
            f"The par file has UNITS=TT. JUG currently only supports TDB/TCB par files.\n\n"
            f"Workaround: Convert your par file to TDB using PINT or Tempo2.\n"
        )
    else:
        raise ValueError(f"{context}: Unknown par file timescale '{timescale}'")


# List of epoch-like parameters that are in the par file timescale (non-exhaustive)
EPOCH_PARAMETERS = {
    'PEPOCH',    # Spin epoch
    'POSEPOCH',  # Position epoch  
    'DMEPOCH',   # DM epoch
    'T0',        # Binary epoch (DD, BT, T2)
    'TASC',      # Binary epoch (ELL1)
    'TZRMJD',    # Reference TOA epoch
    'START',     # Data span start
    'FINISH',    # Data span end
    'PBEPOCH',   # Period derivative reference epoch (rare)
    'GLEP_1', 'GLEP_2', 'GLEP_3', 'GLEP_4', 'GLEP_5',  # Glitch epochs
}
"""Parameters that represent absolute epochs in the par file timescale.

NOTE: This set is for documentation only and may not be exhaustive.
Other epoch-like parameters may exist (e.g., DMX epochs, noise model epochs).

These parameters are stored as MJD values in whatever timescale the par file
uses (TDB or TCB, as indicated by the UNITS keyword). JUG treats them as
already being in the model timescale and does NOT apply UTC->TDB clock
corrections to them.

The par file timescale is determined by:
1. The UNITS keyword (TDB or TCB) if present
2. Default to TDB if UNITS is not specified

TCB par files are automatically converted to TDB via validate_par_timescale().
"""


def get_longdouble(params: Dict[str, Any], key: str, default: Optional[float] = None) -> np.longdouble:
    """Get a parameter as np.longdouble with full precision.

    Parameters
    ----------
    params : dict
        Parameter dictionary from parse_par_file()
    key : str
        Parameter name (case-insensitive)
    default : float, optional
        Default value if parameter not found

    Returns
    -------
    np.longdouble
        Parameter value with maximum precision

    Raises
    ------
    KeyError
        If parameter not found and no default provided

    Examples
    --------
    >>> params = parse_par_file("J0437-4715.par")
    >>> f0 = get_longdouble(params, 'F0')
    >>> print(f"{f0:.20Lf}")  # Print with full precision
    """
    hp = params.get('_high_precision', {})
    key = key.upper()

    if key in hp:
        # Use original string for maximum precision
        val_str = hp[key]
        # Handle Fortran D-notation (e.g., -6.205D-16)
        if 'D' in val_str or 'd' in val_str:
            val_str = val_str.replace('D', 'E').replace('d', 'e')
        return np.longdouble(val_str)
    elif key in params:
        return np.longdouble(params[key])
    elif default is not None:
        return np.longdouble(default)
    else:
        raise KeyError(f"Parameter {key} not found in .par file")


def parse_ra(ra_str: str) -> float:
    """Parse RA string (HH:MM:SS.sss) to radians.

    Parameters
    ----------
    ra_str : str
        Right ascension in format "HH:MM:SS.sss"

    Returns
    -------
    float
        Right ascension in radians

    Examples
    --------
    >>> ra_rad = parse_ra("06:37:24.0")
    >>> ra_deg = ra_rad * 180 / np.pi
    """
    # If already a float (radians), return as-is
    if isinstance(ra_str, (int, float)):
        return float(ra_str)
    parts = ra_str.split(':')
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    ra_hours = h + m/60.0 + s/3600.0
    ra_deg = ra_hours * 15.0
    return ra_deg * np.pi / 180.0


def parse_dec(dec_str: str) -> float:
    """Parse DEC string (DD:MM:SS.sss) to radians.

    Parameters
    ----------
    dec_str : str
        Declination in format "DD:MM:SS.sss" or "-DD:MM:SS.sss"

    Returns
    -------
    float
        Declination in radians

    Examples
    --------
    >>> dec_rad = parse_dec("-47:15:09.0")
    >>> dec_deg = dec_rad * 180 / np.pi
    """
    # If already a float (radians), return as-is
    if isinstance(dec_str, (int, float)):
        return float(dec_str)
    parts = dec_str.split(':')
    sign = -1 if parts[0].startswith('-') else 1
    d, m, s = abs(float(parts[0])), float(parts[1]), float(parts[2])
    dec_deg = sign * (d + m/60.0 + s/3600.0)
    return dec_deg * np.pi / 180.0


def format_ra(ra_rad: float) -> str:
    """Format RA in radians to HMS string.

    Parameters
    ----------
    ra_rad : float
        Right ascension in radians

    Returns
    -------
    str
        RA in format "HH:MM:SS.sssssss"
    """
    ra_deg = ra_rad * 180.0 / np.pi
    ra_hours = ra_deg / 15.0
    
    h = int(ra_hours)
    m_frac = (ra_hours - h) * 60.0
    m = int(m_frac)
    s = (m_frac - m) * 60.0
    
    # Guard against floating-point rounding producing 60.0 seconds
    if s >= 59.999999995:
        s = 0.0
        m += 1
        if m >= 60:
            m = 0
            h += 1
            if h >= 24:
                h = 0
    
    return f"{h:02d}:{m:02d}:{s:011.8f}"


def format_dec(dec_rad: float) -> str:
    """Format DEC in radians to DMS string.

    Parameters
    ----------
    dec_rad : float
        Declination in radians

    Returns
    -------
    str
        DEC in format "DD:MM:SS.ssssss"
    """
    dec_deg = dec_rad * 180.0 / np.pi
    sign = '-' if dec_deg < 0 else ''
    dec_deg = abs(dec_deg)
    
    d = int(dec_deg)
    m_frac = (dec_deg - d) * 60.0
    m = int(m_frac)
    s = (m_frac - m) * 60.0
    
    # Guard against floating-point rounding producing 60.0 seconds
    if s >= 59.99999995:
        s = 0.0
        m += 1
        if m >= 60:
            m = 0
            d += 1
    
    return f"{sign}{d:02d}:{m:02d}:{s:010.7f}"


def convert_equatorial_to_ecliptic(
    ra_rad: float,
    dec_rad: float,
    pmra: float = 0.0,
    pmdec: float = 0.0,
    ecl_frame: str = 'IERS2010'
) -> Dict[str, float]:
    """Convert equatorial coordinates to ecliptic.

    Inverse of the ecliptic->equatorial conversion in
    ``_convert_ecliptic_to_equatorial``.

    Parameters
    ----------
    ra_rad : float
        Right ascension in radians.
    dec_rad : float
        Declination in radians.
    pmra : float
        Proper motion in RA (mas/yr, includes cos(dec) factor).
    pmdec : float
        Proper motion in DEC (mas/yr).
    ecl_frame : str
        Ecliptic frame name for obliquity lookup (default ``'IERS2010'``).

    Returns
    -------
    dict
        Keys: ``'LAMBDA'``, ``'BETA'`` (degrees),
        ``'PMLAMBDA'``, ``'PMBETA'`` (mas/yr with cos-lat factor).
    """
    obl_rad = OBLIQUITY_ARCSEC.get(ecl_frame, OBLIQUITY_ARCSEC['IERS2010']) * np.pi / (180.0 * 3600.0)

    cos_ra = np.cos(ra_rad)
    sin_ra = np.sin(ra_rad)
    cos_dec = np.cos(dec_rad)
    sin_dec = np.sin(dec_rad)
    cos_obl = np.cos(obl_rad)
    sin_obl = np.sin(obl_rad)

    # Cartesian in equatorial
    x = cos_ra * cos_dec
    y = sin_ra * cos_dec
    z = sin_dec

    # Rotate about x-axis by +epsilon (equatorial -> ecliptic)
    x_ecl = x
    y_ecl = y * cos_obl + z * sin_obl
    z_ecl = -y * sin_obl + z * cos_obl

    lon_rad = np.arctan2(y_ecl, x_ecl) % (2 * np.pi)
    lat_rad = np.arctan2(z_ecl, np.sqrt(x_ecl**2 + y_ecl**2))

    result = {
        'LAMBDA': np.degrees(lon_rad),
        'BETA': np.degrees(lat_rad),
        'PMLAMBDA': 0.0,
        'PMBETA': 0.0,
    }

    if pmra != 0.0 or pmdec != 0.0:
        # Cartesian velocity in equatorial
        dx = -sin_ra * pmra - cos_ra * sin_dec * pmdec
        dy = cos_ra * pmra - sin_ra * sin_dec * pmdec
        dz = cos_dec * pmdec

        # Rotate to ecliptic
        dx_ecl = dx
        dy_ecl = dy * cos_obl + dz * sin_obl
        dz_ecl = -dy * sin_obl + dz * cos_obl

        # Project to spherical: pm_lon*cos(lat) and pm_lat
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)

        result['PMLAMBDA'] = -sin_lon * dx_ecl + cos_lon * dy_ecl
        result['PMBETA'] = -cos_lon * sin_lat * dx_ecl - sin_lon * sin_lat * dy_ecl + cos_lat * dz_ecl

    return result
