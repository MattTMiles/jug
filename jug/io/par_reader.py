"""Parser for Tempo2-style .par files.

This module handles parsing of pulsar timing model parameters from .par files
with special handling for high-precision parameters that require np.longdouble
to maintain microsecond-level timing accuracy.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from jug.utils.constants import HIGH_PRECISION_PARAMS


def parse_par_file(path: Path | str) -> Dict[str, Any]:
    """Parse tempo2-style .par file with high precision for timing-critical parameters.

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

    path = Path(path)
    par_filename = path.name.lower()
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].upper()
                value_str = parts[1]

                # Store high-precision parameters as strings
                if key in HIGH_PRECISION_PARAMS:
                    params_str[key] = value_str
                    try:
                        params[key] = float(value_str)
                    except ValueError:
                        params[key] = value_str
                else:
                    try:
                        params[key] = float(value_str)
                    except ValueError:
                        params[key] = value_str

    # Store high-precision string values
    params['_high_precision'] = params_str
    
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
        # No UNITS keyword - default to TDB
        # Warn if filename suggests TCB
        params['_par_timescale'] = 'TDB'
        if 'tcb' in par_filename:
            import warnings
            warnings.warn(
                f"Par file '{path.name}' has 'tcb' in filename but no UNITS keyword. "
                f"Assuming TDB timescale. If this is a TCB file, add 'UNITS TCB' to the par file."
            )
    
    return params


def validate_par_timescale(params: Dict[str, Any], context: str = "JUG") -> str:
    """Validate that the par file timescale is supported.
    
    Parameters
    ----------
    params : dict
        Parameter dictionary from parse_par_file()
    context : str
        Description of where this validation is happening (for error messages)
        
    Returns
    -------
    str
        The validated timescale ('TDB')
        
    Raises
    ------
    NotImplementedError
        If the timescale is TCB or TT (not yet supported)
    
    Notes
    -----
    JUG currently only supports TDB par files. TCB support would require:
    
    1. Transforming all epoch parameters (PEPOCH, T0, TASC, etc.) from TCB to TDB
    2. Applying the IAU-defined scaling factor L_B = 1.550519768e-8 to transform
       frequency-like parameters (F0, F1, F2, PB, FB0, etc.)
    3. Properly handling the TDB-TCB offset (~17 seconds per year since 1977)
    
    Until this is implemented, users should convert TCB par files to TDB using
    PINT or Tempo2 before using them with JUG.
    """
    timescale = params.get('_par_timescale', 'TDB')
    
    if timescale == 'TDB':
        return timescale
    elif timescale == 'TCB':
        raise NotImplementedError(
            f"{context}: TCB par files are not yet supported.\n\n"
            f"The par file has UNITS=TCB, indicating epochs and parameters are in "
            f"Barycentric Coordinate Time (TCB).\n\n"
            f"TCB support requires more than just epoch conversion - it also requires:\n"
            f"  1. Applying the IAU scaling factor L_B = 1.550519768e-8 to transform\n"
            f"     spin parameters (F0, F1, F2) and binary periods (PB, FB0, etc.)\n"
            f"  2. Converting epochs (PEPOCH, T0, TASC, TZRMJD, etc.) from TCB to TDB\n"
            f"  3. Properly accounting for the ~17 s/year drift between TCB and TDB\n\n"
            f"Workaround: Convert your par file to TDB using PINT or Tempo2:\n"
            f"  - PINT: model.as_parfile(units='TDB')\n"
            f"  - Tempo2: tempo2 -output par -units TDB -f <parfile> <timfile>\n"
        )
    elif timescale == 'TT':
        raise NotImplementedError(
            f"{context}: TT (Terrestrial Time) par files are not yet supported.\n\n"
            f"The par file has UNITS=TT. JUG currently only supports TDB par files.\n\n"
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

IMPORTANT: JUG currently only supports TDB par files. TCB par files will
trigger a NotImplementedError via validate_par_timescale().
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
        return np.longdouble(hp[key])
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
    
    return f"{sign}{d:02d}:{m:02d}:{s:010.7f}"
