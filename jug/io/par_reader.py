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
    return params


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
