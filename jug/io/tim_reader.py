"""Parser for Tempo2-style .tim files with TDB conversion.

This module handles parsing of TOA (Time of Arrival) data from .tim files
with full support for uncertainties, observatory codes, and flags. It also
provides standalone TDB calculation replacing PINT's clock correction chain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import EarthLocation

from jug.utils.constants import SECS_PER_DAY


@dataclass
class SimpleTOA:
    """Enhanced TOA structure for complete TIM file parsing.

    Attributes
    ----------
    mjd_str : str
        Original MJD string from TIM file (for precision tracking)
    mjd_int : int
        Integer part of MJD
    mjd_frac : float
        Fractional part of MJD
    freq_mhz : float
        Observing frequency in MHz
    error_us : float
        TOA uncertainty in microseconds
    observatory : str
        Observatory code (e.g., 'meerkat', 'parkes', 'gbt')
    flags : dict
        Additional flags from TIM file (e.g., {'fe': 'L-wide', 'be': 'GUPPI'})
    """
    mjd_str: str
    mjd_int: int
    mjd_frac: float
    freq_mhz: float
    error_us: float
    observatory: str = 'meerkat'
    flags: Dict[str, str] = field(default_factory=dict)


def parse_tim_file_mjds(path: Path | str) -> List[SimpleTOA]:
    """Parse TIM file to extract all TOA information.

    Extracts:
    - MJD values (high precision split into int + frac)
    - Observing frequencies
    - TOA uncertainties (errors)
    - Observatory codes
    - Additional flags (e.g., -fe, -be, -sys)

    Parameters
    ----------
    path : Path or str
        Path to .tim file

    Returns
    -------
    list of SimpleTOA
        List of parsed TOA objects

    Notes
    -----
    TIM file format (IPTA/Tempo2):
        observatory freq mjd error [flags...]

    Example TIM line:
        meerkat 1284.0 58000.123456789 1.5 -fe L-wide -be PTUSE

    Examples
    --------
    >>> toas = parse_tim_file_mjds("J0437-4715.tim")
    >>> print(f"Loaded {len(toas)} TOAs")
    >>> print(f"First TOA: MJD={toas[0].mjd_int}.{toas[0].mjd_frac}, freq={toas[0].freq_mhz} MHz")
    """
    toas = []
    path = Path(path)
    # Default to Tempo2 FORMAT 1 (filename freq mjd error site [flags...])
    tim_format = 1

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines, comments, and directives
            if not line or line.startswith('#'):
                continue
            if line.startswith('C '):
                continue

            # Track FORMAT/MODE directives
            if line.startswith('FORMAT'):
                parts_fmt = line.split()
                if len(parts_fmt) >= 2:
                    tim_format = int(parts_fmt[1])
                continue
            if line.startswith('MODE'):
                parts_fmt = line.split()
                if len(parts_fmt) >= 2:
                    tim_format = int(parts_fmt[1])
                continue
            if line.startswith(('JUMP', 'PHASE', 'INCLUDE', 'END')):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            if tim_format == 1:
                # FORMAT 1 (Tempo2): filename freq mjd error site [flags...]
                if len(parts) < 5:
                    continue
                freq_mhz = float(parts[1])
                mjd_str = parts[2]
                error_us = float(parts[3])
                observatory = parts[4].lower()
                flag_start = 5
            else:
                # Princeton format: site freq mjd error [flags...]
                observatory = parts[0].lower()
                freq_mhz = float(parts[1])
                mjd_str = parts[2]
                error_us = float(parts[3])
                flag_start = 4

            # Parse MJD with high precision
            mjd_int, mjd_frac = parse_mjd_string(mjd_str)

            # Parse optional flags (format: -flag value)
            flags = {}
            i = flag_start
            while i < len(parts):
                if parts[i].startswith('-') and i + 1 < len(parts):
                    flag_name = parts[i][1:]  # Remove leading '-'
                    flag_value = parts[i + 1]
                    flags[flag_name] = flag_value
                    i += 2
                else:
                    i += 1

            toas.append(SimpleTOA(
                mjd_str=mjd_str,
                mjd_int=mjd_int,
                mjd_frac=mjd_frac,
                freq_mhz=freq_mhz,
                error_us=error_us,
                observatory=observatory,
                flags=flags
            ))

    return toas


def parse_mjd_string(mjd_str: str) -> tuple[int, float]:
    """Parse high-precision MJD string into (int, frac) components.

    Preserves full precision by keeping fractional part separate.

    Parameters
    ----------
    mjd_str : str
        MJD string (e.g., "58000.123456789")

    Returns
    -------
    mjd_int : int
        Integer part of MJD
    mjd_frac : float
        Fractional part of MJD

    Examples
    --------
    >>> mjd_int, mjd_frac = parse_mjd_string("58000.123456789")
    >>> print(f"MJD = {mjd_int} + {mjd_frac}")
    MJD = 58000 + 0.123456789
    """
    if '.' in mjd_str:
        int_str, frac_str = mjd_str.split('.')
        mjd_int = int(int_str)
        mjd_frac = float('0.' + frac_str)
    else:
        mjd_int = int(mjd_str)
        mjd_frac = 0.0

    return mjd_int, mjd_frac


def compute_tdb_standalone_vectorized(
    mjd_ints, mjd_fracs,
    mk_clock, gps_clock, bipm_clock,
    location: EarthLocation,
    time_offsets: np.ndarray | None = None
) -> np.ndarray:
    """Compute TDB from UTC MJDs using standalone clock chain (VECTORIZED).

    This is ~10x faster than per-TOA version by vectorizing clock
    corrections and creating Time objects in batches.

    Parameters
    ----------
    mjd_ints : array-like
        Integer parts of UTC MJDs
    mjd_fracs : array-like
        Fractional parts of UTC MJDs
    mk_clock : dict
        MeerKAT (or observatory) clock correction data {'mjd': array, 'offset': array}
    gps_clock : dict
        GPS clock correction data {'mjd': array, 'offset': array}
    bipm_clock : dict
        BIPM clock correction data (TAI→TT) {'mjd': array, 'offset': array}
    location : EarthLocation
        Observatory location for TDB conversion
    time_offsets : np.ndarray or None, optional
        Per-TOA time offsets in seconds (e.g. from TIM ``-to`` flags).
        Added to the clock corrections before TDB conversion.

    Returns
    -------
    np.ndarray
        TDB MJD values as np.longdouble for full precision

    Notes
    -----
    Clock correction chain:
        Observatory → UTC → GPS/TAI → TT → TDB

    The BIPM correction includes the TAI→TT offset (32.184 seconds).

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation
    >>> location = EarthLocation.of_site('meerkat')
    >>> tdb_mjds = compute_tdb_standalone_vectorized(
    ...     mjd_ints=[58000, 58001],
    ...     mjd_fracs=[0.5, 0.5],
    ...     mk_clock=mk_data,
    ...     gps_clock=gps_data,
    ...     bipm_clock=bipm_data,
    ...     location=location
    ... )
    """
    from jug.io.clock import interpolate_clock_vectorized

    n_toas = len(mjd_ints)
    mjd_vals = np.array(mjd_ints, dtype=np.float64) + np.array(mjd_fracs, dtype=np.float64)

    # Vectorized clock corrections (10x faster using searchsorted)
    mk_corrs = interpolate_clock_vectorized(mk_clock, mjd_vals)
    gps_corrs = interpolate_clock_vectorized(gps_clock, mjd_vals)
    bipm_corrs = np.interp(mjd_vals, bipm_clock['mjd'], bipm_clock['offset']) - 32.184

    total_corrs = mk_corrs + gps_corrs + bipm_corrs

    # Add per-TOA time offsets (e.g. TIM -to flags, TEMPO TIME statements)
    if time_offsets is not None:
        total_corrs = total_corrs + np.asarray(time_offsets, dtype=np.float64)

    # Create Time objects in batch (much faster than per-TOA)
    time_utc = Time(
        val=np.array(mjd_ints, dtype=np.float64),
        val2=np.array(mjd_fracs, dtype=np.float64),
        format='mjd',
        scale='utc',
        location=location,
        precision=9
    )

    # Apply clock corrections
    time_utc = time_utc + TimeDelta(total_corrs, format='sec')

    # Convert to TDB (vectorized)
    # Return TDB with full precision using double-double representation
    tdb_time = time_utc.tdb

    # Extract as longdouble: (jd1 - MJD_offset) + jd2
    MJD_OFFSET = 2400000.5
    tdb_mjd = np.array(tdb_time.jd1 - MJD_OFFSET, dtype=np.longdouble) + \
              np.array(tdb_time.jd2, dtype=np.longdouble)

    return tdb_mjd
