"""Parser for Tempo2-style .tim files with TDB conversion.

This module handles parsing of TOA (Time of Arrival) data from .tim files
with full support for uncertainties, observatory codes, and flags. It also
provides standalone TDB calculation replacing PINT's clock correction chain.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict
import numpy as np
import erfa
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


def parse_tim_file_mjds(path: Path | str, _state: dict | None = None) -> List[SimpleTOA]:
    """Parse TIM file to extract all TOA information.

    Extracts:
    - MJD values (high precision split into int + frac)
    - Observing frequencies
    - TOA uncertainties (errors)
    - Observatory codes
    - Additional flags (e.g., -fe, -be, -sys)

    Supports TEMPO2 ``TIME`` directives which add cumulative time offsets
    (in seconds) to subsequent TOA MJDs.

    Parameters
    ----------
    path : Path or str
        Path to .tim file
    _state : dict or None
        Internal parsing state (for recursive INCLUDE handling).
        Users should not pass this parameter.

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
    if _state is None:
        _state = {'time_offset': 0.0, 'tim_format': 1}
    # Default to Tempo2 FORMAT 1 (filename freq mjd error site [flags...])
    tim_format = _state['tim_format']

    with open(path) as f:
        for line in f:
            line = line.strip()

            # Skip empty lines, comments, and directives
            if not line or line.startswith('#'):
                continue

            # Tempo2 convention: any line starting with uppercase 'C'
            # (that isn't a directive) is a comment.
            if line[0] == 'C' and len(line) > 1 and line[1] != ' ':
                # Could be CC, CC?, C??, Cfilename — all comments
                if not line.startswith(('CLK', 'CLOCK')):
                    continue
            if line.startswith('C '):
                continue

            # 'end' directive: stop reading current file (Tempo2 convention)
            if line.lower().startswith('end'):
                break

            # Track FORMAT/MODE directives
            if line.startswith('FORMAT'):
                parts_fmt = line.split()
                if len(parts_fmt) >= 2:
                    tim_format = int(parts_fmt[1])
                    _state['tim_format'] = tim_format
                continue
            if line.startswith('MODE'):
                parts_fmt = line.split()
                if len(parts_fmt) >= 2:
                    tim_format = int(parts_fmt[1])
                    _state['tim_format'] = tim_format
                continue
            if line.startswith('INCLUDE'):
                inc_parts = line.split(None, 1)
                if len(inc_parts) == 2:
                    inc_path = path.parent / inc_parts[1].strip()
                    # Tempo2 scopes TIME and FORMAT to each file
                    # (local variables in readTim reset per recursive call)
                    toas.extend(parse_tim_file_mjds(inc_path, _state=None))
                continue
            # TIME directive: cumulative time offset in seconds added to MJDs
            if line.startswith('TIME'):
                parts_time = line.split()
                if len(parts_time) >= 2:
                    _state['time_offset'] += float(parts_time[1])
                continue
            if line.startswith(('JUMP', 'PHASE')):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            if tim_format == 1:
                # FORMAT 1 (Tempo2): filename freq mjd error site [flags...]
                if len(parts) < 5:
                    continue
                try:
                    freq_mhz = float(parts[1])
                except ValueError:
                    # Unparseable frequency — likely a commented-out TOA
                    # (e.g. 'C' prepended to filename: Cc059968.align...)
                    continue
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

            # Apply cumulative TIME offset (seconds -> fractional day)
            if _state['time_offset'] != 0.0:
                mjd_frac += _state['time_offset'] / 86400.0
                # Normalize: handle overflow/underflow of fractional day
                if mjd_frac >= 1.0:
                    mjd_int += int(mjd_frac)
                    mjd_frac -= int(mjd_frac)
                elif mjd_frac < 0.0:
                    shift = int(-mjd_frac) + 1
                    mjd_int -= shift
                    mjd_frac += shift

            # Parse optional flags (format: -flag value)
            # Duplicate flag names (e.g. -j MEDUSA_58925 -j MEDUSA_59200) are
            # stored as lists so JUMP matching can check all values.
            flags = {}
            i = flag_start
            while i < len(parts):
                if parts[i].startswith('-') and i + 1 < len(parts):
                    flag_name = parts[i][1:]  # Remove leading '-'
                    flag_value = parts[i + 1]
                    if flag_name in flags:
                        existing = flags[flag_name]
                        if isinstance(existing, list):
                            existing.append(flag_value)
                        else:
                            flags[flag_name] = [existing, flag_value]
                    else:
                        flags[flag_name] = flag_value
                    i += 2
                else:
                    i += 1

            # Apply -addsat flag: adds integer seconds to TOA (satellite pass correction)
            if 'addsat' in flags:
                try:
                    addsat_sec = float(flags['addsat'])
                    mjd_frac += addsat_sec / 86400.0
                    if mjd_frac >= 1.0:
                        mjd_int += int(mjd_frac)
                        mjd_frac -= int(mjd_frac)
                    elif mjd_frac < 0.0:
                        shift = int(-mjd_frac) + 1
                        mjd_int -= shift
                        mjd_frac += shift
                except (ValueError, TypeError):
                    pass

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
    obs_chain, bipm_clock,
    location: EarthLocation,
    time_offsets: np.ndarray | None = None,
    # Legacy keyword arguments kept for backward compatibility; ignored.
    gps_clock=None, mk_clock=None, skip_gps_correction=None,
) -> np.ndarray:
    """Compute TDB from UTC MJDs using a pre-merged observatory clock chain.

    This is ~10x faster than per-TOA version by vectorizing clock
    corrections and creating Time objects in batches.

    The clock chain has already been resolved by :class:`ClockGraph` in
    ``simple_calculator.py``: *obs_chain* is the merged ``UTC(obs) → UTC``
    correction (sum of all hops along the Dijkstra path).  This mirrors
    Tempo2's design exactly — the graph finds the shortest path and the
    corrections are summed before this function is called.

    Parameters
    ----------
    mjd_ints : array-like
        Integer parts of UTC MJDs
    mjd_fracs : array-like
        Fractional parts of UTC MJDs
    obs_chain : dict
        Merged observatory-to-UTC clock chain ``{'mjd': array, 'offset': array}``.
        This is the sum of all clock corrections from ``UTC(obs)`` to ``UTC``
        as determined by the graph-based chain finder.
    bipm_clock : dict
        BIPM clock data (``TAI → TT``): ``{'mjd': array, 'offset': array}``.
        The 32.184 s TAI–TT offset is subtracted from the interpolated value
        (convention: file stores offsets relative to TAI including the 32.184 s
        constant, so we remove it since astropy handles TAI→TT internally).
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
    Clock correction chain::

        UTC(obs) --[obs_chain]--> UTC --[ERFA/astropy]--> TT(BIPM) ---> TDB

    The BIPM correction (``bipm_clock``) accounts for the difference between
    TT(TAI) (i.e. TAI + 32.184 s) and TT(BIPMyear).  Astropy converts
    UTC → TAI → TT(TAI) internally; we add the BIPM correction on top.

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation
    >>> location = EarthLocation.of_site('meerkat')
    >>> tdb_mjds = compute_tdb_standalone_vectorized(
    ...     mjd_ints=[58000, 58001],
    ...     mjd_fracs=[0.5, 0.5],
    ...     obs_chain=chain_data,
    ...     bipm_clock=bipm_data,
    ...     location=location
    ... )
    """
    from jug.io.clock import interpolate_clock_vectorized

    mjd_vals = np.array(mjd_ints, dtype=np.float64) + np.array(mjd_fracs, dtype=np.float64)

    # obs_chain already encodes the full UTC(obs) → UTC path (Dijkstra-merged)
    obs_corrs  = interpolate_clock_vectorized(obs_chain, mjd_vals)
    bipm_corrs = np.interp(mjd_vals, bipm_clock['mjd'], bipm_clock['offset']) - 32.184

    total_corrs = obs_corrs + bipm_corrs

    # Add per-TOA time offsets (e.g. TIM -to flags, TEMPO TIME statements)
    if time_offsets is not None:
        total_corrs = total_corrs + np.asarray(time_offsets, dtype=np.float64)

    # Create Time objects using pulsar MJD convention for UTC.
    # Standard astropy format='mjd' with scale='utc' prorates leap seconds
    # across the day (e.g., on MJD 54831 = 2008-12-31, a leap second day,
    # astropy treats 0.293 of a day as 0.293*(86401/86400) days, introducing
    # a ~0.293 s error). Pulsar TOA MJDs always use 86400 s/day fractions.
    # We convert fraction -> H:M:S assuming 86400 s/day, then use erfa.dtf2d
    # to get proper JD values that handle leap seconds correctly (same as
    # PINT's pulsar_mjd format).

    int_arr = np.array(mjd_ints, dtype=np.float64)
    frac_arr = np.array(mjd_fracs, dtype=np.float64)

    # Add clock corrections to the fractional day
    frac_arr = frac_arr + total_corrs / SECS_PER_DAY

    # Convert MJD integer+frac to calendar date via ERFA
    y, mo, d, fd = erfa.jd2cal(erfa.DJM0 + int_arr, frac_arr)

    # Convert fractional day to H:M:S using 86400 s/day (pulsar convention)
    fd_sec = fd * SECS_PER_DAY
    h = np.floor(fd_sec / 3600.0).astype(int)
    fd_sec -= h * 3600.0
    m = np.floor(fd_sec / 60.0).astype(int)
    s = fd_sec - m * 60.0

    # Use erfa.dtf2d to create JD values that properly handle leap seconds
    jd1, jd2 = erfa.dtf2d("UTC", y, mo, d, h, m, s)

    time_utc = Time(val=jd1, val2=jd2, format='jd', scale='utc',
                    location=location, precision=9)

    # Convert to TDB (vectorized)
    # Return TDB with full precision using double-double representation
    tdb_time = time_utc.tdb

    # Extract as longdouble: (jd1 - MJD_offset) + jd2
    MJD_OFFSET = 2400000.5
    tdb_mjd = np.array(tdb_time.jd1 - MJD_OFFSET, dtype=np.longdouble) + \
              np.array(tdb_time.jd2, dtype=np.longdouble)

    return tdb_mjd


def write_tim_file(toas: List[SimpleTOA], path: Path | str) -> None:
    """Write a list of SimpleTOA objects to a Tempo2-format .tim file.

    Uses the adjusted ``mjd_int`` / ``mjd_frac`` values (which include any
    TIME directive offsets applied during parsing) to reconstruct the MJD
    string with full precision, rather than the raw ``mjd_str`` which may
    not include those offsets.

    Parameters
    ----------
    toas : list of SimpleTOA
        TOAs to write.
    path : Path or str
        Output file path.
    """
    path = Path(path)
    with open(path, 'w') as f:
        f.write("FORMAT 1\n")
        for toa in toas:
            # Reconstruct MJD from int+frac (TIME-adjusted) with full precision
            frac_ld = np.longdouble(toa.mjd_frac)
            mjd_str = f"{toa.mjd_int}.{format(frac_ld, '.19f').split('.')[1]}"
            # Build flags string
            flags_str = ""
            if toa.flags:
                flags_str = " " + " ".join(
                    f"-{k} {v}" for k, v in toa.flags.items()
                )
            # FORMAT 1: filename freq mjd error site [flags...]
            f.write(
                f"  jug {toa.freq_mhz:.6f} "
                f"{mjd_str} {toa.error_us:.4f} {toa.observatory}{flags_str}\n"
            )
