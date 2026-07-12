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
from astropy.time.formats import TimeFormat
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

            # Apply cumulative TIME offset (seconds -> fractional day).
            # Done in longdouble: a plain float64 frac addition rounds at
            # ~5 ns, and the synced mjd_str must stay exact for TDB parity.
            mjd_frac_ld = np.longdouble(mjd_frac)
            if _state['time_offset'] != 0.0:
                mjd_frac_ld = mjd_frac_ld + (
                    np.longdouble(_state['time_offset']) / np.longdouble(86400.0)
                )
                # Normalize: handle overflow/underflow of fractional day
                if mjd_frac_ld >= 1.0:
                    shift = int(mjd_frac_ld)
                    mjd_int += shift
                    mjd_frac_ld -= np.longdouble(shift)
                elif mjd_frac_ld < 0.0:
                    shift = int(-mjd_frac_ld) + 1
                    mjd_int -= shift
                    mjd_frac_ld += np.longdouble(shift)
                mjd_frac = float(mjd_frac_ld)

            # Parse optional flags (format: -flag value)
            # Duplicate flag names (e.g. -j MEDUSA_58925 -j MEDUSA_59200) are
            # stored as lists so JUMP matching can check all values.
            flags = {}
            mjd_modified = _state['time_offset'] != 0.0
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
                    mjd_frac_ld = mjd_frac_ld + (
                        np.longdouble(addsat_sec) / np.longdouble(86400.0)
                    )
                    if mjd_frac_ld >= 1.0:
                        shift = int(mjd_frac_ld)
                        mjd_int += shift
                        mjd_frac_ld -= np.longdouble(shift)
                    elif mjd_frac_ld < 0.0:
                        shift = int(-mjd_frac_ld) + 1
                        mjd_int -= shift
                        mjd_frac_ld += np.longdouble(shift)
                    mjd_frac = float(mjd_frac_ld)
                    mjd_modified = True
                except (ValueError, TypeError):
                    pass

            if mjd_modified:
                mjd_str = _sync_toa_mjd_str(mjd_int, mjd_frac_ld)

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



def _two_sum(a, b):
    """Error-free transform of a + b for float64 arrays/scalars."""
    x = a + b
    eb = x - a
    ea = x - eb
    return x, (a - ea) + (b - eb)


def _split(a):
    c = 134217729.0 * a  # 2**27 + 1, for IEEE double splitting
    abig = c - a
    ahi = c - abig
    alo = a - ahi
    return ahi, alo


def _two_product(a, b):
    """Error-free transform of a * b for float64 arrays/scalars."""
    x = a * b
    ahi, alo = _split(a)
    bhi, blo = _split(b)
    y = alo * blo - (((x - ahi * bhi) - alo * bhi) - ahi * blo)
    return x, y


def _day_frac(val1, val2, factor=None, divisor=None):
    """Return val1 + val2 as two float64 parts without losing low bits."""
    sum12, err12 = _two_sum(val1, val2)

    if factor is not None:
        sum12, carry = _two_product(sum12, factor)
        carry += err12 * factor
        sum12, err12 = _two_sum(sum12, carry)

    if divisor is not None:
        q1 = sum12 / divisor
        p1, p2 = _two_product(q1, divisor)
        d1, d2 = _two_sum(sum12, -p1)
        d2 += err12
        d2 -= p2
        q2 = (d1 + d2) / divisor
        sum12, err12 = _two_sum(q1, q2)

    day = np.round(sum12)
    extra, frac = _two_sum(sum12, -day)
    frac += extra + err12

    excess = np.round(frac)
    day += excess
    extra, frac = _two_sum(sum12, -day)
    frac += extra + err12
    return day, frac


def _mjd_strings_to_split(mjd_strings):
    """Parse TIM MJD strings to compensated day/fraction float64 parts."""
    mjd_strings = list(mjd_strings)
    imjd = np.empty(len(mjd_strings), dtype=np.float64)
    fmjd = np.empty(len(mjd_strings), dtype=np.float64)
    for i, mjd_str in enumerate(mjd_strings):
        ss = str(mjd_str).lower().strip()
        if 'e' in ss or 'd' in ss:
            ss = ss.replace('d', 'e')
            mjd_ld = np.longdouble(ss)
            whole = np.floor(mjd_ld)
            imjd[i] = float(whole)
            fmjd[i] = float(mjd_ld - whole)
            continue
        parts = ss.split('.')
        if len(parts) == 1:
            parts.append('0')
        int_part, frac_part = parts
        imjd[i] = float(int(int_part))
        fmjd[i] = float('0.' + frac_part)
        if ss.startswith('-'):
            fmjd[i] = -fmjd[i]
    return _day_frac(imjd, fmjd)



class _JUGPulsarMJD(TimeFormat):
    """Astropy Time format for pulsar UTC MJDs with 86400-second days."""

    name = "jug_pulsar_mjd"

    def set_jds(self, val1, val2):
        if self.scale == "utc":
            self.jd1, self.jd2 = _mjds_to_jds_pulsar(val1, val2)
        else:
            self.jd1, self.jd2 = _day_frac(val1 + erfa.DJM0, val2)

    @property
    def value(self):
        return _time_to_mjd_long(self)


def _mjds_to_jds_pulsar(mjd1, mjd2):
    """Convert pulsar UTC MJD split to JD split using 86400-second days."""
    v1, v2 = _day_frac(mjd1, mjd2)
    y, mo, d, f = erfa.jd2cal(erfa.DJM0 + v1, v2)

    f *= 24.0
    h = np.floor(f).astype(int)
    f -= h
    f *= 60.0
    m = np.floor(f).astype(int)
    f -= m
    s = f * 60.0
    return erfa.dtf2d("UTC", y, mo, d, h, m, s)


def _time_to_mjd_long(time_obj):
    """Extract Time MJD using compensated JD split, without PINT helpers."""
    mjd1, mjd2 = _day_frac(time_obj.jd1 - erfa.DJM0, time_obj.jd2)
    return np.asarray(mjd1, dtype=np.longdouble) + np.asarray(mjd2, dtype=np.longdouble)


def _sync_toa_mjd_str(mjd_int: int, mjd_frac) -> str:
    """Format flag-adjusted ``(mjd_int, mjd_frac)`` for TDB/TT construction.

    TIM ``mjd_str`` is the on-disk value before ``TIME`` / ``-addsat`` etc.
    Once those flags modify ``mjd_int``/``mjd_frac`` (readTimfile.C parity),
    the stored string must match or ``compute_tdb_standalone_vectorized`` will
    build UTC Time from the unshifted MJD while clocks use the shifted SAT.

    The integer and fractional parts are formatted separately: collapsing
    them into one float64 (as done previously) rounds at the MJD-scale ULP
    (~0.6 µs at MJD 52000), far too coarse for tempo2 parity.
    """
    frac = np.longdouble(mjd_frac)
    frac_repr = np.format_float_positional(
        frac, precision=20, unique=False, trim="k"
    )
    digits = frac_repr.split(".", 1)[1] if "." in frac_repr else "0"
    return f"{int(mjd_int)}.{digits}"


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
    mjd_strings: list[str] | np.ndarray | None = None,
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
    mjd_strings : list[str] or np.ndarray or None, optional
        Original TIM MJD strings. When available, these are used for UTC Time
        construction to avoid losing one longdouble MJD ULP through float64
        fractional-day reconstruction.

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

    if mjd_strings is not None:
        int_arr, frac_arr = _mjd_strings_to_split(mjd_strings)
    else:
        int_arr = np.array(mjd_ints, dtype=np.float64)
        frac_arr = np.array(mjd_fracs, dtype=np.float64)

    # Build the raw pulsar UTC time first, then add clock corrections as a
    # TimeDelta. Folding sub-second clock terms into an MJD fraction can lose
    # one longdouble MJD ULP near modern epochs. Using a TimeFormat keeps
    # astropy's internal split consistent through UTC->TDB conversion.
    time_utc = Time(val=int_arr, val2=frac_arr, format='jug_pulsar_mjd',
                    scale='utc', location=location, precision=9)
    time_utc = time_utc + TimeDelta(total_corrs, format='sec')

    # Convert to TDB (vectorized)
    # Return TDB with full precision using double-double representation
    tdb_time = time_utc.tdb

    return _time_to_mjd_long(tdb_time)


def compute_tt_correction_sec_vectorized(
    mjd_ints,
    mjd_fracs,
    obs_chain,
    bipm_clock,
    location: EarthLocation,
    time_offsets: np.ndarray | None = None,
    mjd_strings: list[str] | np.ndarray | None = None,
    clock_eval_offset_sec: np.ndarray | None = None,
) -> np.ndarray:
    """Tempo2 ``getCorrectionTT``: (TT − sat) in seconds per TOA.

    Uses the same UTC(obs)→TT clock chain as :func:`compute_tdb_standalone_vectorized`
    but stops at TT scale (no TDB/IFTE leap).  Matches tempo2 ``formBats.C`` slot.

    When ``clock_eval_offset_sec`` is supplied, the BIPM table is evaluated at
    ``sat + offset/SECDAY`` per ``clkcorr.C`` feedback. The observatory chain
    is always evaluated at raw SAT: tempo2 shifts each hop only by the
    corrections accumulated BEFORE it, which for the UTC(obs)->UTC hops is
    the ~µs-scale site correction itself (a femtosecond-level epoch effect).
    Shifting the obs chain by the full TT-UTC (~66 s) instead samples noisy
    maser segments ~66 s off-epoch — measured up to ~7 ns error on EFF/JBO
    TOAs where the site clock wanders at µs/day.
    """
    from jug.io.clock import interpolate_clock_vectorized

    mjd_vals = np.array(mjd_ints, dtype=np.float64) + np.array(mjd_fracs, dtype=np.float64)
    eval_mjd = mjd_vals
    if clock_eval_offset_sec is not None:
        eval_mjd = mjd_vals + np.asarray(clock_eval_offset_sec, dtype=np.float64) / SECS_PER_DAY
    obs_corrs = interpolate_clock_vectorized(obs_chain, mjd_vals)
    bipm_corrs = np.interp(eval_mjd, bipm_clock["mjd"], bipm_clock["offset"]) - 32.184
    total_corrs = obs_corrs + bipm_corrs
    if time_offsets is not None:
        total_corrs = total_corrs + np.asarray(time_offsets, dtype=np.float64)

    if mjd_strings is not None:
        int_arr, frac_arr = _mjd_strings_to_split(mjd_strings)
    else:
        int_arr = np.array(mjd_ints, dtype=np.float64)
        frac_arr = np.array(mjd_fracs, dtype=np.float64)

    time_utc = Time(
        val=int_arr,
        val2=frac_arr,
        format="jug_pulsar_mjd",
        scale="utc",
        location=location,
        precision=9,
    )
    time_utc = time_utc + TimeDelta(total_corrs, format="sec")
    tt_mjd = _time_to_mjd_long(time_utc.tt)
    # Subtract in longdouble against the same SAT the Time object was built
    # from.  Downcasting tt_mjd to float64 first quantises the correction at
    # the MJD ULP (~0.6 µs near MJD 52000); the clkcorr.C feedback delta is a
    # difference of two of these corrections, so a rounding-boundary crossing
    # would inject a full ULP (~629 ns) into the emission time.
    sat_ref = (
        np.asarray(int_arr, dtype=np.longdouble)
        + np.asarray(frac_arr, dtype=np.longdouble)
    )
    return np.asarray((tt_mjd - sat_ref) * SECS_PER_DAY, dtype=np.float64)


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
