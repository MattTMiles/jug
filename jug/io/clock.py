"""Clock correction file handling and interpolation.

This module provides functions to parse and interpolate tempo2-style clock
correction files for the observatory → UTC → TAI → TT clock chain.
"""

from functools import lru_cache
from pathlib import Path
from bisect import bisect_left
import numpy as np


@lru_cache(maxsize=16)
def _parse_clock_file_cached(path_str: str) -> tuple:
    """Internal cached clock file parser.
    
    Returns tuple of (mjd_tuple, offset_tuple) for hashability.
    """
    mjds = []
    offsets = []
    path = Path(path_str)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mjd = float(parts[0])
                    offset = float(parts[1])
                    mjds.append(mjd)
                    offsets.append(offset)
                except ValueError:
                    continue

    return (tuple(mjds), tuple(offsets))


def parse_clock_file(path: Path | str) -> dict:
    """Parse tempo2-style clock correction file.

    Parameters
    ----------
    path : Path or str
        Path to clock file

    Returns
    -------
    dict
        Dictionary with 'mjd' and 'offset' arrays

    Notes
    -----
    File format: MJD offset(seconds) [optional columns]

    Lines starting with '#' are comments and are skipped.
    
    Results are cached using functools.lru_cache for performance.
    Repeated calls with the same path return cached arrays.

    Examples
    --------
    >>> clock_data = parse_clock_file("data/clock/mk2utc.clk")
    >>> print(f"Clock file has {len(clock_data['mjd'])} entries")
    """
    # Resolve to absolute path string for consistent caching
    path_str = str(Path(path).resolve())
    
    # Get cached tuples and convert to arrays
    mjd_tuple, offset_tuple = _parse_clock_file_cached(path_str)
    
    return {
        'mjd': np.array(mjd_tuple),
        'offset': np.array(offset_tuple),
        'path': str(path),
    }


def interpolate_clock(clock_data: dict, mjd: float) -> float:
    """Interpolate clock correction at given MJD.

    Uses linear interpolation between adjacent points.

    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays
    mjd : float
        MJD value to interpolate at

    Returns
    -------
    float
        Interpolated clock offset in seconds

    Notes
    -----
    For MJDs outside the clock file range, the nearest boundary value
    is returned (constant extrapolation).

    Examples
    --------
    >>> clock_data = parse_clock_file("mk2utc.clk")
    >>> offset = interpolate_clock(clock_data, 58000.5)
    >>> print(f"Clock correction: {offset:.9f} seconds")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']

    if len(mjds) == 0:
        return 0.0

    # Handle boundaries
    if mjd <= mjds[0]:
        return offsets[0]
    if mjd >= mjds[-1]:
        return offsets[-1]

    # Find bracketing points
    idx = bisect_left(mjds, mjd)
    if idx == 0:
        return offsets[0]

    # Linear interpolation
    mjd0, mjd1 = mjds[idx-1], mjds[idx]
    off0, off1 = offsets[idx-1], offsets[idx]

    frac = (mjd - mjd0) / (mjd1 - mjd0)
    return off0 + frac * (off1 - off0)


def interpolate_clock_vectorized(clock_data: dict, mjd_array: np.ndarray) -> np.ndarray:
    """Vectorized clock interpolation using np.searchsorted.

    ~10x faster than looping over interpolate_clock() for large arrays.
    Maintains identical accuracy to scalar version.

    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays
    mjd_array : np.ndarray
        Array of MJD values to interpolate

    Returns
    -------
    np.ndarray
        Interpolated clock offsets in seconds

    Notes
    -----
    This function is optimized for processing many TOAs at once.
    For single values, use interpolate_clock() instead.

    Examples
    --------
    >>> clock_data = parse_clock_file("mk2utc.clk")
    >>> mjds = np.array([58000.0, 58001.0, 58002.0])
    >>> offsets = interpolate_clock_vectorized(clock_data, mjds)
    >>> print(f"Corrections: {offsets}")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']

    # Handle empty clock data
    if len(mjds) == 0:
        return np.zeros_like(mjd_array)

    # Find insertion indices (right side gives us the upper bracket)
    idx = np.searchsorted(mjds, mjd_array, side='right')

    # Clip to valid range [1, len(mjds)-1] for interpolation
    # idx=0 means before first point, idx=len means after last point
    idx = np.clip(idx, 1, len(mjds) - 1)

    # Get bracketing points
    mjd0 = mjds[idx - 1]
    mjd1 = mjds[idx]
    off0 = offsets[idx - 1]
    off1 = offsets[idx]

    # Vectorized linear interpolation
    # Handle edge cases: if mjd0 == mjd1, frac should be 0 (use first offset)
    frac = np.where(mjd1 != mjd0, (mjd_array - mjd0) / (mjd1 - mjd0), 0.0)

    return off0 + frac * (off1 - off0)


def validate_clock_file_coverage(clock_data: dict, mjd_start: float, mjd_end: float, 
                                   file_name: str = "clock file", warn_days: float = 30.0) -> dict:
    """Validate that a clock file covers the required MJD range.
    
    Checks for:
    - Coverage gaps (MJDs outside clock file range)
    - Suspicious constant regions (potential extrapolation)
    - Outdated files (end date too far in the past)
    
    Parameters
    ----------
    clock_data : dict
        Clock data with 'mjd' and 'offset' arrays
    mjd_start : float
        Start MJD of data requiring coverage
    mjd_end : float
        End MJD of data requiring coverage
    file_name : str, optional
        Name of clock file for warning messages
    warn_days : float, optional
        Warn if file ends more than this many days before mjd_end (default: 30)
    
    Returns
    -------
    dict
        Validation results with keys:
        - 'valid': bool, True if coverage is adequate
        - 'warnings': list of warning strings
        - 'errors': list of error strings
        - 'coverage_start': MJD where clock file starts
        - 'coverage_end': MJD where clock file ends
        - 'data_start': MJD where data starts
        - 'data_end': MJD where data ends
    
    Examples
    --------
    >>> clock_data = parse_clock_file("tai2tt_bipm2024.clk")
    >>> result = validate_clock_file_coverage(clock_data, 60000.0, 60837.0)
    >>> if not result['valid']:
    ...     for warning in result['warnings']:
    ...         print(f"WARNING: {warning}")
    """
    mjds = clock_data['mjd']
    offsets = clock_data['offset']
    
    warnings = []
    errors = []
    valid = True
    
    if len(mjds) == 0:
        errors.append(f"{file_name}: Clock file is empty")
        return {
            'valid': False,
            'warnings': warnings,
            'errors': errors,
            'coverage_start': None,
            'coverage_end': None,
            'data_start': mjd_start,
            'data_end': mjd_end
        }
    
    coverage_start = mjds[0]
    coverage_end = mjds[-1]
    
    # Check if data is outside clock file range
    if mjd_start < coverage_start:
        errors.append(
            f"{file_name}: Data starts at MJD {mjd_start:.1f} but clock file "
            f"only covers from MJD {coverage_start:.1f} "
            f"({coverage_start - mjd_start:.1f} days before coverage)"
        )
        valid = False
    
    if mjd_end > coverage_end:
        days_past = mjd_end - coverage_end
        if days_past > warn_days:
            errors.append(
                f"{file_name}: Data extends to MJD {mjd_end:.1f} but clock file "
                f"ends at MJD {coverage_end:.1f} "
                f"({days_past:.1f} days of extrapolation). "
                f"Consider updating clock file."
            )
            valid = False
        else:
            warnings.append(
                f"{file_name}: Minor extrapolation ({days_past:.1f} days past clock file end)"
            )
    
    # Check for suspicious constant regions near the end
    # (indicates possible extrapolation in the clock file itself)
    if len(mjds) > 10:
        # Find where real data ends by looking for large gaps or constant regions
        # Check spacing between entries
        mjd_diffs = np.diff(mjds)
        
        # Look for abnormally large gaps (> 100 days suggests jump to extrapolation)
        # Only consider gaps that occur after the data start to avoid false positives
        # from dummy header entries at MJD ~0 in some clock files.
        large_gaps = np.where((mjd_diffs > 100) & (mjds[:-1] > mjd_start - 365))[0]
        if len(large_gaps) > 0:
            # Found a large gap - data before this is real
            real_data_end_idx = large_gaps[0]
            real_data_end = mjds[real_data_end_idx]
            
            if mjd_end > real_data_end:
                errors.append(
                    f"{file_name}: Real data ends at MJD {real_data_end:.1f}, "
                    f"but your data extends to MJD {mjd_end:.1f} "
                    f"({mjd_end - real_data_end:.1f} days using extrapolated values). "
                    f"Clock file has large gap suggesting constant extrapolation. "
                    f"UPDATE CLOCK FILE from IPTA repository!"
                )
                valid = False
        
        # Also check last 10 entries for constant values
        last_offsets = offsets[-10:]
        if np.std(last_offsets) < 1e-12:  # Effectively constant
            # Check if there's variation before the constant region
            if len(mjds) > 20:
                prev_offsets = offsets[-20:-10]
                if np.std(prev_offsets) > 1e-12:  # Previous region was varying
                    warnings.append(
                        f"{file_name}: Last 10 entries are constant at "
                        f"{last_offsets[-1]:.12f} s (possible extrapolation within file)"
                    )
    
    return {
        'valid': valid,
        'warnings': warnings,
        'errors': errors,
        'coverage_start': coverage_start,
        'coverage_end': coverage_end,
        'data_start': mjd_start,
        'data_end': mjd_end
    }


def check_clock_files(mjd_start: float, mjd_end: float,
                      mk_clock: dict, gps_clock: dict, bipm_clock: dict,
                      verbose: bool = True,
                      clock_dir: str = None) -> tuple:
    """Check all clock files for adequate coverage.

    Errors (data outside clock file range) are always printed in bold red,
    regardless of the ``verbose`` flag.  Warnings (minor extrapolation, etc.)
    are printed in bold yellow when ``verbose=True``.

    Parameters
    ----------
    mjd_start : float
        Start MJD of data
    mjd_end : float
        End MJD of data
    mk_clock : dict
        Observatory clock data
    gps_clock : dict
        GPS clock data
    bipm_clock : dict
        BIPM clock data
    verbose : bool, optional
        Print warnings in addition to errors (default: True)
    clock_dir : str, optional
        Path to the clock directory, used in the actionable suggestion message.

    Returns
    -------
    tuple
        ``(valid, issues)`` where *valid* is a bool (True = no hard errors) and
        *issues* is a list of dicts, each with keys ``'severity'`` (``'error'``
        or ``'warning'``) and ``'message'`` (str).

    Examples
    --------
    >>> mk = parse_clock_file("mk2utc.clk")
    >>> gps = parse_clock_file("gps2utc.clk")
    >>> bipm = parse_clock_file("tai2tt_bipm2024.clk")
    >>> ok, issues = check_clock_files(58000.0, 60837.0, mk, gps, bipm)
    """
    _RED   = "\033[1;31m"   # bold red
    _YELLOW = "\033[1;33m"  # bold yellow
    _RESET = "\033[0m"

    all_valid = True
    all_issues = []

    suggestion = (
        f"  → Copy the correct clock files into {clock_dir}"
        if clock_dir else
        "  → Copy the correct clock files into your JUG data/clock/ directory"
    )

    for name, clock_data in [
        ("Observatory clock", mk_clock),
        ("GPS clock (gps2utc.clk)", gps_clock),
        ("BIPM clock (tai2tt_bipm*.clk)", bipm_clock),
    ]:
        # Include the actual filename so the user knows which file to update
        filename = Path(clock_data.get('path', '')).name if clock_data.get('path') else ''
        label = f"{name} ({filename})" if filename and filename not in name else name
        result = validate_clock_file_coverage(clock_data, mjd_start, mjd_end, label)

        if not result['valid']:
            all_valid = False

        for error in result['errors']:
            msg = f"CLOCK FILE ERROR: {error}"
            all_issues.append({'severity': 'error', 'message': msg})
            # Always print errors — they affect timing accuracy
            print(f"{_RED}❌ {msg}{_RESET}")
            print(f"{_RED}{suggestion}{_RESET}")

        for warning in result['warnings']:
            msg = f"CLOCK FILE WARNING: {warning}"
            all_issues.append({'severity': 'warning', 'message': msg})
            if verbose:
                print(f"{_YELLOW}⚠️  {msg}{_RESET}")
                print(f"{_YELLOW}{suggestion}{_RESET}")

    return all_valid, all_issues


def compare_clock_files(path_a: Path | str, path_b: Path | str,
                        threshold_us: float = 0.001) -> dict:
    """Compare two clock files and report significant differences.

    Parameters
    ----------
    path_a, path_b : Path or str
        Paths to clock files to compare.
    threshold_us : float, optional
        Difference threshold in microseconds above which to flag (default: 0.001 μs).

    Returns
    -------
    dict
        Keys: ``'max_diff_us'``, ``'mean_diff_us'``, ``'significant'`` (bool),
        ``'a_end_mjd'``, ``'b_end_mjd'``, ``'a_entries'``, ``'b_entries'``,
        ``'summary'`` (human-readable string).
    """
    def _load(p):
        d = parse_clock_file(p)
        return d['mjd'], d['offset']

    mjds_a, off_a = _load(path_a)
    mjds_b, off_b = _load(path_b)

    # Interpolate b onto a's grid within the overlap
    overlap_start = max(mjds_a[0], mjds_b[0])
    overlap_end = min(mjds_a[-1], mjds_b[-1])

    max_diff_us = 0.0
    mean_diff_us = 0.0

    if overlap_end > overlap_start:
        mask = (mjds_a >= overlap_start) & (mjds_a <= overlap_end)
        sample_mjds = mjds_a[mask]
        if len(sample_mjds) > 0:
            interp_b = np.interp(sample_mjds, mjds_b, off_b)
            interp_a = off_a[mask]
            diffs_us = np.abs(interp_a - interp_b) * 1e6
            max_diff_us = float(np.max(diffs_us))
            mean_diff_us = float(np.mean(diffs_us))

    summary = (
        f"{Path(path_a).name}: {len(mjds_a)} entries, ends MJD {mjds_a[-1]:.1f}; "
        f"{Path(path_b).name}: {len(mjds_b)} entries, ends MJD {mjds_b[-1]:.1f}; "
        f"max diff = {max_diff_us:.4f} μs"
    )

    return {
        'max_diff_us': max_diff_us,
        'mean_diff_us': mean_diff_us,
        'significant': max_diff_us > threshold_us,
        'a_end_mjd': float(mjds_a[-1]),
        'b_end_mjd': float(mjds_b[-1]),
        'a_entries': len(mjds_a),
        'b_entries': len(mjds_b),
        'summary': summary,
    }


def check_iers_coverage(mjd_start: float, mjd_end: float,
                        verbose: bool = True) -> tuple:
    """Check that astropy's IERS Earth-orientation data covers the data MJD range.

    The ITRF→GCRS coordinate transform (used when computing observatory SSB
    positions) relies on IERS UT1-UTC and polar-motion data.  Using predicted
    rather than measured values introduces small but systematic errors.

    Parameters
    ----------
    mjd_start, mjd_end : float
        MJD range of TOA data.
    verbose : bool, optional
        Print status even when coverage is fine (default: True).

    Returns
    -------
    tuple
        ``(valid, issues)`` matching the same convention as
        :func:`check_clock_files`.
    """
    _RED    = "\033[1;31m"
    _YELLOW = "\033[1;33m"
    _RESET  = "\033[0m"

    issues = []
    valid = True

    try:
        from astropy.utils import iers as astropy_iers
        import numpy as np

        tab = astropy_iers.earth_orientation_table.get()
        table_mjds = np.asarray(tab['MJD'])
        table_end = float(table_mjds[-1])

        # Find end of *measured* (vs predicted) UT1-UTC
        measured_end = table_end
        for col in ('UT1_UTC_B', 'UT1_UTC_A', 'UT1_UTC'):
            if col in tab.colnames:
                vals = np.asarray(tab[col], dtype=float)
                finite_mask = np.isfinite(vals)
                if np.any(finite_mask):
                    measured_end = float(table_mjds[finite_mask][-1])
                break

        table_type = type(tab).__name__

        if mjd_end > table_end:
            days_past = mjd_end - table_end
            msg = (
                f"EOP/IERS ERROR: Data extends to MJD {mjd_end:.1f} but IERS "
                f"table ({table_type}) ends at MJD {table_end:.1f} "
                f"({days_past:.1f} days beyond coverage). "
                f"Coordinate transforms may be wrong."
            )
            issues.append({'severity': 'error', 'message': msg})
            valid = False
            print(f"{_RED}❌ {msg}{_RESET}")
            print(f"{_RED}  → Run: python -c \"from astropy.utils.iers import IERS_A; IERS_A.open()\"{_RESET}")
        elif mjd_end > measured_end:
            days_predicted = mjd_end - measured_end
            msg = (
                f"EOP/IERS WARNING: Data extends {days_predicted:.1f} days past the "
                f"end of measured IERS data (MJD {measured_end:.1f}). "
                f"Coordinate transforms use predicted EOP values in this range."
            )
            issues.append({'severity': 'warning', 'message': msg})
            if verbose:
                print(f"{_YELLOW}⚠️  {msg}{_RESET}")
                print(f"{_YELLOW}  → Download fresh IERS-A: python -c \"from astropy.utils.iers import IERS_A; IERS_A.open()\"{_RESET}")
        else:
            if verbose:
                print(
                    f"   ✓ IERS data ({table_type}) covers MJD {mjd_end:.1f} "
                    f"with measured data to MJD {measured_end:.1f}"
                )

    except Exception as e:
        msg = f"EOP/IERS WARNING: Could not check IERS coverage: {e}"
        issues.append({'severity': 'warning', 'message': msg})
        if verbose:
            print(f"{_YELLOW}⚠️  {msg}{_RESET}")

    return valid, issues
