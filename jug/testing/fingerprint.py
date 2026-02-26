"""Configuration fingerprinting for Tempo2 <-> JUG parity checks.

Extracts and compares timing-model configuration from par files to ensure
both codes run with compatible settings before comparing residuals.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple


# Keys that affect residual computation and MUST match
CRITICAL_KEYS = [
    'EPHEM',            # Solar system ephemeris (DE440, DE436, ...)
    'CLK',              # Clock standard (TT(BIPM2024), TT(TAI), ...)
    'UNITS',            # Timescale (TDB required)
    'TIMEEPH',          # Time ephemeris (FB90, IF99)
    'T2CMETHOD',        # Precession/nutation model
    'BINARY',           # Binary model (ELL1, ELL1H, DD, DDK, T2, ...)
    'PLANET_SHAPIRO',   # Planet Shapiro delays (Y/N)
    'CORRECT_TROPOSPHERE',  # Tropospheric correction (Y/N)
    'DILATEFREQ',       # Doppler-corrected frequencies (Y/N)
    'NITS',             # Number of iterations (0 = evaluate only)
    'MODE',             # Timing mode (1 = TDB)
]

# Keys that affect WRMS comparison
WRMS_KEYS = [
    'NTOA',
    'TRES',
    'CHI2R',
]

# Keys that affect fitting DOF
FIT_KEYS = [
    'TNsubtractPoly',
    'TNglobalEF',
    'TNglobalEQ',
]

# Expected values for JUG compatibility
JUG_REQUIREMENTS = {
    'UNITS': 'TDB',
    'EPHEM': 'DE440',
    'CLK': 'TT(BIPM2024)',
}


def extract_fingerprint(par_path: Path) -> Dict[str, Any]:
    """Extract configuration fingerprint from a par file.

    Reads all configuration keywords, fitted parameter list, and
    metadata needed to verify parity between Tempo2 and JUG.

    Parameters
    ----------
    par_path : Path
        Path to the .par file.

    Returns
    -------
    dict
        Configuration fingerprint with keys:
        - 'config': dict of CRITICAL_KEYS values
        - 'wrms': dict of WRMS_KEYS values
        - 'fit': dict of FIT_KEYS values
        - 'fitted_params': list of parameter names with fit flag = 1
        - 'n_fitted': number of fitted parameters
        - 'par_file': str path to par file
    """
    config = {}
    wrms = {}
    fit = {}
    fitted_params = []

    all_keys = set(CRITICAL_KEYS + WRMS_KEYS + FIT_KEYS)

    with open(par_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            key = parts[0]

            # Extract config values
            if key in all_keys and len(parts) >= 2:
                val = parts[1]
                if key in CRITICAL_KEYS:
                    config[key] = val
                elif key in WRMS_KEYS:
                    wrms[key] = val
                elif key in FIT_KEYS:
                    fit[key] = val

            # Special case: CHI2R has two values "CHI2R value ndof"
            if key == 'CHI2R' and len(parts) >= 3:
                wrms['CHI2R'] = parts[1]
                wrms['CHI2R_NDOF'] = parts[2]

            # Detect fitted parameters (fit flag = 1)
            if len(parts) >= 3:
                try:
                    flag = int(parts[2])
                    if flag == 1:
                        fitted_params.append(key)
                except ValueError:
                    pass

            # MODE is bare "MODE 1"
            if key == 'MODE' and len(parts) >= 2:
                config['MODE'] = parts[1]

    return {
        'config': config,
        'wrms': wrms,
        'fit': fit,
        'fitted_params': sorted(fitted_params),
        'n_fitted': len(fitted_params),
        'par_file': str(par_path),
    }


def compare_fingerprints(
    fp1: Dict[str, Any],
    fp2: Dict[str, Any],
    label1: str = "File 1",
    label2: str = "File 2",
) -> Tuple[bool, List[str]]:
    """Compare two fingerprints and report discrepancies.

    Parameters
    ----------
    fp1, fp2 : dict
        Fingerprints from extract_fingerprint().
    label1, label2 : str
        Labels for reporting.

    Returns
    -------
    ok : bool
        True if all critical keys match.
    issues : list of str
        Human-readable list of discrepancies.
    """
    issues = []

    # Compare critical config keys
    for key in CRITICAL_KEYS:
        v1 = fp1['config'].get(key)
        v2 = fp2['config'].get(key)
        if v1 != v2:
            issues.append(
                f"Config mismatch: {key} = {v1!r} ({label1}) vs {v2!r} ({label2})"
            )

    # Compare fitted parameter lists
    s1 = set(fp1['fitted_params'])
    s2 = set(fp2['fitted_params'])
    if s1 != s2:
        only1 = s1 - s2
        only2 = s2 - s1
        if only1:
            issues.append(f"Params only in {label1}: {sorted(only1)}")
        if only2:
            issues.append(f"Params only in {label2}: {sorted(only2)}")

    ok = len(issues) == 0
    return ok, issues


def validate_jug_compatible(fp: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check that a par file is compatible with JUG's assumptions.

    JUG requires UNITS=TDB, EPHEM=DE440, CLK=TT(BIPM2024).

    Parameters
    ----------
    fp : dict
        Fingerprint from extract_fingerprint().

    Returns
    -------
    ok : bool
        True if par file is JUG-compatible.
    issues : list of str
        Human-readable list of incompatibilities.
    """
    issues = []
    for key, expected in JUG_REQUIREMENTS.items():
        actual = fp['config'].get(key)
        if actual is None:
            issues.append(f"Missing required key: {key} (expected {expected})")
        elif actual != expected:
            issues.append(
                f"Incompatible {key}: {actual!r} (JUG requires {expected!r})"
            )

    # UNITS must be TDB
    units = fp['config'].get('UNITS')
    if units and units != 'TDB':
        issues.append(f"UNITS={units}: JUG only supports TDB")

    return len(issues) == 0, issues


def fingerprint_report(par_path: Path) -> str:
    """Generate a human-readable fingerprint report.

    Parameters
    ----------
    par_path : Path
        Path to the .par file.

    Returns
    -------
    str
        Multi-line report string.
    """
    fp = extract_fingerprint(par_path)
    lines = [f"Configuration Fingerprint: {par_path.name}"]
    lines.append("=" * 60)

    lines.append("\nCritical Settings:")
    for key in CRITICAL_KEYS:
        val = fp['config'].get(key, '(not set)')
        lines.append(f"  {key:25s} = {val}")

    lines.append(f"\nFitted Parameters ({fp['n_fitted']}):")
    for p in fp['fitted_params']:
        lines.append(f"  {p}")

    lines.append("\nWRMS Metadata:")
    for key in WRMS_KEYS + ['CHI2R_NDOF']:
        val = fp['wrms'].get(key, '(not set)')
        lines.append(f"  {key:25s} = {val}")

    lines.append("\nFit Options:")
    for key in FIT_KEYS:
        val = fp['fit'].get(key, '(not set)')
        lines.append(f"  {key:25s} = {val}")

    ok, issues = validate_jug_compatible(fp)
    lines.append(f"\nJUG Compatibility: {'[x] OK' if ok else '[ ] FAIL'}")
    for issue in issues:
        lines.append(f"  [ ] {issue}")

    return '\n'.join(lines)
