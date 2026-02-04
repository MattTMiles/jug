"""
Test data path utilities for JUG tests.

This module provides a consistent mechanism for resolving PAR/TIM file paths
for tests. It supports:

1. Environment variables for CI/portable testing:
   - JUG_TEST_DATA_DIR: Base directory for all test data
   - JUG_TEST_J1713_PAR, JUG_TEST_J1713_TIM: J1713+0747 dataset
   - JUG_TEST_J2241_PAR, JUG_TEST_J2241_TIM: J2241-5236 dataset
   - JUG_TEST_J1909_PAR, JUG_TEST_J1909_TIM: J1909-3744 dataset
   - JUG_TEST_J1022_PAR, JUG_TEST_J1022_TIM: J1022+1001 dataset

2. Defaults to Matt's local paths if env vars not set (for backwards compat).

3. SKIP support: Returns None if files don't exist, allowing tests to skip
   gracefully rather than fail.

Usage in tests:
    from tests.test_paths import get_j1713_paths, skip_if_missing

    PAR, TIM = get_j1713_paths()
    skip_if_missing(PAR, TIM)  # Prints SKIP message and returns False if missing
"""

import os
from pathlib import Path
from typing import Tuple, Optional

# =============================================================================
# Default paths (Matt's local setup - used when env vars not set)
# =============================================================================

_DEFAULT_MPTA_DIR = Path("/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb")

_DEFAULT_PATHS = {
    'J1713': {
        'par': _DEFAULT_MPTA_DIR / "J1713+0747_tdb.par",
        'tim': _DEFAULT_MPTA_DIR / "J1713+0747.tim",
    },
    'J2241': {
        'par': _DEFAULT_MPTA_DIR / "J2241-5236_tdb.par",
        'tim': _DEFAULT_MPTA_DIR / "J2241-5236.tim",
    },
    'J1909': {
        'par': Path("data/pulsars/J1909-3744_tdb.par"),  # Relative to repo root
        'tim': Path("data/pulsars/J1909-3744.tim"),
    },
    'J1022': {
        'par': _DEFAULT_MPTA_DIR / "J1022+1001_tdb.par",
        'tim': _DEFAULT_MPTA_DIR / "J1022+1001.tim",
    },
}


# =============================================================================
# Path resolution functions
# =============================================================================

def _get_env_path(env_var: str, default: Optional[Path] = None) -> Optional[Path]:
    """Get a path from an environment variable, with optional default."""
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return default


def _resolve_pulsar_paths(
    pulsar_key: str,
    par_env: str,
    tim_env: str
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve PAR/TIM paths for a pulsar from env vars or defaults.
    
    Returns (None, None) if files don't exist.
    """
    # Check for specific env vars first
    par = _get_env_path(par_env)
    tim = _get_env_path(tim_env)
    
    # Check for base directory env var
    base_dir = _get_env_path('JUG_TEST_DATA_DIR')
    if base_dir and not par:
        # Try common naming conventions
        par = base_dir / f"{pulsar_key}_tdb.par"
        if not par.exists():
            par = base_dir / f"{pulsar_key}.par"
    if base_dir and not tim:
        tim = base_dir / f"{pulsar_key}.tim"
    
    # Fall back to defaults
    if not par and pulsar_key in _DEFAULT_PATHS:
        par = _DEFAULT_PATHS[pulsar_key]['par']
    if not tim and pulsar_key in _DEFAULT_PATHS:
        tim = _DEFAULT_PATHS[pulsar_key]['tim']
    
    return par, tim


def get_j1713_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1713+0747 PAR/TIM paths.
    
    Returns:
        Tuple of (par_path, tim_path). May be None if not configured.
    """
    return _resolve_pulsar_paths(
        'J1713',
        'JUG_TEST_J1713_PAR',
        'JUG_TEST_J1713_TIM'
    )


def get_j2241_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J2241-5236 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        'J2241',
        'JUG_TEST_J2241_PAR',
        'JUG_TEST_J2241_TIM'
    )


def get_j1909_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1909-3744 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        'J1909',
        'JUG_TEST_J1909_PAR',
        'JUG_TEST_J1909_TIM'
    )


def get_j1022_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1022+1001 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        'J1022',
        'JUG_TEST_J1022_PAR',
        'JUG_TEST_J1022_TIM'
    )


def files_exist(par: Optional[Path], tim: Optional[Path]) -> bool:
    """Check if both PAR and TIM files exist."""
    if par is None or tim is None:
        return False
    return par.exists() and tim.exists()


def skip_if_missing(par: Optional[Path], tim: Optional[Path], test_name: str = "") -> bool:
    """Check if files exist, print SKIP message if not.
    
    Returns:
        True if files exist and test should run.
        False if files are missing and test should skip.
    """
    prefix = f"[{test_name}] " if test_name else ""
    
    if par is None:
        print(f"{prefix}SKIP: PAR path not configured (set JUG_TEST_*_PAR env var)")
        return False
    if tim is None:
        print(f"{prefix}SKIP: TIM path not configured (set JUG_TEST_*_TIM env var)")
        return False
    if not par.exists():
        print(f"{prefix}SKIP: PAR file not found: {par}")
        return False
    if not tim.exists():
        print(f"{prefix}SKIP: TIM file not found: {tim}")
        return False
    return True


def require_files(par: Optional[Path], tim: Optional[Path], test_name: str = ""):
    """Like skip_if_missing but raises RuntimeError instead of returning False.
    
    Use this for "must run" tests that should fail rather than skip.
    """
    prefix = f"[{test_name}] " if test_name else ""
    
    if par is None:
        raise RuntimeError(f"{prefix}PAR path not configured (set JUG_TEST_*_PAR env var)")
    if tim is None:
        raise RuntimeError(f"{prefix}TIM path not configured (set JUG_TEST_*_TIM env var)")
    if not par.exists():
        raise RuntimeError(f"{prefix}PAR file not found: {par}")
    if not tim.exists():
        raise RuntimeError(f"{prefix}TIM file not found: {tim}")


# =============================================================================
# Convenience function for getting all available test datasets
# =============================================================================

def get_available_datasets() -> dict:
    """Return dict of all configured test datasets that actually exist.
    
    Returns:
        Dict mapping pulsar name to {'par': Path, 'tim': Path}
    """
    datasets = {}
    
    for name, getter in [
        ('J1713', get_j1713_paths),
        ('J2241', get_j2241_paths),
        ('J1909', get_j1909_paths),
        ('J1022', get_j1022_paths),
    ]:
        par, tim = getter()
        if files_exist(par, tim):
            datasets[name] = {'par': par, 'tim': tim}
    
    return datasets


if __name__ == "__main__":
    # Quick diagnostic when run directly
    print("JUG Test Data Path Configuration")
    print("=" * 60)
    
    for name, getter in [
        ('J1713+0747', get_j1713_paths),
        ('J2241-5236', get_j2241_paths),
        ('J1909-3744', get_j1909_paths),
        ('J1022+1001', get_j1022_paths),
    ]:
        par, tim = getter()
        exists = files_exist(par, tim)
        status = "✓ AVAILABLE" if exists else "✗ NOT FOUND"
        print(f"\n{name}: {status}")
        print(f"  PAR: {par}")
        print(f"  TIM: {tim}")
    
    print("\n" + "=" * 60)
    print("To configure paths, set environment variables:")
    print("  JUG_TEST_DATA_DIR=/path/to/data  (base directory)")
    print("  JUG_TEST_J1713_PAR=/path/to/par  (per-pulsar override)")
    print("  JUG_TEST_J1713_TIM=/path/to/tim")
