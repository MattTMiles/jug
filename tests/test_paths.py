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
   - JUG_TEST_J0613_PAR, JUG_TEST_J0613_TIM: J0613-0200 dataset
   - JUG_TEST_J0125_PAR, JUG_TEST_J0125_TIM: J0125-2327 dataset
   - JUG_TEST_J0125_POSTFIT_PAR: J0125-2327 post-fit par (optional)

2. Bundled fixtures in tests/data_mpta/ (MPTA DR2 subsets).

3. Legacy Matt local paths as a last-resort fallback.

4. SKIP support: Returns None if files don't exist, allowing tests to skip
   gracefully rather than fail.

Usage in tests:
    from tests.test_paths import get_j1713_paths, skip_if_missing

    PAR, TIM = get_j1713_paths()
    skip_if_missing(PAR, TIM)  # Prints SKIP message and returns False if missing
"""

import json
import os
from pathlib import Path
from typing import Optional, Tuple

# =============================================================================
# Fixture directories
# =============================================================================

_TESTS_DIR = Path(__file__).parent
_MPTA_DIR = _TESTS_DIR / "data_mpta"
_LEGACY_MPTA_DIR = Path(
    "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb"
)
_LEGACY_MPTA_ADS_DIR = Path(
    "/home/mattm/projects/MPTA/github/mpta-6yr/data/fifth_pass/32ch_tdb_ads"
)

# Pulsar key -> bundled MPTA fixture id
_MPTA_FIXTURE_IDS = {
    "J1713": "j1713_binary",
    "J2241": "j2241_fb",
    "J1909": "j1909_t2",
    "J1022": "j1022_ell1h",
    "J0613": "j0613_ell1h",
    "J0125": "j0125_ell1h",
}

# Legacy filename defaults (Matt's fifth_pass/32ch_tdb layout)
_LEGACY_PATHS = {
    "J1713": {
        "par": _LEGACY_MPTA_DIR / "J1713+0747_tdb.par",
        "tim": _LEGACY_MPTA_DIR / "J1713+0747.tim",
    },
    "J2241": {
        "par": _LEGACY_MPTA_DIR / "J2241-5236_tdb.par",
        "tim": _LEGACY_MPTA_DIR / "J2241-5236.tim",
    },
    "J1909": {
        "par": _LEGACY_MPTA_DIR / "J1909-3744_tdb.par",
        "tim": _LEGACY_MPTA_DIR / "J1909-3744.tim",
    },
    "J1022": {
        "par": _LEGACY_MPTA_DIR / "J1022+1001_tdb.par",
        "tim": _LEGACY_MPTA_DIR / "J1022+1001.tim",
    },
    "J0613": {
        "par": _LEGACY_MPTA_DIR / "J0613-0200_tdb.par",
        "tim": _LEGACY_MPTA_DIR / "J0613-0200.tim",
    },
    "J0125": {
        "par": _LEGACY_MPTA_ADS_DIR / "J0125-2327_tdb.par",
        "tim": _LEGACY_MPTA_ADS_DIR / "J0125-2327.tim",
        "postfit_par": _LEGACY_MPTA_ADS_DIR / "J0125-2327_test.par",
    },
}


# =============================================================================
# Path resolution functions
# =============================================================================

def _load_mpta_manifest() -> list:
    manifest_file = _MPTA_DIR / "manifest.json"
    if not manifest_file.exists():
        return []
    with open(manifest_file) as f:
        return json.load(f)


def get_mpta_fixture_paths(fixture_id: str) -> Tuple[Path, Path]:
    """Get bundled MPTA fixture PAR/TIM paths by fixture id."""
    for row in _load_mpta_manifest():
        if row["id"] == fixture_id:
            return _MPTA_DIR / row["par"], _MPTA_DIR / row["tim"]
    known = ", ".join(row["id"] for row in _load_mpta_manifest())
    raise KeyError(f"Unknown MPTA fixture {fixture_id!r}. Known fixtures: {known}")


def get_mpta_postfit_par(fixture_id: str) -> Optional[Path]:
    """Return optional post-fit par path for an MPTA fixture, if declared."""
    for row in _load_mpta_manifest():
        if row["id"] == fixture_id and "postfit_par" in row:
            return _MPTA_DIR / row["postfit_par"]
    return None


def _get_env_path(env_var: str, default: Optional[Path] = None) -> Optional[Path]:
    """Get a path from an environment variable, with optional default."""
    val = os.environ.get(env_var)
    if val:
        return Path(val)
    return default


def _first_existing(*candidates: Optional[Path]) -> Optional[Path]:
    for path in candidates:
        if path is not None and path.exists():
            return path
    return candidates[0] if candidates else None


def _resolve_pulsar_paths(
    pulsar_key: str,
    par_env: str,
    tim_env: str,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve PAR/TIM paths for a pulsar from env vars or defaults."""
    par = _get_env_path(par_env)
    tim = _get_env_path(tim_env)

    base_dir = _get_env_path("JUG_TEST_DATA_DIR")
    if base_dir and not par:
        legacy = _LEGACY_PATHS.get(pulsar_key, {})
        par_name = legacy.get("par", Path(f"{pulsar_key}.par")).name
        par = base_dir / par_name
    if base_dir and not tim:
        legacy = _LEGACY_PATHS.get(pulsar_key, {})
        tim_name = legacy.get("tim", Path(f"{pulsar_key}.tim")).name
        tim = base_dir / tim_name

    fixture_id = _MPTA_FIXTURE_IDS.get(pulsar_key)
    bundled_par = bundled_tim = None
    if fixture_id:
        try:
            bundled_par, bundled_tim = get_mpta_fixture_paths(fixture_id)
        except (KeyError, FileNotFoundError):
            bundled_par = bundled_tim = None

    legacy = _LEGACY_PATHS.get(pulsar_key, {})
    par = _first_existing(par, bundled_par, legacy.get("par"))
    tim = _first_existing(tim, bundled_tim, legacy.get("tim"))

    return par, tim


def get_j1713_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1713+0747 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        "J1713",
        "JUG_TEST_J1713_PAR",
        "JUG_TEST_J1713_TIM",
    )


def get_j2241_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J2241-5236 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        "J2241",
        "JUG_TEST_J2241_PAR",
        "JUG_TEST_J2241_TIM",
    )


def get_j1909_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1909-3744 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        "J1909",
        "JUG_TEST_J1909_PAR",
        "JUG_TEST_J1909_TIM",
    )


def get_j1022_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J1022+1001 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        "J1022",
        "JUG_TEST_J1022_PAR",
        "JUG_TEST_J1022_TIM",
    )


def get_j0613_paths() -> Tuple[Optional[Path], Optional[Path]]:
    """Get J0613-0200 PAR/TIM paths."""
    return _resolve_pulsar_paths(
        "J0613",
        "JUG_TEST_J0613_PAR",
        "JUG_TEST_J0613_TIM",
    )


def get_j0125_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Get J0125-2327 pre-fit PAR, TIM, and optional post-fit PAR paths."""
    par, tim = _resolve_pulsar_paths(
        "J0125",
        "JUG_TEST_J0125_PAR",
        "JUG_TEST_J0125_TIM",
    )
    postfit_par = _get_env_path("JUG_TEST_J0125_POSTFIT_PAR")
    if postfit_par is None:
        postfit_par = get_mpta_postfit_par("j0125_ell1h")
    if postfit_par is None or not postfit_par.exists():
        legacy = _LEGACY_PATHS.get("J0125", {})
        legacy_postfit = legacy.get("postfit_par")
        if legacy_postfit is not None and legacy_postfit.exists():
            postfit_par = legacy_postfit
        else:
            postfit_par = None
    return par, tim, postfit_par


def get_mini_paths() -> Tuple[Path, Path]:
    """Get bundled J1909_mini PAR/TIM paths."""
    golden_dir = _TESTS_DIR / "data_golden"
    par = golden_dir / "J1909_mini.par"
    tim = golden_dir / "J1909_mini.tim"
    return par, tim


def get_golden_reference(name: str = "J1909_mini") -> Optional[dict]:
    """Load golden reference values from JSON."""
    golden_file = _TESTS_DIR / "data_golden" / f"{name}_golden.json"
    if not golden_file.exists():
        return None
    with open(golden_file) as f:
        return json.load(f)


def get_tempo2_fixture_paths(fixture_id: str) -> Tuple[Path, Path]:
    """Get curated Tempo2-style fixture PAR/TIM paths by fixture id."""
    tempo2_dir = _TESTS_DIR / "data_tempo2"
    manifest_file = tempo2_dir / "manifest.json"
    with open(manifest_file) as f:
        manifest = json.load(f)
    for row in manifest:
        if row["id"] == fixture_id:
            return tempo2_dir / row["par"], tempo2_dir / row["tim"]
    known = ", ".join(row["id"] for row in manifest)
    raise KeyError(f"Unknown Tempo2 fixture {fixture_id!r}. Known fixtures: {known}")


def files_exist(par: Optional[Path], tim: Optional[Path]) -> bool:
    """Check if both PAR and TIM files exist."""
    if par is None or tim is None:
        return False
    return par.exists() and tim.exists()


def skip_if_missing(par: Optional[Path], tim: Optional[Path], test_name: str = "") -> bool:
    """Check if files exist, print SKIP message if not."""
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
    """Like skip_if_missing but raises RuntimeError instead of returning False."""
    prefix = f"[{test_name}] " if test_name else ""

    if par is None:
        raise RuntimeError(f"{prefix}PAR path not configured (set JUG_TEST_*_PAR env var)")
    if tim is None:
        raise RuntimeError(f"{prefix}TIM path not configured (set JUG_TEST_*_TIM env var)")
    if not par.exists():
        raise RuntimeError(f"{prefix}PAR file not found: {par}")
    if not tim.exists():
        raise RuntimeError(f"{prefix}TIM file not found: {tim}")


def get_available_datasets() -> dict:
    """Return dict of all configured test datasets that actually exist."""
    datasets = {}

    for name, getter in [
        ("J1713", get_j1713_paths),
        ("J2241", get_j2241_paths),
        ("J1909", get_j1909_paths),
        ("J1022", get_j1022_paths),
        ("J0613", get_j0613_paths),
    ]:
        par, tim = getter()
        if files_exist(par, tim):
            datasets[name] = {"par": par, "tim": tim}

    par, tim, postfit = get_j0125_paths()
    if files_exist(par, tim):
        datasets["J0125"] = {"par": par, "tim": tim}
        if postfit is not None and postfit.exists():
            datasets["J0125"]["postfit_par"] = postfit

    return datasets


if __name__ == "__main__":
    print("JUG Test Data Path Configuration")
    print("=" * 60)

    for name, getter in [
        ("J1713+0747", get_j1713_paths),
        ("J2241-5236", get_j2241_paths),
        ("J1909-3744", get_j1909_paths),
        ("J1022+1001", get_j1022_paths),
        ("J0613-0200", get_j0613_paths),
    ]:
        par, tim = getter()
        exists = files_exist(par, tim)
        status = "✓ AVAILABLE" if exists else "✗ NOT FOUND"
        print(f"\n{name}: {status}")
        print(f"  PAR: {par}")
        print(f"  TIM: {tim}")

    par, tim, postfit = get_j0125_paths()
    exists = files_exist(par, tim)
    status = "✓ AVAILABLE" if exists else "✗ NOT FOUND"
    print(f"\nJ0125-2327: {status}")
    print(f"  PAR: {par}")
    print(f"  TIM: {tim}")
    if postfit is not None and postfit.exists():
        print(f"  POSTFIT PAR: {postfit}")
    else:
        print("  POSTFIT PAR: (not bundled — Tempo2-baseline tests will skip)")

    print("\n" + "=" * 60)
    print("To configure paths, set environment variables:")
    print("  JUG_TEST_DATA_DIR=/path/to/data  (base directory)")
    print("  JUG_TEST_J1713_PAR=/path/to/par  (per-pulsar override)")
    print("  JUG_TEST_J1713_TIM=/path/to/tim")
