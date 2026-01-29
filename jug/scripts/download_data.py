#!/usr/bin/env python3
"""
JUG Data Prefetch Command
=========================

Download and cache data needed for cold starts:
- IERS data for coordinate transforms
- JPL ephemeris kernels (DE440)

Also provides data integrity verification via manifest.json.

This is optional but recommended for:
- First-time setup
- Offline/HPC environments
- Reproducible science

Usage:
    python -m jug.scripts.download_data           # Download all data
    python -m jug.scripts.download_data --verify  # Verify data integrity
    python -m jug.scripts.download_data --status  # Show cache status
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


# =============================================================================
# Data Paths and Manifest
# =============================================================================

def get_data_dir() -> Path:
    """Get the JUG data directory path."""
    # Navigate from this script to the data directory
    script_dir = Path(__file__).parent
    return script_dir.parent.parent / "data"


def get_cache_dir() -> Path:
    """Get the JUG cache directory path.

    Uses JUG_CACHE_DIR environment variable if set,
    otherwise defaults to ~/.cache/jug/
    """
    cache_dir = os.environ.get('JUG_CACHE_DIR')
    if cache_dir:
        return Path(cache_dir)
    return Path.home() / ".cache" / "jug"


def load_manifest() -> dict:
    """Load the data manifest.

    Returns
    -------
    dict
        Manifest with file checksums and sizes

    Raises
    ------
    FileNotFoundError
        If manifest.json doesn't exist
    """
    manifest_path = get_data_dir() / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path) as f:
        return json.load(f)


# =============================================================================
# Verification Functions
# =============================================================================

def verify_data_integrity(verbose: bool = True) -> dict:
    """Verify data file integrity against manifest.

    Parameters
    ----------
    verbose : bool
        Print status messages

    Returns
    -------
    dict
        Results with keys: valid, missing, corrupted, verified
    """
    data_dir = get_data_dir()

    try:
        manifest = load_manifest()
    except FileNotFoundError as e:
        if verbose:
            print(f"ERROR: {e}")
        return {'valid': False, 'missing': [], 'corrupted': [], 'verified': [], 'error': str(e)}

    results = {
        'valid': True,
        'missing': [],
        'corrupted': [],
        'verified': [],
    }

    if verbose:
        print(f"\nVerifying data integrity (manifest v{manifest.get('version', 'unknown')})...")
        print(f"Data directory: {data_dir}")

    for rel_path, expected in manifest.get('files', {}).items():
        full_path = data_dir / rel_path

        if not full_path.exists():
            results['missing'].append(rel_path)
            results['valid'] = False
            if verbose:
                print(f"  MISSING: {rel_path}")
            continue

        # Verify sha256
        with open(full_path, 'rb') as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest()

        if actual_sha256 != expected['sha256']:
            results['corrupted'].append({
                'path': rel_path,
                'expected': expected['sha256'],
                'actual': actual_sha256,
            })
            results['valid'] = False
            if verbose:
                print(f"  CORRUPTED: {rel_path}")
                print(f"    Expected: {expected['sha256'][:16]}...")
                print(f"    Actual:   {actual_sha256[:16]}...")
        else:
            results['verified'].append(rel_path)
            if verbose:
                print(f"  OK: {rel_path}")

    if verbose:
        print()
        print(f"Verified: {len(results['verified'])}")
        print(f"Missing: {len(results['missing'])}")
        print(f"Corrupted: {len(results['corrupted'])}")

    return results


def run_offline_safe() -> bool:
    """Check if JUG can run offline without network access.

    This verifies that all essential data files are present.
    Call this at the start of computation to fail early if data is missing.

    Returns
    -------
    bool
        True if all essential data is present

    Raises
    ------
    RuntimeError
        If essential data is missing and JUG_OFFLINE=1 is set
    """
    offline_mode = os.environ.get('JUG_OFFLINE', '').lower() in ('1', 'true', 'yes')

    results = verify_data_integrity(verbose=False)

    if not results['valid']:
        missing = results['missing']
        corrupted = [r['path'] for r in results['corrupted']]

        msg = "Data integrity check failed.\n"
        if missing:
            msg += f"  Missing: {', '.join(missing)}\n"
        if corrupted:
            msg += f"  Corrupted: {', '.join(corrupted)}\n"

        if offline_mode:
            msg += "\nJUG_OFFLINE=1 is set - refusing to proceed without verified data."
            raise RuntimeError(msg)
        else:
            # Just warn, don't fail
            print(f"WARNING: {msg}")
            return False

    return True


def prefetch_iers():
    """Prefetch IERS data."""
    print("\n[1/2] Prefetching IERS data...")
    
    from jug.utils.astropy_config import prefetch_iers_data
    success = prefetch_iers_data(verbose=True)
    
    return success


def prefetch_ephemeris():
    """Prefetch JPL ephemeris kernel."""
    print("\n[2/2] Prefetching JPL ephemeris (DE440)...")
    
    try:
        from astropy.coordinates import solar_system_ephemeris, get_body_barycentric
        from astropy.time import Time
        
        # Trigger ephemeris download by computing a position
        with solar_system_ephemeris.set('de440'):
            t = Time(59000.0, format='mjd', scale='tdb')
            pos = get_body_barycentric('earth', t)
            print(f"  DE440 ephemeris loaded successfully")
            print(f"  Test: Earth position at MJD 59000 = ({pos.x.to('km').value:.1f}, {pos.y.to('km').value:.1f}, {pos.z.to('km').value:.1f}) km")
        
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def show_cache_status():
    """Show status of all caches."""
    print("\n" + "=" * 60)
    print("CACHE STATUS")
    print("=" * 60)
    
    # IERS status
    from jug.utils.astropy_config import get_iers_status
    iers_status = get_iers_status()
    print(f"\nIERS:")
    print(f"  Configured: {iers_status['configured']}")
    print(f"  Auto-download: {iers_status['auto_download']}")
    print(f"  Table type: {iers_status['iers_table_type']}")
    print(f"  Cache dir: {iers_status['cache_dir']}")
    
    # JAX cache status
    from jug.utils.jax_cache import get_cache_info
    jax_status = get_cache_info()
    print(f"\nJAX Compilation Cache:")
    print(f"  Configured: {jax_status['configured']}")
    print(f"  Cache dir: {jax_status['cache_dir']}")
    print(f"  JAX version: {jax_status['jax_version']}")
    
    # Geometry cache status
    from jug.utils.geom_cache import get_geometry_cache
    geom_cache = get_geometry_cache()
    geom_stats = geom_cache.get_cache_stats()
    print(f"\nGeometry Disk Cache:")
    print(f"  Enabled: {geom_stats['enabled']}")
    print(f"  Cache dir: {geom_stats['cache_dir']}")
    print(f"  Entries: {geom_stats['entry_count']}")
    print(f"  Size: {geom_stats['total_size_mb']:.2f} MB")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Download and cache data for JUG cold starts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m jug.scripts.download_data           # Download all data
    python -m jug.scripts.download_data --verify  # Verify data integrity
    python -m jug.scripts.download_data --status  # Show cache status only

Environment Variables:
    JUG_CACHE_DIR     Override default cache directory (~/.cache/jug/)
    JUG_OFFLINE=1     Fail if data is missing (no network downloads)
"""
    )

    parser.add_argument(
        '--status',
        action='store_true',
        help='Show cache status without downloading'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify data integrity against manifest'
    )

    parser.add_argument(
        '--clear-geom-cache',
        action='store_true',
        help='Clear geometry disk cache'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("JUG Data Prefetch")
    print("=" * 60)

    if args.clear_geom_cache:
        from jug.utils.geom_cache import get_geometry_cache
        cache = get_geometry_cache()
        count = cache.clear()
        print(f"Cleared {count} geometry cache entries")
        return

    if args.verify:
        results = verify_data_integrity(verbose=True)
        if results['valid']:
            print("\n✓ All data files verified successfully")
            sys.exit(0)
        else:
            print("\n✗ Data integrity check failed")
            sys.exit(1)

    if args.status:
        show_cache_status()
        return

    # Prefetch all data
    iers_ok = prefetch_iers()
    eph_ok = prefetch_ephemeris()
    
    # Show final status
    show_cache_status()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if iers_ok and eph_ok:
        print("✓ All data prefetched successfully")
        print("\nJUG is ready for offline use.")
        sys.exit(0)
    else:
        print("⚠ Some data could not be prefetched")
        print("  JUG will still work but may need network access on first use.")
        sys.exit(1)


if __name__ == '__main__':
    main()
