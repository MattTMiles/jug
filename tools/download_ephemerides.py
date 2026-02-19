#!/usr/bin/env python3
"""
Pre-download common JPL ephemerides for JUG.

This script downloads the most commonly used planetary ephemeris files
and caches them locally so JUG can work offline or when NASA servers
are unavailable.

Note: JUG automatically downloads ephemerides on first use, but running
this script pre-caches them for better offline support.
"""

import sys
from astropy.coordinates import solar_system_ephemeris
from astropy.config import get_cache_dir

# Most common ephemerides in pulsar timing
RECOMMENDED_EPHEMERIDES = [
    ('de440', 'Current JPL standard (2020+), high accuracy - RECOMMENDED'),
    ('de436', 'Common in existing datasets (downloaded from JPL SSD server)'),
    ('de430', 'Previous standard (2013-2020)'),
]

def download_ephemeris(name):
    """Download and cache an ephemeris file."""
    print(f"Downloading {name.upper()}...", end=' ', flush=True)
    try:
        # For DE436/DE441, use JUG's resolver which knows about SSD server
        from jug.residuals.simple_calculator import _resolve_ephemeris
        import sys
        from io import StringIO
        
        # Capture stderr to suppress download messages
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        
        path = _resolve_ephemeris(name)
        
        sys.stderr = old_stderr
        
        # Verify it's a valid path
        import os
        if os.path.exists(path):
            print("✓")
            return True
        else:
            print(f"✗ (returned {path}, not a file)")
            return False
    except Exception as e:
        print(f"✗ ({e})")
        return False

def main():
    print("JUG Ephemeris Downloader")
    print("=" * 60)
    print(f"Cache directory: {get_cache_dir()}\n")
    
    success_count = 0
    total = len(RECOMMENDED_EPHEMERIDES)
    
    for name, description in RECOMMENDED_EPHEMERIDES:
        print(f"\n{name.upper()}: {description}")
        if download_ephemeris(name):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{total} ephemerides successfully")
    
    if success_count < total:
        print("\nNote: Some ephemerides failed to download.")
        print("This is likely due to NASA server issues.")
        print("JUG will still work with the available ephemerides.")
        return 1
    
    print("\n✓ All recommended ephemerides are now cached!")
    print("\nJUG will automatically download ephemerides on first use.")
    print("Cached ephemerides enable offline operation.")
    return 0

if __name__ == '__main__':
    sys.exit(main())
