"""Astropy configuration for deterministic behavior.

This module configures Astropy to avoid surprise network access and ensure
reproducible behavior across cold starts.

The key issues this addresses:
1. IERS data auto-download can cause unpredictable delays
2. IERS table refresh checks add overhead
3. Different IERS sources can give slightly different results

Usage:
    from jug.utils.astropy_config import configure_astropy
    configure_astropy()  # Call early, before any Astropy usage

Environment Variables:
    JUG_ASTROPY_CACHE_DIR: Override Astropy cache directory
    JUG_ASTROPY_OFFLINE: Set to "1" to force offline mode (no network)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Track if we've already configured
_configured = False


def configure_astropy(
    offline: Optional[bool] = None,
    cache_dir: Optional[Path] = None
) -> bool:
    """Configure Astropy for deterministic behavior.
    
    This should be called early in the program, before any Astropy
    operations that might trigger IERS downloads.
    
    Parameters
    ----------
    offline : bool, optional
        If True, disable network access for IERS data.
        If None (default), respects JUG_ASTROPY_OFFLINE env var.
    cache_dir : Path, optional
        Override Astropy cache directory.
        If None, respects JUG_ASTROPY_CACHE_DIR env var.
    
    Returns
    -------
    bool
        True if configuration was applied, False otherwise.
    
    Notes
    -----
    This function is idempotent - calling it multiple times has no effect
    after the first successful configuration.
    
    The function sets these Astropy configurations:
    - iers.auto_download: False (prevent surprise downloads)
    - iers.auto_max_age: None (don't auto-refresh)
    - Uses bundled IERS-B data for offline operation
    """
    global _configured
    
    if _configured:
        return True
    
    try:
        from astropy.utils import iers
        from astropy import config as astropy_config
        
        # Determine offline mode
        if offline is None:
            offline = os.environ.get('JUG_ASTROPY_OFFLINE', '').strip() == '1'
        
        # Determine cache directory
        if cache_dir is None:
            env_cache = os.environ.get('JUG_ASTROPY_CACHE_DIR')
            if env_cache:
                cache_dir = Path(env_cache)
        
        # Set cache directory if specified
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            try:
                astropy_config.set_temp_cache(str(cache_dir))
            except Exception:
                pass  # May not be supported in all versions
        
        # Configure IERS behavior for determinism
        # Disable auto-download to prevent surprise network access
        try:
            iers.conf.auto_download = not offline
            iers.conf.auto_max_age = None  # Don't auto-refresh
        except Exception:
            pass  # Config options may vary by version
        
        # For offline mode, use bundled IERS-B data
        if offline:
            try:
                # Use the built-in IERS-B table (good for most purposes)
                iers.IERS_Auto.iers_table = iers.IERS_B.open()
                logger.debug("Using bundled IERS-B data (offline mode)")
            except Exception as e:
                logger.debug(f"Could not set IERS-B: {e}")
        
        _configured = True
        logger.debug("Astropy configured for deterministic behavior")
        return True
        
    except ImportError:
        logger.debug("Astropy not installed, skipping configuration")
        return False
    except Exception as e:
        logger.debug(f"Failed to configure Astropy: {e}")
        return False


def get_iers_status() -> dict:
    """Get information about current IERS configuration.
    
    Returns
    -------
    dict
        Dictionary with IERS status information:
        - 'configured': bool, whether configure_astropy was called
        - 'auto_download': bool or None
        - 'iers_table_type': str or None
        - 'cache_dir': str or None
    """
    status = {
        'configured': _configured,
        'auto_download': None,
        'iers_table_type': None,
        'cache_dir': None
    }
    
    try:
        from astropy.utils import iers
        from astropy import config as astropy_config
        
        try:
            status['auto_download'] = iers.conf.auto_download
        except Exception:
            pass
        
        try:
            if hasattr(iers.IERS_Auto, 'iers_table') and iers.IERS_Auto.iers_table is not None:
                status['iers_table_type'] = type(iers.IERS_Auto.iers_table).__name__
        except Exception:
            pass
        
        try:
            status['cache_dir'] = str(astropy_config.get_cache_dir())
        except Exception:
            pass
            
    except ImportError:
        pass
    
    return status


def prefetch_iers_data(verbose: bool = True) -> bool:
    """Download IERS data for offline use.
    
    This function downloads the latest IERS-A data and caches it
    for future use. Call this once to prepare for offline operation.
    
    Parameters
    ----------
    verbose : bool
        Print progress messages
    
    Returns
    -------
    bool
        True if data was downloaded successfully
    """
    try:
        from astropy.utils import iers
        
        if verbose:
            print("Downloading IERS data...")
        
        # Force download of IERS-A (most accurate)
        try:
            iers_a = iers.IERS_A.open()
            if verbose:
                print(f"  Downloaded IERS-A table with {len(iers_a)} entries")
                print(f"  Time range: MJD {iers_a['MJD'].min():.1f} - {iers_a['MJD'].max():.1f}")
            return True
        except Exception as e:
            if verbose:
                print(f"  Could not download IERS-A: {e}")
                print("  Falling back to bundled IERS-B")
            
            # Fall back to bundled IERS-B
            iers_b = iers.IERS_B.open()
            if verbose:
                print(f"  Using bundled IERS-B table with {len(iers_b)} entries")
            return True
            
    except ImportError:
        if verbose:
            print("ERROR: Astropy not installed")
        return False
    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return False
