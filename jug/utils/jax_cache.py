"""JAX compilation cache configuration for JUG.

This module configures JAX's persistent compilation cache to reduce cold-start
JIT compilation times across process launches.

The cache directory is chosen in this order:
1. JUG_JAX_CACHE_DIR environment variable (if set)
2. $TMPDIR/jug_jax_cache (if TMPDIR is set and writable)
3. ~/.cache/jug/jax_compilation (default fallback)

Usage:
    from jug.utils.jax_cache import configure_jax_compilation_cache
    configure_jax_compilation_cache()  # Call early, before any JIT functions

Environment Variables:
    JUG_JAX_CACHE_DIR: Override cache directory path
    JUG_JAX_EXPLAIN_CACHE_MISSES: Set to "1" to enable cache miss logging
    JUG_JAX_CACHE_MIN_COMPILE_SECS: Minimum compile time to cache (default: 0.0)
    JUG_JAX_CACHE_MIN_ENTRY_BYTES: Minimum entry size to cache (default: 0)
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Track if we've already configured (avoid duplicate calls)
_configured = False


def get_cache_directory() -> Path:
    """Determine the JAX compilation cache directory.
    
    Returns
    -------
    Path
        Directory path for JAX compilation cache
    """
    # Priority 1: Explicit environment variable
    if env_dir := os.environ.get('JUG_JAX_CACHE_DIR'):
        return Path(env_dir)
    
    # Priority 2: TMPDIR if set and writable
    if tmpdir := os.environ.get('TMPDIR'):
        tmpdir_path = Path(tmpdir)
        if tmpdir_path.exists() and os.access(tmpdir_path, os.W_OK):
            return tmpdir_path / "jug_jax_cache"
    
    # Priority 3: User cache directory (default)
    return Path.home() / ".cache" / "jug" / "jax_compilation"


def configure_jax_compilation_cache(cache_dir: Optional[Path] = None) -> bool:
    """Configure JAX persistent compilation cache.
    
    This should be called early in the program, before any JAX JIT functions
    are invoked, to enable caching of compiled XLA programs.
    
    Parameters
    ----------
    cache_dir : Path, optional
        Override cache directory. If None, uses get_cache_directory().
    
    Returns
    -------
    bool
        True if cache was successfully configured, False otherwise.
    
    Notes
    -----
    This function is safe to call even if JAX is not installed or is an
    incompatible version. It will log a debug message and return False.
    
    The function is idempotent - calling it multiple times has no effect
    after the first successful configuration.
    
    Environment Variables (for tuning):
        JUG_JAX_CACHE_MIN_COMPILE_SECS: Minimum compile time to cache (default: 0.0)
        JUG_JAX_CACHE_MIN_ENTRY_BYTES: Minimum entry size to cache (default: 0)
    
    Examples
    --------
    >>> from jug.utils.jax_cache import configure_jax_compilation_cache
    >>> configure_jax_compilation_cache()
    True
    """
    global _configured
    
    if _configured:
        return True
    
    try:
        import jax
        
        # Determine cache directory
        if cache_dir is None:
            cache_dir = get_cache_directory()
        
        # Ensure directory exists
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure cache thresholds from environment variables
        min_compile_secs = float(os.environ.get('JUG_JAX_CACHE_MIN_COMPILE_SECS', '0.0'))
        min_entry_bytes = int(os.environ.get('JUG_JAX_CACHE_MIN_ENTRY_BYTES', '0'))
        
        # Configure cache - use the most compatible approach for the JAX version
        # JAX 0.9.0+ has jax_enable_compilation_cache
        try:
            # Enable the compilation cache
            jax.config.update("jax_enable_compilation_cache", True)
            jax.config.update("jax_compilation_cache_dir", str(cache_dir))
            
            # Set thresholds to allow caching of smaller/faster compilations
            try:
                jax.config.update("jax_persistent_cache_min_compile_time_secs", min_compile_secs)
                jax.config.update("jax_persistent_cache_min_entry_size_bytes", min_entry_bytes)
            except Exception:
                pass  # These options may not exist in all JAX versions
            
            logger.debug(f"JAX compilation cache enabled: {cache_dir}")
        except Exception:
            # Try the experimental API (JAX 0.4.x)
            try:
                from jax.experimental.compilation_cache import compilation_cache
                compilation_cache.set_cache_dir(str(cache_dir))
                logger.debug(f"JAX compilation cache configured (experimental): {cache_dir}")
            except (ImportError, AttributeError) as e:
                logger.debug(f"JAX compilation cache not available: {e}")
                return False
        
        # Optional: Enable cache miss logging for debugging
        if os.environ.get('JUG_JAX_EXPLAIN_CACHE_MISSES', '').strip() == '1':
            try:
                jax.config.update("jax_explain_cache_misses", True)
                logger.debug("JAX cache miss logging enabled")
            except Exception:
                pass  # Not critical
        
        _configured = True
        return True
        
    except ImportError:
        logger.debug("JAX not installed, skipping compilation cache configuration")
        return False
    except Exception as e:
        logger.debug(f"Failed to configure JAX compilation cache: {e}")
        return False


def get_cache_info() -> dict:
    """Get information about the current JAX cache configuration.
    
    Returns
    -------
    dict
        Dictionary with cache configuration information:
        - 'configured': bool, whether cache is configured
        - 'cache_dir': str or None, cache directory path
        - 'jax_version': str or None, JAX version
    """
    info = {
        'configured': _configured,
        'cache_dir': None,
        'jax_version': None
    }
    
    try:
        import jax
        info['jax_version'] = jax.__version__
        
        if _configured:
            info['cache_dir'] = str(get_cache_directory())
    except ImportError:
        pass
    
    return info
