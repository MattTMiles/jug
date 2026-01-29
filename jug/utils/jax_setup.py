"""Centralized JAX configuration for JUG.

This module provides the single source of truth for JAX configuration,
ensuring float64 precision and compilation caching are set up correctly.

IMPORTANT: JAX must NOT be imported at package import time. Entry points
should call ensure_jax_x64() explicitly before any JAX operations.

Usage:
    # In entry points (jug-gui, jug-fit, compute_residuals):
    from jug.utils.jax_setup import ensure_jax_x64
    ensure_jax_x64()  # Call before any JAX imports/operations

    # In modules that use JAX (delays, fitting, residuals):
    from jug.utils.jax_setup import ensure_jax_x64
    ensure_jax_x64()  # Called at module load time

Design:
    - ensure_jax_x64() is idempotent (safe to call multiple times)
    - The function only imports JAX when called, not at module import
    - Compilation cache setup is integrated and also idempotent
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Track if we've already configured (avoid duplicate calls)
_jax_configured = False


def ensure_jax_x64(enable_cache: bool = True) -> bool:
    """Ensure JAX is configured for float64 precision.

    This function MUST be called before any JAX operations in JUG.
    It is safe to call multiple times (idempotent).

    Parameters
    ----------
    enable_cache : bool, optional
        Whether to also configure the JAX compilation cache. Default True.

    Returns
    -------
    bool
        True if JAX was configured (or already configured), False if JAX
        is not available.

    Notes
    -----
    This function:
    1. Imports JAX (only when called, not at module import)
    2. Sets jax_enable_x64=True for pulsar timing precision
    3. Optionally configures the persistent compilation cache

    The function is idempotent - subsequent calls are no-ops.

    Examples
    --------
    >>> from jug.utils.jax_setup import ensure_jax_x64
    >>> ensure_jax_x64()
    True
    >>> # Now safe to import and use JAX
    >>> import jax.numpy as jnp
    """
    global _jax_configured

    if _jax_configured:
        return True

    try:
        import jax

        # Enable float64 precision - CRITICAL for pulsar timing
        # which requires microsecond/nanosecond precision
        jax.config.update('jax_enable_x64', True)

        logger.debug("JAX float64 precision enabled")
        _jax_configured = True

        # Configure compilation cache if requested
        if enable_cache:
            from jug.utils.jax_cache import configure_jax_compilation_cache
            configure_jax_compilation_cache()

        return True

    except ImportError:
        logger.warning("JAX not available - JUG requires JAX for computations")
        return False
    except Exception as e:
        logger.error(f"Failed to configure JAX: {e}")
        return False


def assert_jax_x64() -> None:
    """Assert that JAX float64 mode is enabled.

    Raises
    ------
    RuntimeError
        If JAX is not configured or x64 mode is not enabled.

    Examples
    --------
    >>> from jug.utils.jax_setup import ensure_jax_x64, assert_jax_x64
    >>> ensure_jax_x64()
    True
    >>> assert_jax_x64()  # No error
    """
    if not _jax_configured:
        raise RuntimeError(
            "JAX not configured. Call ensure_jax_x64() before using JAX operations."
        )

    try:
        import jax
        if not jax.config.jax_enable_x64:
            raise RuntimeError(
                "JAX x64 mode not enabled. This should not happen if "
                "ensure_jax_x64() was called."
            )
    except ImportError:
        raise RuntimeError("JAX not available")


def is_jax_configured() -> bool:
    """Check if JAX has been configured.

    Returns
    -------
    bool
        True if ensure_jax_x64() has been called successfully.
    """
    return _jax_configured


def get_jax_info() -> dict:
    """Get information about JAX configuration.

    Returns
    -------
    dict
        Dictionary with JAX configuration information:
        - 'configured': bool, whether JAX is configured
        - 'x64_enabled': bool or None, whether x64 is enabled
        - 'version': str or None, JAX version
        - 'devices': list of str, available devices
    """
    info = {
        'configured': _jax_configured,
        'x64_enabled': None,
        'version': None,
        'devices': [],
    }

    if not _jax_configured:
        return info

    try:
        import jax
        info['version'] = jax.__version__
        info['x64_enabled'] = jax.config.jax_enable_x64
        info['devices'] = [str(d) for d in jax.devices()]
    except ImportError:
        pass

    return info


# For testing: reset the configured state (not for production use)
def _reset_for_testing() -> None:
    """Reset the configured state for testing purposes only."""
    global _jax_configured
    _jax_configured = False
