"""JUG: JAX-based Pulsar Timing Package.

CRITICAL: This module MUST be imported before any other jug modules
to ensure JAX is configured correctly for float64 precision.
"""

import jax

# Enable float64 precision for ALL JAX operations
# This is CRITICAL for pulsar timing which requires microsecond precision
jax.config.update('jax_enable_x64', True)

# Version
__version__ = "0.1.0"
