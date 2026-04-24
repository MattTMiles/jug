"""JUG: JAX-based Pulsar Timing Package.

JUG provides high-performance pulsar timing analysis with JAX acceleration.

IMPORTANT: JAX configuration is NOT done at package import time.
Entry points (jug-gui, jug-fit, jug-compute-residuals) and modules that
use JAX must call ensure_jax_x64() explicitly before any JAX operations.

Example:
    from jug.utils.jax_setup import ensure_jax_x64
    ensure_jax_x64()  # Call before importing JAX-using modules

    from jug.residuals.simple_calculator import compute_residuals_simple
    result = compute_residuals_simple(par_file, tim_file)
"""

import os
# JAX backend selection: default to CPU to avoid RuntimeError tracebacks on
# machines where the CUDA plugin is installed but no GPU is present.
# GPU users should set JAX_PLATFORMS=cuda12 (or rocm, tpu, etc.) in their
# environment BEFORE importing jug.
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
# Suppress noisy PjRt/XLA C++ version-compatibility warnings that appear on
# every JIT compilation when JAX CPU backend is used.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

# Version
__version__ = "0.1.0"
