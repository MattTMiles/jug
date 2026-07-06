"""JAX tempo2-native clock/delay/spin chain.

Production API exports JAX entry points only. Deprecated NumPy reference code
lives in ``chain_numpy`` and is intentionally not re-exported here.
"""

from jug.residuals.tempo2_native.types import Tempo2NativeTerms, native_terms_to_numpy
from jug.residuals.tempo2_native.chain_jax import (
    compute_tempo2_native_residuals_jax,
    compute_tempo2_native_terms_jax,
    prepare_native_chain_from_simple_result,
)

__all__ = [
    "Tempo2NativeTerms",
    "native_terms_to_numpy",
    "compute_tempo2_native_terms_jax",
    "compute_tempo2_native_residuals_jax",
    "prepare_native_chain_from_simple_result",
]
