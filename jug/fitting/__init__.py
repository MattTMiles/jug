"""JUG fitting module for parameter optimization.

This module provides linearized least squares fitting (standard pulsar timing approach).
"""

from jug.fitting.params import (
    extract_fittable_params,
    pack_params,
    unpack_params,
    get_param_scales
)
from jug.fitting.optimizer import fit_linearized

__all__ = [
    'extract_fittable_params',
    'pack_params',
    'unpack_params',
    'get_param_scales',
    'fit_linearized'
]
