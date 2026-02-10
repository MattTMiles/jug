"""Analytical derivatives for solar wind parameter NE_SW.

The solar wind delay is:
    τ_SW = K_DM * NE_SW * geometry_pc / freq²

where geometry_pc = (π * ρ) / (r_AU * sin(ρ)) encodes the Sun-Earth-pulsar
geometry (ρ = π - solar elongation).

The partial derivative is:
    ∂τ_SW / ∂NE_SW = K_DM * geometry_pc / freq²

References:
    Edwards, Hobbs & Manchester (2006), MNRAS 372, 1549 (Tempo2 solar wind)
    You et al. (2007), MNRAS 378, 493
"""

from jug.utils.jax_setup import ensure_jax_x64
ensure_jax_x64()

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List

# K_DM = 1/(2.41e-4) MHz^2 pc cm^-3 s  (DM constant)
K_DM_SEC = 4.148808e3  # s MHz^2 pc^-1 cm^3


@jax.jit
def d_delay_d_NE_SW(
    sw_geometry_pc: jnp.ndarray,
    freq_mhz: jnp.ndarray,
) -> jnp.ndarray:
    """Partial derivative of solar wind delay w.r.t. NE_SW.

    Parameters
    ----------
    sw_geometry_pc : jnp.ndarray
        Solar wind geometry factor in parsecs per TOA.
    freq_mhz : jnp.ndarray
        Barycentric observing frequency in MHz per TOA.

    Returns
    -------
    jnp.ndarray
        d(delay)/d(NE_SW) in seconds per (cm^-3).
    """
    return K_DM_SEC * sw_geometry_pc / (freq_mhz ** 2)


def compute_sw_derivatives(
    sw_geometry_pc: jnp.ndarray,
    freq_mhz: jnp.ndarray,
    fit_params: List[str],
) -> Dict[str, jnp.ndarray]:
    """Compute solar wind parameter derivatives for the design matrix.

    Parameters
    ----------
    sw_geometry_pc : jnp.ndarray
        Solar wind geometry factor in parsecs, shape (N,).
    freq_mhz : jnp.ndarray
        Barycentric frequency in MHz, shape (N,).
    fit_params : list of str
        Parameters to compute derivatives for (should contain 'NE_SW').

    Returns
    -------
    dict
        Mapping parameter name -> derivative array.
    """
    derivatives = {}
    for param in fit_params:
        if param.upper() in ('NE_SW', 'NE1AU'):
            derivatives[param] = d_delay_d_NE_SW(sw_geometry_pc, freq_mhz)
    return derivatives
