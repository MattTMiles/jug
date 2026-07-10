"""High-level tempo2 JAX residual evaluation."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from jug.residuals.tempo2.types import Tempo2Terms
from .fit_setup import prepare_tempo2_chain_from_simple_result
from .terms import compute_tempo2_residuals_jax


def compute_eval_residuals_jax(
    *,
    params: dict,
    toas: list[Any],
    jug_result: dict,
    pulse_numbers=None,
    pn_add=None,
    jump_phase=None,
    tzr_phase=None,
    subtract_mean: bool = True,
    mean_mode: str = "unweighted",
    track_val: int = -2,
    weights=None,
    addsat_sec=None,
) -> tuple[jnp.ndarray, jnp.ndarray, Tempo2Terms]:
    """Production residuals: unified in-graph delay chain + spin/track."""
    del addsat_sec  # -addsat is applied to SAT at timfile read (readTimfile.C)
    native = prepare_tempo2_chain_from_simple_result(jug_result, params, toas)
    jump_j = None if jump_phase is None else jnp.asarray(jump_phase, dtype=jnp.float64)
    tzr_j = None if tzr_phase is None else jnp.asarray(tzr_phase, dtype=jnp.float64)
    pn_j = None if pulse_numbers is None else jnp.asarray(pulse_numbers, dtype=jnp.int64)
    pn_add_j = None if pn_add is None else jnp.asarray(pn_add, dtype=jnp.int64)
    if weights is None:
        weights = jnp.ones(native.sat_mjd.shape[0], dtype=jnp.float64)
    return compute_tempo2_residuals_jax(
        native_terms=native,
        params=params,
        weights=jnp.asarray(weights, dtype=jnp.float64),
        pulse_numbers=pn_j,
        pn_add=pn_add_j,
        jump_phase=jump_j,
        tzr_phase=tzr_j,
        subtract_mean=subtract_mean,
        mean_mode=mean_mode,
        track_val=track_val,
    )
