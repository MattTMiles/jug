"""Backward-compatible shim — import from ``tempo2_graph_config`` instead."""

from jug.residuals.tempo2_graph_config import (
    TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_NATIVE_GRAPH_FULL,
    TEMPO2_NATIVE_GRAPH_STAGED_BCLT,
    USE_NATIVE_BBAT_PHASE5,
    _TEMPO2_NATIVE_GRAPH_MODE_DEFAULT,
    _TEMPO2_NATIVE_GRAPH_MODES,
    tempo2_native_graph_mode,
)

__all__ = [
    "USE_NATIVE_BBAT_PHASE5",
    "TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR",
    "TEMPO2_NATIVE_GRAPH_STAGED_BCLT",
    "TEMPO2_NATIVE_GRAPH_FULL",
    "tempo2_native_graph_mode",
    "_TEMPO2_NATIVE_GRAPH_MODE_DEFAULT",
    "_TEMPO2_NATIVE_GRAPH_MODES",
]
