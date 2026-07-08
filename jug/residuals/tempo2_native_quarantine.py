"""Backward-compatible re-export of tempo2 graph mode helpers."""

from jug.residuals.tempo2_graph_config import (  # noqa: F401
    TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_NATIVE_GRAPH_FULL,
    TEMPO2_NATIVE_GRAPH_STAGED_BCLT,
    tempo2_native_graph_mode,
)

__all__ = [
    "TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR",
    "TEMPO2_NATIVE_GRAPH_FULL",
    "TEMPO2_NATIVE_GRAPH_STAGED_BCLT",
    "tempo2_native_graph_mode",
]
