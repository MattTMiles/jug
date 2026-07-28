"""Tempo2-native JAX graph mode selector and quarantined host spin flag.

``tempo2_graph_mode`` selects the differentiable tempo2 timing graph used by
autodiff / ``residual_delta_jax``:

- ``fixed_state_bclt``: freeze host ephemeris/clocks and reference BCLT
  ``dt_ssb``; one-pass BCLT + full tempo2 tail (no BCLT fixed-point scan).
- ``fixed_state_stripped``: same host freeze as ``fixed_state_bclt``; BBAT lite
  subgraph (single pert eval vs host-cached ref BBAT; no phase5/TRACK−2).
- ``fixed_state_stripped`` (default): same host freeze as ``fixed_state_bclt``;
  BBAT lite subgraph for fast compile/eval.
- ``staged_bclt``: freeze ephemeris/clocks/observer state; recompute
  BCLT scan, formBats, Shklovskii, and spin in JAX.
- ``full``: clocks/SPK/EOP/IFTE/tropo/BCLT all inside XLA (oracle/dev only).

There is no backward-compatibility alias layer for graph mode strings — callers
must use the exact canonical names above.

``USE_NATIVE_BBAT_PHASE5`` gates an experimental host path using ``phase5@bbat``
in ``compute_phase_residuals``. Production keeps this ``False`` (strict-parity
probes: worse than Taylor on wsrt167). See ``PARITY_ROADMAP.md``.
"""

from __future__ import annotations

# When True, ``compute_phase_residuals`` uses ``compute_tempo2_phase5`` at formBats
# ``bbat`` with ``track_minus2_frac_phase``. Do not enable for parity gates.
USE_NATIVE_BBAT_PHASE5 = False

TEMPO2_GRAPH_STAGED_BCLT = "staged_bclt"
TEMPO2_GRAPH_FIXED_STATE_BCLT = "fixed_state_bclt"
TEMPO2_GRAPH_FIXED_STATE_STRIPPED = "fixed_state_stripped"
TEMPO2_GRAPH_FULL = "full"

TEMPO2_GRAPH_MODE_DEFAULT = TEMPO2_GRAPH_FIXED_STATE_STRIPPED
_TEMPO2_GRAPH_MODE_DEFAULT = TEMPO2_GRAPH_MODE_DEFAULT
_TEMPO2_GRAPH_MODES = frozenset(
    {
        TEMPO2_GRAPH_STAGED_BCLT,
        TEMPO2_GRAPH_FIXED_STATE_BCLT,
        TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
        TEMPO2_GRAPH_FULL,
    }
)


def tempo2_graph_mode_allowed_strings() -> tuple[str, ...]:
    """Return sorted canonical tempo2 graph mode strings."""
    return tuple(sorted(_TEMPO2_GRAPH_MODES))


def tempo2_graph_mode(mode: str | None = None) -> str:
    """Normalize and return the canonical tempo2-native JAX graph mode.

    When *mode* is ``None``, returns the default ``fixed_state_stripped``.
    """
    if mode is None:
        return _TEMPO2_GRAPH_MODE_DEFAULT
    normalized = str(mode).strip().lower()
    if normalized not in _TEMPO2_GRAPH_MODES:
        allowed = ", ".join(tempo2_graph_mode_allowed_strings())
        raise ValueError(
            f"Unknown tempo2-native graph mode={normalized!r}; expected one of {allowed}"
        )
    return normalized


def is_fixed_state_bclt_mode(mode: str | None) -> bool:
    """Return whether *mode* selects the one-pass fixed BCLT graph."""
    return tempo2_graph_mode(mode) == TEMPO2_GRAPH_FIXED_STATE_BCLT


def is_fixed_state_stripped_mode(mode: str | None) -> bool:
    """Return whether *mode* selects the BBAT-lite stripped fitting graph."""
    return tempo2_graph_mode(mode) == TEMPO2_GRAPH_FIXED_STATE_STRIPPED