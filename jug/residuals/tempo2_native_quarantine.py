"""Quarantined experimental tempo2-native spin path and graph-mode selector.

**Not on the production parity route.** Production tempo2 mode uses emission-time
Taylor spin at geometry ``model_mjd`` plus legacy TRACK −2 wrapping (~16 ns on
wsrt167). Phase D Step 2 (2026-07-06) showed ``phase5`` at oracle ``bbat`` +
``track_minus2_frac_phase`` is **~17.5 ns** — worse than production. Do not
enable ``USE_NATIVE_BBAT_PHASE5`` alone for parity gates.

See ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § "Phase D Step 2".

Tempo2-native JAX graph modes
-----------------------------
``JUG_TEMPO2_NATIVE_GRAPH_MODE`` selects the differentiable tempo2 timing graph:

- ``fixed_state_nonlinear``: freeze host ephemeris/clocks and reference BCLT
  ``dt_ssb``; recompute Roemer/Shapiro/DM/formBats/spin nonlinearly without a
  BCLT fixed-point scan.
- ``staged_bclt`` (default): freeze ephemeris/clocks/observer state; recompute
  BCLT scan, formBats, Shklovskii, and spin in JAX.
- ``full``: clocks/SPK/EOP/IFTE/tropo/BCLT all inside XLA (oracle/dev only).
"""

from __future__ import annotations

import os

# When True, ``compute_phase_residuals`` uses ``compute_tempo2_phase5`` at formBats
# ``bbat`` with ``track_minus2_frac_phase``. Do not enable for parity gates.
USE_NATIVE_BBAT_PHASE5 = False

TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR = "fixed_state_nonlinear"
TEMPO2_NATIVE_GRAPH_STAGED_BCLT = "staged_bclt"
TEMPO2_NATIVE_GRAPH_FULL = "full"

_TEMPO2_NATIVE_GRAPH_MODE_DEFAULT = TEMPO2_NATIVE_GRAPH_STAGED_BCLT
_TEMPO2_NATIVE_GRAPH_MODES = {
    TEMPO2_NATIVE_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_NATIVE_GRAPH_STAGED_BCLT,
    TEMPO2_NATIVE_GRAPH_FULL,
}


def tempo2_native_graph_mode() -> str:
    """Return the active tempo2-native JAX graph mode."""
    mode = os.environ.get(
        "JUG_TEMPO2_NATIVE_GRAPH_MODE",
        _TEMPO2_NATIVE_GRAPH_MODE_DEFAULT,
    )
    mode = mode.strip().lower().replace("-", "_")
    if mode not in _TEMPO2_NATIVE_GRAPH_MODES:
        allowed = ", ".join(sorted(_TEMPO2_NATIVE_GRAPH_MODES))
        raise ValueError(
            f"Unknown JUG_TEMPO2_NATIVE_GRAPH_MODE={mode!r}; expected one of {allowed}"
        )
    return mode
