"""Tempo2-native JAX graph mode selector and quarantined host spin flag.

``JUG_TEMPO2_NATIVE_GRAPH_MODE`` selects the differentiable tempo2 timing graph
used by autodiff / ``residual_delta_jax``:

- ``fixed_state_nonlinear``: freeze host ephemeris/clocks and reference BCLT
  ``dt_ssb``; recompute Roemer/Shapiro/DM/formBats/spin nonlinearly without a
  BCLT fixed-point scan.
- ``staged_bclt`` (default): freeze ephemeris/clocks/observer state; recompute
  BCLT scan, formBats, Shklovskii, and spin in JAX.
- ``full``: clocks/SPK/EOP/IFTE/tropo/BCLT all inside XLA (oracle/dev only).

``USE_NATIVE_BBAT_PHASE5`` gates an experimental host path using ``phase5@bbat``
in ``compute_phase_residuals``. Production keeps this ``False`` (strict-parity
probes: worse than Taylor on wsrt167). See ``PARITY_ROADMAP.md``.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jug.timing import Tempo2NativeConfig

# When True, ``compute_phase_residuals`` uses ``compute_tempo2_phase5`` at formBats
# ``bbat`` with ``track_minus2_frac_phase``. Do not enable for parity gates.
USE_NATIVE_BBAT_PHASE5 = False

TEMPO2_GRAPH_FIXED_STATE_NONLINEAR = "fixed_state_nonlinear"
TEMPO2_GRAPH_STAGED_BCLT = "staged_bclt"
TEMPO2_GRAPH_FULL = "full"

_TEMPO2_GRAPH_MODE_DEFAULT = TEMPO2_GRAPH_STAGED_BCLT
_TEMPO2_GRAPH_MODES = {
    TEMPO2_GRAPH_FIXED_STATE_NONLINEAR,
    TEMPO2_GRAPH_STAGED_BCLT,
    TEMPO2_GRAPH_FULL,
}


def tempo2_native_graph_mode(
    config: "Tempo2NativeConfig | None" = None,
) -> str:
    """Return the active tempo2-native JAX graph mode.

    Precedence: explicit ``config.graph_mode``, then ``JUG_TEMPO2_NATIVE_GRAPH_MODE`` env
    (deprecated), then default ``staged_bclt``.
    """
    if config is not None:
        mode = str(config.graph_mode).strip().lower().replace("-", "_")
    else:
        env_mode = os.environ.get("JUG_TEMPO2_NATIVE_GRAPH_MODE")
        if env_mode is not None:
            warnings.warn(
                "JUG_TEMPO2_NATIVE_GRAPH_MODE is deprecated; pass tempo2_native= "
                "to TimingSession or store Tempo2NativeConfig on GeneralFitSetup.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = env_mode.strip().lower().replace("-", "_")
        else:
            mode = _TEMPO2_GRAPH_MODE_DEFAULT
    if mode not in _TEMPO2_GRAPH_MODES:
        allowed = ", ".join(sorted(_TEMPO2_GRAPH_MODES))
        raise ValueError(
            f"Unknown tempo2-native graph mode={mode!r}; expected one of {allowed}"
        )
    return mode
