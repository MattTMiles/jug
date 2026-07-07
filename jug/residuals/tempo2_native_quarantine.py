"""Quarantined experimental tempo2-native spin path and native-chain switch.

**Not on the production parity route.** Production tempo2 mode uses emission-time
Taylor spin at geometry ``model_mjd`` plus legacy TRACK −2 wrapping (~16 ns on
wsrt167). Phase D Step 2 (2026-07-06) showed ``phase5`` at oracle ``bbat`` +
``track_minus2_frac_phase`` is **~17.5 ns** — worse than production. Do not
enable ``USE_NATIVE_BBAT_PHASE5`` alone for parity gates.

See ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § "Phase D Step 2".

Native-chain flags (two independent switches)
---------------------------------------------
``USE_JAX_TEMPO2_NATIVE_CHAIN`` (default **True**)
    Master switch for tempo2-native **fitting** and ``residual_delta_jax``.
    When True with tempo2 compatibility, ``GeneralFitSetup`` gets
    ``use_jax_tempo2_native_chain`` and ``native_chain_static`` (requires
    ``term_diagnostics`` in the residual cache). This is the production hybrid
    path: host-frozen geometry/clocks + slim differentiable JAX tail.

``USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH`` (default **False**; env ``JUG_TEMPO2_NATIVE_FULL_INGRAPH=1``)
    **Only** selects the slow unified in-graph model inside the native chain.
    Leave off for interactive fitting. Enable only for dev_oracle cross-checks
    against ``compute_tempo2_toa_model_jax`` (multi-minute first JIT).
"""

from __future__ import annotations

import os

# When True, ``compute_phase_residuals`` uses ``compute_tempo2_phase5`` at formBats
# ``bbat`` with ``track_minus2_frac_phase``. Do not enable for parity gates.
USE_NATIVE_BBAT_PHASE5 = False

# Production tempo2 path: host-frozen geometry/clocks + slim JAX tail (BCLT → spin).
USE_JAX_TEMPO2_NATIVE_CHAIN = True

# Production default: host-frozen geometry/clocks + slim JAX tail (BCLT → spin).
# Set True only for dev_oracle cross-checks against the unified in-graph model.
# WARNING: first JIT compile of compute_tempo2_toa_model_jax can take minutes on
# real TOA batches (SPK + EOP + IFTE bootstrap inside one graph). Do not enable
# in interactive fitting sessions or CI fast loops.
USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH = False


def tempo2_native_full_ingraph_enabled() -> bool:
    """Return True only when the slow unified in-graph JAX path is explicitly enabled."""
    if os.environ.get("JUG_TEMPO2_NATIVE_FULL_INGRAPH", "").lower() in ("1", "true", "yes"):
        return True
    return USE_JAX_TEMPO2_NATIVE_FULL_INGRAPH
