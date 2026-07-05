"""Quarantined experimental tempo2-native spin path.

**Not on the production parity route.** Production tempo2 mode uses emission-time
Taylor spin at geometry ``model_mjd`` plus legacy TRACK −2 wrapping (~16 ns on
wsrt167). The native ``phase5`` + ``track_minus2_frac_phase`` stack here is kept
for future research only; enabling it regresses parity (~36 ns on wsrt167).

See ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § "Parity review (2026-07-05)".
"""

from __future__ import annotations

# When True, ``compute_phase_residuals`` uses ``compute_tempo2_phase5`` at formBats
# ``bbat`` with ``track_minus2_frac_phase``. Do not enable for parity gates.
USE_NATIVE_BBAT_PHASE5 = False
