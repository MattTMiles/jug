"""DEV ORACLE — native bbat parity vs tempo2 (Phase 2).

``bbat_mjd`` is an assembled MJD epoch scalar, not a delay-component sum. On wsrt167
the unified JAX path matches ``bat_corr_days`` to tempo2 at ~1 ns but ``bbat_mjd`` at
~304 ns because JUG uses float64 single-sum assembly while tempo2 ``formBats.C`` uses
split ``long double`` summation (``sat + tt/86400 + (other)/86400``).

This is documented in ``PARITY_ROADMAP.md`` § "formBats bat_mjd / bbat_mjd assembly".
A failing test here does **not** mean delay physics is wrong; gate ``bat_corr_days``
in ``test_tempo2_native_formbats_closure.py`` for that.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from tempo2_native_test_helpers import delta_ns


def test_native_bbat_strict_formbats_wsrt167(wsrt167_native_terms, wsrt167_pytempo_oracle):
    """Epoch assembly gate — expected ~304 ns fail until split summation is ported.

    Not a delay-physics gate: ``bat_corr_days`` is ~1 ns on the same native terms.
    Not necessarily a residual blocker: native ``torb`` is a closure against ``bbat``.
    """
    bbat = np.asarray(jax.device_get(wsrt167_native_terms.bbat_mjd), dtype=np.float64)
    delta = delta_ns(bbat, wsrt167_pytempo_oracle.fields["bbat_mjd"], is_mjd=True)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0
