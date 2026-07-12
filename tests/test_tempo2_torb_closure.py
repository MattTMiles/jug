"""DEV ORACLE — native torb vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from tempo2_test_helpers import delta_ns


TORB_JAX_DEBT_RMS_NS = 1.0


def test_native_torb_sec_vs_pytempo_wsrt167(wsrt167_native_terms, wsrt167_pytempo_oracle):
    """``Tempo2Terms.torb_sec`` vs pytempo (host ``dt_emit`` vs JAX ``bbat`` closure)."""
    torb = np.asarray(jax.device_get(wsrt167_native_terms.torb_sec), dtype=np.float64)
    delta = delta_ns(torb, wsrt167_pytempo_oracle.fields["torb_sec"])
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < TORB_JAX_DEBT_RMS_NS, f"torb rms={rms:.1f} ns"