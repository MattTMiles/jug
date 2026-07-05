"""Unit tests for tempo2-native spin helpers."""

from __future__ import annotations

import numpy as np

from jug.residuals.tempo2_spin import compute_tempo2_torb_sec


def test_compute_tempo2_torb_sec_matches_spin_delta_identity():
    """``torb = dt - (bbat - PEPOCH)*86400`` closes tempo2 ``deltaT`` identity."""
    pepoch = 50000.0
    bbat = np.array([50100.25, 50101.5], dtype=np.float64)
    dt = np.array([1.0e8, 1.1e8], dtype=np.float64)
    torb = compute_tempo2_torb_sec(bbat, dt, pepoch)
    delta_t = (bbat - pepoch) * 86400.0 + np.asarray(torb, dtype=np.float64)
    np.testing.assert_allclose(delta_t, np.asarray(dt, dtype=np.float64), rtol=0, atol=0)
