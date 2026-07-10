"""Unit tests for tempo2-native spin helpers."""

from __future__ import annotations

import numpy as np

from jug.residuals.tempo2_spin import (
    compute_tempo2_torb_sec,
    track_minus2_frac_phase,
)
from jug.testing.tempo2_track2_oracle import compute_pn_new_relative


def test_compute_tempo2_torb_sec_matches_spin_delta_identity():
    """``torb = dt - (bbat - PEPOCH)*86400`` closes tempo2 ``deltaT`` identity."""
    pepoch = 50000.0
    bbat = np.array([50100.25, 50101.5], dtype=np.float64)
    dt = np.array([1.0e8, 1.1e8], dtype=np.float64)
    torb = compute_tempo2_torb_sec(bbat, dt, pepoch)
    delta_t = (bbat - pepoch) * 86400.0 + np.asarray(torb, dtype=np.float64)
    np.testing.assert_allclose(delta_t, np.asarray(dt, dtype=np.float64), rtol=0, atol=0)


def test_track_minus2_uses_pn_relative_to_first_toa():
    """``pnAct`` uses ``-pn[i] - -pn[0]`` so ``addPhase`` stays O(1) turns."""
    f0 = 326.0
    bbat = np.array([50000.0, 50001.0, 50002.0], dtype=np.float64)
    phase5 = np.array([1.2e10, 1.2e10 + f0 * 86400.0, 1.2e10 + 2 * f0 * 86400.0])
    pn_new = compute_pn_new_relative(phase5, bbat, f0)
    pn_tim = pn_new + np.int64(-999)  # absolute-looking offset at obsn[0]
    pn_add = np.array([-1, -1, -1], dtype=np.int64)
    frac, _ = track_minus2_frac_phase(phase5, bbat, f0, pn_tim, pn_add)
    np.testing.assert_allclose(frac, np.ones(3), rtol=0, atol=1e-12)

