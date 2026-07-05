"""DEV ORACLE — TRACK -2 wsrt167 debt pin (delete with oracle harness).

Requires libstempo + tempo2 runtime. Not part of standalone JUG CI.
See ``jug/testing/DEV_ORACLE.md``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import _delta_stats_ns

# Isolated wsrt167 vs libstempo (µs RMS × 1000 → ns), measured 2026-07-05.
MEASURED_ISOLATED_RMS_NS = 2.633360e2
MEASURED_ISOLATED_MAX_NS = 5.500000e2


def test_wsrt167_isolated_track2_debt_pin():
    """Guard wsrt167 isolated RMS while closing TRACK -2 phase debt."""
    fixture = get_tempo2_fixture("wsrt167")
    assert fixture["toa_count"] == 167

    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    assert jug["n_toas"] == ref.ntoa == 167
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)

    assert stats["rms"] > 5.0
    np.testing.assert_allclose(stats["rms"], MEASURED_ISOLATED_RMS_NS, rtol=0.05)
    np.testing.assert_allclose(stats["max_abs"], MEASURED_ISOLATED_MAX_NS, rtol=0.05)
    assert abs(stats["mean"]) < 50.0
