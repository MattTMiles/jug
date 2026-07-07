"""DEV ORACLE — TRACK -2 wsrt167 parity (delete with oracle harness).

Requires libstempo + tempo2 runtime. Not part of standalone JUG CI.
See ``jug/testing/DEV_ORACLE.md``, ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § Phase D,
and ``tests/test_tempo2_track2_pnnew.py`` (Step 1 done; Step 2 ``phase5@bbat`` ruled out).
Next: Step 14 torb/bbat oracle (~330 ns); Step 13 batCorr temp prototype done.
"""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import _assert_residual_parity


@pytest.mark.dev_oracle
@pytest.mark.tempo2
@pytest.mark.xfail(
    strict=True,
    reason="wsrt167 strict 5 ns gate — bulk spin floor ~15.5 ns remains",
)
def test_wsrt167_isolated_track2_tempo2_parity():
    """wsrt167 TCB TRACK -2 vs libstempo (strict ns gate; currently failing)."""
    fixture = get_tempo2_fixture("wsrt167")
    assert fixture["toa_count"] == 167

    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
def test_wsrt167_track2_bulk_spin_debt_pin():
    """Fast debt pin: wsrt167 RMS must stay below 25 ns (bulk floor ~15.5 ns)."""
    from test_tempo2_residual_parity import _delta_stats_ns

    fixture = get_tempo2_fixture("wsrt167")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] < 25.0, f"wsrt167 rms={stats['rms']:.2f} ns"
