"""DEV ORACLE — TRACK -2 wsrt167 parity (delete with oracle harness).

Requires libstempo + tempo2 runtime. Not part of standalone JUG CI.
See ``jug/testing/DEV_ORACLE.md``, ``TEMPO2_NATIVE_CLOCK_STATUS.md`` § Phase D,
and ``tests/test_tempo2_track2_pnnew.py`` (Step 1 done; Step 2 ``phase5@bbat`` ruled out).
Next: WSRT ``-padd`` / ``jump_phase``; outlier idx 85 (+110 ns).
"""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import _assert_residual_parity


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
