"""Pre-fit residual parity: IPTA DR2 EPTA J0613-0200 vs libstempo/tempo2."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture
from test_tempo2_residual_parity import (
    FINAL_MAX_DELTA_NS,
    FINAL_P99_DELTA_NS,
    FINAL_RMS_DELTA_NS,
    _assert_residual_parity,
    _delta_stats_ns,
)

# IPTA DR2 EPTA single-PTA dataset (J0613-0200.par + J0613-0200_all.tim).
FIXTURE_ID = "epta_j0613_t2_ipta_all"

# Measured 2026-07-03: JUG(tempo2) - libstempo on all 1369 TOAs.
MEASURED_RMS_NS = 2.892594e6
MEASURED_MAX_NS = 4.875855e6


@pytest.mark.tempo2
@pytest.mark.xfail(
    strict=True,
    reason=(
        "IPTA DR2 EPTA J0613: JUG(tempo2) lacks raw residual parity with libstempo "
        f"(RMS ~{MEASURED_RMS_NS / 1e6:.2f} ms vs {FINAL_RMS_DELTA_NS} ns gate)"
    ),
)
def test_tempo2_mode_epta_j0613_ipta_dr2_residual_parity():
    """JUG(tempo2) pre-fit residuals must match libstempo on the full EPTA dataset."""
    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    _assert_residual_parity(jug, ref, fixture["id"])


@pytest.mark.tempo2
def test_epta_j0613_ipta_dr2_parity_debt_is_large():
    """Pin the known residual debt so regressions are visible in CI."""
    fixture = get_tempo2_fixture(FIXTURE_ID)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])

    assert jug["n_toas"] == ref.ntoa == 1369
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)

    assert stats["rms"] > FINAL_RMS_DELTA_NS
    assert stats["rms"] > 1.0e6  # >> 1 ms
    assert stats["max_abs"] > FINAL_MAX_DELTA_NS
    assert stats["p99_abs"] > FINAL_P99_DELTA_NS

    # Guard against silent drift while the parity gap remains open.
    np.testing.assert_allclose(stats["rms"], MEASURED_RMS_NS, rtol=0.05)
    np.testing.assert_allclose(stats["max_abs"], MEASURED_MAX_NS, rtol=0.05)
