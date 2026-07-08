"""Dev-oracle gates for clkcorr feedback delta vs residual scatter."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("libstempo")
pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

from jug.testing.tempo2_addsat_dtchain_diag import diagnose_addsat_dtchain

from tempo2_fixtures import get_tempo2_fixture

# Measured 2026-07-08 vs live libstempo on epta_j0613_t2_ipta_all.
MEASURED_RMS_NS = 1.2171675226204746
MEASURED_NON_ADDSAT_RMS_NS = 1.22  # bulk ≈ overall at sub-ns
MEASURED_ADDSAT_MAX_NS = 2.3264357314349837


def test_epta_j0613_feedback_delta_zero_and_debt_pins():
    """Merged IPTA chains: clkcorr 3−1 iter delta is zero; debt pinned at ~1.2 ns RMS."""
    fx = get_tempo2_fixture("epta_j0613_t2_ipta_all")
    report = diagnose_addsat_dtchain(
        fx["par_path"], fx["tim_path"], fixture_id=fx["id"]
    )
    assert report.n_toa == 1369
    assert report.feedback_delta_rms_ns == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(report.corr_delta_vs_feedback) or abs(
        report.corr_delta_vs_feedback
    ) < 0.1
    # Track B gate not met on merged chains — feedback delta is not the driver.
    assert report.predicted_rms_after_feedback_ns == pytest.approx(
        report.residual_rms_ns, rel=0.01
    )
    assert report.residual_rms_ns == pytest.approx(MEASURED_RMS_NS, rel=0.05)
    assert report.non_addsat_rms_ns == pytest.approx(
        MEASURED_NON_ADDSAT_RMS_NS, rel=0.05
    )
    assert report.addsat_toa_max_ns == pytest.approx(MEASURED_ADDSAT_MAX_NS, rel=0.05)


def test_epta_j0613_addsat_sat_matches_pytempo():
    """JUG SAT (incl -addsat) must match pytempo sat_mjd at ns level."""
    fx = get_tempo2_fixture("epta_j0613_t2_ipta_all")
    report = diagnose_addsat_dtchain(
        fx["par_path"], fx["tim_path"], fixture_id=fx["id"]
    )
    assert report.sat_vs_pytempo_max_ns < 1.0
    assert report.addsat_sat_vs_pytempo_max_ns < 1.0
    assert report.addsat_closure_max_sec < 1e-6
    assert report.addsat_toa_indices == [247, 256, 561]
    # Addsat TOAs closed to low-ns vs libstempo (2026-07-08 re-baseline).
    assert report.addsat_toa_max_ns < 5.0
    assert report.non_addsat_rms_ns < 5.0
