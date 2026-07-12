"""Dev-oracle gates for clkcorr feedback delta vs residual scatter."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("libstempo")
pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

from jug.testing.tempo2_addsat_dtchain_diag import diagnose_addsat_dtchain

from tempo2_fixtures import get_tempo2_fixture

# Picosecond-tier bounds after the 2026-07-12 parity closure (was pinned at
# 1.2171675 ns RMS / 2.33 ns addsat max). The clkcorr feedback delta is no
# longer exactly zero: per tempo2 clkcorr.C the observatory chain is sampled
# at raw SAT and only the post-leap (BIPM) hop sees the accumulated-correction
# feedback, whose 3-vs-1-iteration delta is a genuine sub-ps quantity.
BOUND_RMS_NS = 0.1
BOUND_ADDSAT_MAX_NS = 0.3
BOUND_FEEDBACK_DELTA_NS = 0.01


def test_epta_j0613_feedback_delta_zero_and_debt_pins():
    """Merged IPTA chains: clkcorr feedback delta sub-ps; parity at ps tier."""
    fx = get_tempo2_fixture("epta_j0613_t2_ipta_all")
    report = diagnose_addsat_dtchain(
        fx["par_path"], fx["tim_path"], fixture_id=fx["id"]
    )
    assert report.n_toa == 1369
    assert report.feedback_delta_rms_ns < BOUND_FEEDBACK_DELTA_NS
    # Feedback delta is not the parity driver.
    assert report.predicted_rms_after_feedback_ns == pytest.approx(
        report.residual_rms_ns, rel=0.05
    )
    assert report.residual_rms_ns < BOUND_RMS_NS
    assert report.non_addsat_rms_ns < BOUND_RMS_NS
    assert report.addsat_toa_max_ns < BOUND_ADDSAT_MAX_NS


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
