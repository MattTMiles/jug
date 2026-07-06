"""DEV ORACLE — formBats clock-chain characterization (Step 11).

Documents the +65 s ``correctionTT_TB`` gap vs libstempo ``batCorrs`` on wsrt167.
These tests characterize the open bug; they do not assert full clock parity yet.
"""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_clock_chain import compare_formbats_clock_chain
from tempo2_fixtures import get_tempo2_fixture


def test_wsrt167_formbats_clock_chain_closed_links():
    """Site arrival and libstempo/pytempo batCorr exports agree at ns level."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_formbats_clock_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.sat_rms_ns < 1.0
    assert report.bat_corr_lib_vs_pt_rms_ns < 1.0


def test_wsrt167_step11_tdis_confound_documents_tt_tb_inversion():
    """Inverting batCorrs with dm+sw as tdis falsely imputes +65 s to TT_TB (Step 12)."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_formbats_clock_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    # Confounded inversion — documents why Step 11 looked like a TT_TB bug.
    assert abs(report.tt_tb_gap_mean_sec - report.utc_to_tdb_mean_sec) < 0.05
    assert report.tt_tb_gap_mean_sec > 60.0
    # True tt2tdb export is ~14 s, not ~79 s implied when dm+sw is wrong tdis.
    assert abs(report.tt_tb_mean_sec - 14.4) < 0.1


def test_wsrt167_production_spin_baseline_unchanged():
    """Production IFTE model_mjd spin path still at ~16 ns on wsrt167."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_formbats_clock_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert 14.0 < report.production_rms_ns < 20.0
