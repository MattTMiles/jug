"""DEV ORACLE — Step 12 batCorrs vs production model_mjd epoch chain."""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_clock_chain import compare_batcorr_epoch_chain
from tempo2_fixtures import get_tempo2_fixture


def test_wsrt167_batcorr_matches_model_mjd_minus_prebinary():
    """libstempo batCorrs equals (model_mjd - sat)*86400 - prebinary_delay."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_batcorr_epoch_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.batcorr_model_identity_rms_ns < 500.0
    assert report.batcorr_utc_model_tdb_rms_ns < 500.0


def test_wsrt167_formbats_dm_sw_does_not_match_batcorr():
    """Naive formBats with dm+sw misses utc_to_tdb (~65 s) in batCorrs oracle."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_batcorr_epoch_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.formbats_dm_sw_rms_ns > 1e10
    assert report.formbats_tdis_implied_rms_ns < 1.0


def test_wsrt167_model_tdb_tracks_ifte_linear():
    """model_mjd - tdb_mjd matches IFTE epoch map better than raw tt2tdb scatter."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_batcorr_epoch_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.model_tdb_vs_ifte_linear_rms_ns < report.model_tdb_vs_tt_tb_rms_ns
    assert abs(report.model_tdb_mean_sec - report.tt_tb_mean_sec) < 0.01
