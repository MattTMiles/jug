"""DEV ORACLE — Step 13 model-epoch batCorr/bbat diagnostic."""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_model_batcorr_probe import compare_model_batcorr_diagnostic
from tempo2_fixtures import get_tempo2_fixture


def test_wsrt167_model_batcorr_closes_libstempo():
    """Model-epoch batCorr debt vs libstempo; bundled ``bat_mjd`` closes lib."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_model_batcorr_diagnostic(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.model_batcorr_vs_lib_rms_ns < 500.0
    assert report.model_bat_vs_lib_rms_ns < 500.0
    assert report.bundled_bat_vs_lib_rms_sec < 1e-6


def test_wsrt167_model_bbat_matches_oracle():
    """Model bbat helper equals model−prebinary/86400 when DSHK≈0."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_model_batcorr_diagnostic(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.model_bbat_vs_oracle_rms_ns < 1.0
    assert report.shklovskii_max_us < 1.0


@pytest.mark.parametrize("fixture_id", ["wsrt167"])
def test_wsrt167_oracle_bbat_pt_gap_documents_open_spin_oracle(fixture_id):
    pytest.importorskip("pytempo")
    fx = get_tempo2_fixture(fixture_id)
    report = compare_model_batcorr_diagnostic(
        fx["par_path"], fx["tim_path"], fixture_id=fixture_id
    )
    assert 250.0 < report.oracle_bbat_vs_pt_rms_ns < 450.0
