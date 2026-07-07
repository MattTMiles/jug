"""DEV ORACLE — formBats clock-chain characterization (Step 11).

Historically documented a +65 s ``correctionTT_TB`` confound vs libstempo
``batCorrs`` on wsrt167.  The confound is resolved: inverting batCorrs with
the canonical formBats delay signs now reproduces JUG's tt2tdb term exactly,
and the production spin path is at ~1.4 ns after the troposphere-in-dt and
longdouble-wrap fixes (2026-07-07).
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


def test_wsrt167_step11_tdis_confound_resolved():
    """Inverting batCorrs with canonical formBats signs recovers tt2tdb exactly.

    Step 11 originally imputed a spurious +65 s gap to TT_TB because dm+sw was
    used as tdis; with the troposphere folded into the dt chain the inversion
    closes and the implied TT_TB matches JUG's tt2tdb export at the ns level.
    """
    fx = get_tempo2_fixture("wsrt167")
    report = compare_formbats_clock_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert abs(report.tt_tb_gap_mean_sec) < 1e-6
    assert abs(report.formbats_offset_mean_sec) < 1e-6
    # True tt2tdb export is ~14 s on this fixture.
    assert abs(report.tt_tb_mean_sec - 14.4) < 0.1


def test_wsrt167_production_spin_baseline():
    """Production spin path at ~1.4 ns on wsrt167 after tropo+wrap fixes."""
    fx = get_tempo2_fixture("wsrt167")
    report = compare_formbats_clock_chain(
        fx["par_path"], fx["tim_path"], fixture_id="wsrt167"
    )
    assert report.production_rms_ns < 2.5
