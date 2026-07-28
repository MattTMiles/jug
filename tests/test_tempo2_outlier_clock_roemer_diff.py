"""Dev-oracle harness: per-TOA clock / Roemer diff for parity outliers."""

from __future__ import annotations

import pytest

from jug.testing.tempo2_outlier_diff import compare_clock_roemer_per_toa
from tempo2_fixtures import get_tempo2_fixture


pytestmark = pytest.mark.dev_oracle


@pytest.mark.parametrize(
    "fixture_id,outlier_threshold_ns,max_roemer_rms_ns",
    [
        ("epta_j0030_isolated", 10.0, 12.0),
        ("wsrt167", 25.0, 3.0),
    ],
)
def test_outlier_clock_roemer_diff(fixture_id, outlier_threshold_ns, max_roemer_rms_ns):
    """Outlier residual gaps are not dominated by Roemer or site-clock mismatch."""
    fx = get_tempo2_fixture(fixture_id)
    report = compare_clock_roemer_per_toa(
        fx["par_path"],
        fx["tim_path"],
        fixture_id=fixture_id,
        outlier_threshold_ns=outlier_threshold_ns,
    )
    assert report.n_toa > 0
    assert report.roemer_rms_ns < max_roemer_rms_ns, (
        f"{fixture_id}: Roemer RMS {report.roemer_rms_ns:.2f} ns exceeds {max_roemer_rms_ns} ns"
    )
    assert report.sat_rms_ns < 1.0, (
        f"{fixture_id}: site arrival (sat/stoas) RMS {report.sat_rms_ns:.2f} ns"
    )


def test_j0030_outliers_closed():
    """J0030's historical 1999 outlier pair is closed (2026-07-12 parity work).

    The two >10 ns TOAs were driven by the astropy-vs-tempo2 spin-axis and
    jpl_pleph JD-rounding gaps; with those fixed the fixture sits at the
    picosecond parity floor with no outliers above threshold.
    """
    fx = get_tempo2_fixture("epta_j0030_isolated")
    report = compare_clock_roemer_per_toa(
        fx["par_path"],
        fx["tim_path"],
        fixture_id="epta_j0030_isolated",
        outlier_threshold_ns=10.0,
    )
    assert len(report.outlier_indices) == 0
    assert report.residual_rms_ns < 0.1
