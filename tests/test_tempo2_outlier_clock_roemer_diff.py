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


def test_j0030_outliers_not_roemer_dominated():
    """The two 1999 TOAs on J0030 fail parity but Roemer matches libstempo."""
    fx = get_tempo2_fixture("epta_j0030_isolated")
    report = compare_clock_roemer_per_toa(
        fx["par_path"],
        fx["tim_path"],
        fixture_id="epta_j0030_isolated",
        outlier_threshold_ns=10.0,
    )
    assert len(report.outlier_indices) == 2
    for idx in report.outlier_indices:
        row = report.rows[idx]
        assert abs(row.roemer_diff_ns) < 15.0
        assert abs(row.sat_diff_ns) < 1.0
