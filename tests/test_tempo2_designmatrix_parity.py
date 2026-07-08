"""Design-matrix parity tests for Tempo2-compatible mode."""

from __future__ import annotations

import numpy as np
import pytest
from astropy import units as u

pytest.importorskip("libstempo")

from jug.testing.tempo2_reference import tempo2_reference
from jug.fitting.optimized_fitter import compute_designmatrix
from jug.utils.units import validate_column_units

from tempo2_fixtures import get_tempo2_fixture
from tempo2_fixture_assertions import (
    assert_column_matches,
    tempo2_to_pint_vela_scale,
)

TARGET_COLUMNS = ("F0", "F1", "DM", "RAJ", "DECJ", "PB", "A1", "EPS1", "EPS2")

FIXTURE_COLUMNS = {
    "epta_j0030_isolated": ("F0", "F1", "DM"),
    "epta_j1909_t2": TARGET_COLUMNS,
    "ppta_j1902_ell1h": TARGET_COLUMNS + ("FD1",),
    "ng5_j1600_tdb_equatorial": ("F0",),
    "ng5_j1600_tdb_ecliptic_cross_engine": ("F0",),
}


@pytest.mark.tempo2
def test_tempo2_sandbox_designmatrix_smoke():
    fixture = get_tempo2_fixture("epta_j1909_t2")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        include_designmatrix=True,
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert ref.designmatrix.shape[0] == ref.ntoa
    assert ref.designmatrix.shape[1] > 0
    assert ref.designmatrix.shape[1] == len(ref.designmatrix_labels)
    assert np.all(np.isfinite(ref.designmatrix))


@pytest.mark.tempo2
def test_tempo2_designmatrix_column_parity_f0():
    fixture = get_tempo2_fixture("epta_j1909_t2")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=["F0"],
        include_designmatrix=True,
    )

    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        ["F0"],
        compatibility="tempo2",
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert jug.matrix.shape[0] == ref.ntoa
    assert "F0" in jug.labels
    assert jug.unit_convention == "pint-vela"
    assert jug.column_units == validate_column_units(jug.labels)
    assert str(u.Unit(jug.column_units[0])) == str(u.Unit("s / Hz"))
    assert ref.designmatrix_labels == ["Offset", "F0"]
    np.testing.assert_allclose(jug.matrix[:, 0], ref.designmatrix[:, 1], rtol=0.0, atol=0.02)


@pytest.mark.tempo2
@pytest.mark.parametrize(
    "fixture_id",
    [
        "epta_j0030_isolated",
        "epta_j1909_t2",
        "ppta_j1902_ell1h",
        pytest.param(
            "ng5_j1600_tdb_equatorial",
            marks=pytest.mark.slow,
        ),
        pytest.param(
            "ng5_j1600_tdb_ecliptic_cross_engine",
            marks=pytest.mark.slow,
        ),
    ],
)
def test_tempo2_designmatrix_columns_match_libstempo(fixture_id):
    """Compare real JUG timing columns against libstempo by label.

    libstempo includes an explicit offset column.  JUG residual columns are
    generated from mean-subtracted residuals, so equality is checked after
    projecting both columns orthogonal to the constant offset.
    """
    fixture = get_tempo2_fixture(fixture_id)
    fit_params = list(FIXTURE_COLUMNS[fixture_id])
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        include_designmatrix=True,
    )
    jug = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params,
        compatibility="tempo2",
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix_labels is not None
    assert jug.matrix.shape[0] == ref.ntoa
    assert jug.labels == fit_params
    assert jug.unit_convention == "pint-vela"
    assert jug.column_units == validate_column_units(jug.labels)
    assert len(jug.column_units) == len(jug.labels)

    ref_label_to_idx = {label: idx for idx, label in enumerate(ref.designmatrix_labels)}
    for jug_idx, param in enumerate(jug.labels):
        assert param in ref_label_to_idx
        ref_col = ref.designmatrix[:, ref_label_to_idx[param]] * tempo2_to_pint_vela_scale(param)
        assert_column_matches(param, jug.matrix[:, jug_idx], ref_col)
