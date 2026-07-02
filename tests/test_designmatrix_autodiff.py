"""Smoke tests for opt-in autodiff design-matrix assembly."""

from __future__ import annotations

import numpy as np
import pytest

from jug.fitting.optimized_fitter import compute_designmatrix
from tempo2_fixtures import get_tempo2_fixture


@pytest.fixture(scope="module")
def isolated_fixture():
    row = get_tempo2_fixture("epta_j0030_isolated")
    return row["par_path"], row["tim_path"]


def test_compute_designmatrix_autodiff_smoke(isolated_fixture):
    par_path, tim_path = isolated_fixture
    fit_params = ["F0", "F1", "DM"]

    analytic = compute_designmatrix(
        par_path,
        tim_path,
        fit_params,
        compatibility="tempo2",
        design_matrix_method="analytic",
    )
    autodiff = compute_designmatrix(
        par_path,
        tim_path,
        fit_params,
        compatibility="tempo2",
        design_matrix_method="autodiff",
    )

    assert analytic.matrix.shape == autodiff.matrix.shape == (len(analytic.errors_us), 3)
    assert analytic.labels == autodiff.labels == fit_params
    assert np.all(np.isfinite(autodiff.matrix))
    assert np.any(np.abs(autodiff.matrix) > 0.0)


def test_compute_designmatrix_rejects_unknown_method(isolated_fixture):
    par_path, tim_path = isolated_fixture
    with pytest.raises(ValueError, match="design_matrix_method"):
        compute_designmatrix(
            par_path,
            tim_path,
            ["F0"],
            design_matrix_method="jax",
        )
