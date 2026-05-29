"""Design-matrix parity tests for Tempo2-compatible mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.testing.tempo2_reference import tempo2_reference
from jug.fitting.optimized_fitter import compute_designmatrix

from tempo2_fixtures import get_tempo2_fixture


@pytest.mark.tempo2
def test_tempo2_sandbox_designmatrix_smoke():
    fixture = get_tempo2_fixture("epta_j1909_t2")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        include_designmatrix=True,
    )

    assert ref.designmatrix is not None
    assert ref.designmatrix.shape[0] == ref.ntoa
    assert ref.designmatrix.shape[1] > 0
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
    assert jug.matrix.shape[0] == ref.ntoa
    assert "F0" in jug.labels
    assert ref.designmatrix.shape[1] == 2
    np.testing.assert_allclose(jug.matrix[:, 0], ref.designmatrix[:, 1], rtol=0.0, atol=0.02)
