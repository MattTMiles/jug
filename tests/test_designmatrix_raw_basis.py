"""Raw analytic fitter-basis contract for compute_designmatrix."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from jug.fitting.designmatrix_assembly import assemble_analytic_designmatrix
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_files,
    compute_designmatrix,
)
from jug.utils.units import native_to_fit_value

GOLDEN_DIR = Path(__file__).parent / "data_golden"


@pytest.fixture(scope="module")
def j1909_paths():
    par = GOLDEN_DIR / "J1909_mini.par"
    tim = GOLDEN_DIR / "J1909_mini.tim"
    if not par.exists() or not tim.exists():
        pytest.skip("golden J1909 mini dataset not found")
    return par, tim


def test_compute_designmatrix_is_raw_fitter_basis(j1909_paths):
    par, tim = j1909_paths
    labels = ["F0", "RAJ", "DM"]
    result = compute_designmatrix(par, tim, labels)

    assert result.construction == "analytic-fitter"
    assert result.unit_convention == "pint-vela"
    assert result.compatibility == "pint"
    assert not result.matrix.flags.writeable
    assert not result.residuals_us.flags.writeable
    assert not result.errors_us.flags.writeable
    assert len(result.row_tokens) == result.matrix.shape[0]
    assert result.row_tokens[0].startswith("000000|")

    setup = _build_general_fit_setup_from_files(
        Path(par),
        Path(tim),
        labels,
        clock_dir=None,
        verbose=False,
        compatibility="pint",
    )
    np.testing.assert_allclose(
        result.matrix,
        assemble_analytic_designmatrix(setup, labels, output_units="fit"),
        rtol=0,
        atol=0,
    )
    assert result.reference_fit_values == tuple(
        float(native_to_fit_value(p, v))
        for p, v in zip(result.labels, setup.param_values_start, strict=True)
    )


def test_tzr_params_rejected(j1909_paths):
    par, tim = j1909_paths
    with pytest.raises(ValueError):
        compute_designmatrix(par, tim, ["TZRMJD"])


def test_compute_designmatrix_rejects_setup_filtered_parameter(j1909_paths):
    par, tim = j1909_paths
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        par_text = Path(par).read_text()
        # Flag matches no TOAs in the mini fixture -> empty JUMP mask filtered.
        par_with_empty_jump = tmp_path / "empty_jump.par"
        par_with_empty_jump.write_text(
            par_text + "\nJUMP -fe NONEXISTENT_BACKEND 0.0 1\n"
        )
        with pytest.raises(ValueError, match="does not expose a reduced fitter basis"):
            compute_designmatrix(
                par_with_empty_jump, tim, ["F0", "JUMP1"]
            )
