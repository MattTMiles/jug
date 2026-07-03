"""Real-data autodiff regressions on trimmed MPTA DR2 J0613 (ELL1H)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_files,
    compute_designmatrix,
)
from test_paths import get_j0613_trim300_paths, skip_if_missing

FIT_PARAMS = ["A1", "EPS1", "EPS2"]


@pytest.fixture(scope="module")
def j0613_paths():
    par, tim = get_j0613_trim300_paths()
    if not skip_if_missing(par, tim, "j0613_ell1h_trim300"):
        pytest.skip("trimmed J0613 fixture unavailable")
    return par, tim


def test_j0613_autodiff_zero_delta(j0613_paths):
    par, tim = j0613_paths
    setup = _build_general_fit_setup_from_files(
        par,
        tim,
        FIT_PARAMS,
        compatibility="pint",
        design_matrix_method="autodiff",
        clock_dir=None,
        verbose=False,
    )
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=FIT_PARAMS)
    delta = np.asarray(fn(jnp.zeros(len(FIT_PARAMS))))
    np.testing.assert_allclose(delta, 0.0, atol=1e-8, rtol=0.0)


def test_j0613_autodiff_designmatrix_finite(j0613_paths):
    par, tim = j0613_paths
    result = compute_designmatrix(
        par, tim, FIT_PARAMS, compatibility="pint", design_matrix_method="autodiff"
    )
    assert np.all(np.isfinite(result.matrix))
    assert result.matrix.shape[0] <= 320
