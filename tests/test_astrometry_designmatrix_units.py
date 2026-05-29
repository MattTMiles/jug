"""Astrometry design-matrix export unit contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from jug.fitting.optimized_fitter import (
    _designmatrix_param_value,
    _write_param_variant,
    compute_designmatrix,
)
from jug.io.par_reader import parse_par_file
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.units import (
    ASTROMETRY_EXPORT_PARAMS,
    column_unit,
    fd_step_in_fit_units,
    fit_to_native_value,
    fit_unit,
    native_to_fit_value,
    validate_column_units,
)
from tempo2_fixtures import get_tempo2_fixture

FIXTURE_ID = "epta_j1909_t2"
ASTROMETRY_FIT_DELTAS = {
    "RAJ": 1.0e-8,
    "DECJ": 1.0e-8,
    "PMRA": 1.0e-6,
    "PMDEC": 1.0e-6,
}
# PX is included in metadata/backend tests; FD recomputation covers its fit-unit path.
ASTROMETRY_FD_RECOMPUTE_PARAMS = ASTROMETRY_EXPORT_PARAMS
ASTROMETRY_FORWARD_PERTURB_PARAMS = ("RAJ", "DECJ", "PMRA", "PMDEC")


@pytest.fixture(scope="module")
def fixture():
    row = get_tempo2_fixture(FIXTURE_ID)
    return row["par_path"], row["tim_path"]


def _project_offset(column: np.ndarray) -> np.ndarray:
    return column - np.mean(column)


def _residual_change_for_fit_delta(
    par_file: Path | str,
    tim_file: Path | str,
    param: str,
    delta_fit: float,
    *,
    compatibility: str,
) -> np.ndarray:
    """Return residual change (seconds) for a +delta_fit perturbation in fit units."""
    params = parse_par_file(par_file)
    value_native = _designmatrix_param_value(params, param)
    value_fit = native_to_fit_value(param, value_native)
    perturbed_native = fit_to_native_value(param, value_fit + delta_fit)

    base = compute_residuals_simple(
        par_file,
        tim_file,
        verbose=False,
        compatibility=compatibility,
    )
    plus_file = _write_param_variant(par_file, param, perturbed_native)
    try:
        plus = compute_residuals_simple(
            plus_file,
            tim_file,
            verbose=False,
            compatibility=compatibility,
        )
    finally:
        Path(plus_file).unlink(missing_ok=True)

    return (plus["residuals_us"] - base["residuals_us"]) * 1.0e-6


def _manual_fd_column_fit_units(
    par_file: Path | str,
    tim_file: Path | str,
    param: str,
    *,
    compatibility: str,
) -> np.ndarray:
    """Recompute one FD column using the fit-unit step helpers."""
    params = parse_par_file(par_file)
    value_native = _designmatrix_param_value(params, param)
    value_fit = native_to_fit_value(param, value_native)
    step_fit = fd_step_in_fit_units(param, value_fit)
    plus_native = fit_to_native_value(param, value_fit + step_fit)
    minus_native = fit_to_native_value(param, value_fit - step_fit)
    plus_file = _write_param_variant(par_file, param, plus_native)
    minus_file = _write_param_variant(par_file, param, minus_native)
    try:
        plus = compute_residuals_simple(
            plus_file,
            tim_file,
            verbose=False,
            compatibility=compatibility,
        )
        minus = compute_residuals_simple(
            minus_file,
            tim_file,
            verbose=False,
            compatibility=compatibility,
        )
    finally:
        Path(plus_file).unlink(missing_ok=True)
        Path(minus_file).unlink(missing_ok=True)
    return -((plus["residuals_us"] - minus["residuals_us"]) * 1.0e-6) / (2.0 * step_fit)


@pytest.mark.parametrize("param", ASTROMETRY_EXPORT_PARAMS)
def test_astrometry_column_units_match_fit_unit(param):
    expected = column_unit(param)
    assert validate_column_units([param]) == [expected]
    _ensure = u.Unit(expected)
    assert _ensure == u.s / u.Unit(fit_unit(param))


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
@pytest.mark.parametrize("param", ASTROMETRY_FD_RECOMPUTE_PARAMS)
def test_astrometry_fd_column_uses_fit_unit_step(fixture, compatibility, param):
    par_file, tim_file = fixture
    dm = compute_designmatrix(
        par_file,
        tim_file,
        [param],
        compatibility=compatibility,
    )
    manual = _manual_fd_column_fit_units(
        par_file,
        tim_file,
        param,
        compatibility=compatibility,
    )
    np.testing.assert_allclose(dm.matrix[:, 0], manual, rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize("compatibility", ["pint", "tempo2"])
@pytest.mark.parametrize("param", ASTROMETRY_FORWARD_PERTURB_PARAMS)
def test_astrometry_forward_perturbation_matches_column(fixture, compatibility, param):
    par_file, tim_file = fixture
    delta_fit = ASTROMETRY_FIT_DELTAS[param]

    dm = compute_designmatrix(
        par_file,
        tim_file,
        [param],
        compatibility=compatibility,
    )
    assert dm.column_units == [column_unit(param)]
    assert dm.unit_convention == "pint-vela"

    column = np.asarray(dm.matrix[:, 0], dtype=np.float64)
    delta_res = _residual_change_for_fit_delta(
        par_file,
        tim_file,
        param,
        delta_fit,
        compatibility=compatibility,
    )

    expected = -column * delta_fit
    projected_expected = _project_offset(expected)
    projected_delta = _project_offset(delta_res)

    scale = np.dot(projected_delta, projected_expected) / np.dot(
        projected_expected, projected_expected
    )
    assert np.isfinite(scale)
    np.testing.assert_allclose(scale, 1.0, rtol=0.05, atol=0.05)


def test_pint_tempo2_backends_report_same_unit_metadata(fixture):
    par_file, tim_file = fixture
    fit_params = list(ASTROMETRY_EXPORT_PARAMS)

    pint_dm = compute_designmatrix(
        par_file,
        tim_file,
        fit_params,
        compatibility="pint",
    )
    tempo2_dm = compute_designmatrix(
        par_file,
        tim_file,
        fit_params,
        compatibility="tempo2",
    )

    assert pint_dm.unit_convention == tempo2_dm.unit_convention == "pint-vela"
    assert pint_dm.labels == tempo2_dm.labels == fit_params
    assert pint_dm.column_units == tempo2_dm.column_units
    assert pint_dm.column_units == validate_column_units(fit_params)
    for label, unit_str in zip(pint_dm.labels, pint_dm.column_units):
        assert unit_str == column_unit(label)
