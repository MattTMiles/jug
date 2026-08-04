"""Astrometry design-matrix export unit contract tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_files,
    compute_designmatrix,
)
from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives
from jug.io.par_reader import format_dec, format_ra, parse_dec, parse_par_file, parse_ra
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.utils.units import (
    ASTROMETRY_EXPORT_PARAMS,
    column_unit,
    fit_to_native_value,
    fit_unit,
    native_derivative_to_fit_column,
    native_to_fit_value,
    validate_column_units,
)
GOLDEN_DIR = Path(__file__).parent / "data_golden"
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
    par = GOLDEN_DIR / "J1909_proper.par"
    tim = GOLDEN_DIR / "J1909_proper.tim"
    if not par.exists() or not tim.exists():
        pytest.skip("golden J1909 dataset not found")
    return par, tim


def _project_offset(column: np.ndarray) -> np.ndarray:
    return column - np.mean(column)


def _designmatrix_param_value(params: dict, param: str) -> float:
    value = params[param.upper()]
    if param.upper() == "RAJ" and isinstance(value, str):
        return float(parse_ra(value))
    if param.upper() == "DECJ" and isinstance(value, str):
        return float(parse_dec(value))
    return float(value)


def _format_designmatrix_param_value(param: str, value: float) -> str:
    if param.upper() == "RAJ":
        return format_ra(value)
    if param.upper() == "DECJ":
        return format_dec(value)
    return f"{value:.17g}"


def _write_param_variant(par_file: Path | str, param: str, value: float) -> str:
    lines = Path(par_file).read_text().splitlines()
    out_lines = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and stripped.split()[0].upper() == param.upper():
            parts = line.split()
            parts[1] = _format_designmatrix_param_value(param, value)
            out_lines.append(" ".join(parts))
            replaced = True
        else:
            out_lines.append(line)
    if not replaced:
        raise ValueError(f"Parameter {param!r} not found in {par_file}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".par", delete=False)
    with tmp:
        tmp.write("\n".join(out_lines))
        tmp.write("\n")
    return tmp.name


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


@pytest.mark.parametrize("param", ASTROMETRY_EXPORT_PARAMS)
def test_astrometry_column_units_match_fit_unit(param):
    expected = column_unit(param)
    assert validate_column_units([param]) == [expected]
    _ensure = u.Unit(expected)
    assert _ensure == u.s / u.Unit(fit_unit(param))


@pytest.mark.parametrize("compatibility", ["pint"])
@pytest.mark.parametrize("param", ASTROMETRY_FD_RECOMPUTE_PARAMS)
def test_astrometry_column_matches_analytic_fit_unit_derivative(
    fixture, compatibility, param
):
    par_file, tim_file = fixture
    dm = compute_designmatrix(
        par_file,
        tim_file,
        [param],
        compatibility=compatibility,
    )
    setup = _build_general_fit_setup_from_files(
        Path(par_file),
        Path(tim_file),
        [param],
        clock_dir=None,
        verbose=False,
        compatibility=compatibility,
    )
    deriv_native = compute_astrometry_derivatives(
        setup.params,
        setup.tdb_mjd,
        setup.ssb_obs_pos_ls,
        [param],
    )[param]
    expected = native_derivative_to_fit_column(param, np.asarray(deriv_native))
    np.testing.assert_allclose(dm.matrix[:, 0], expected, rtol=0.0, atol=1.0e-15)


@pytest.mark.parametrize("compatibility", ["pint"])
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
    assert dm.column_units == (column_unit(param),)
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


def test_pint_backend_reports_unit_metadata(fixture):
    par_file, tim_file = fixture
    fit_params = list(ASTROMETRY_EXPORT_PARAMS)

    dm = compute_designmatrix(
        par_file,
        tim_file,
        fit_params,
        compatibility="pint",
    )

    assert dm.unit_convention == "pint-vela"
    assert dm.labels == tuple(fit_params)
    assert dm.column_units == tuple(validate_column_units(fit_params))
    for label, unit_str in zip(dm.labels, dm.column_units):
        assert unit_str == column_unit(label)
