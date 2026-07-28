"""Fit parity tests for Tempo2-compatible mode.

These tests cover WLS parity for the curated white-noise Tempo2 fixtures.
GLS/TN parity is intentionally deferred: none of the selected fixtures carries
Tempo2 noise parameters (TNRedAmp/RNAMP/TNEF/etc.), so there is no real oracle
case to assert without adding a separate noise-bearing fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.fitting.optimized_fitter import (
    compute_designmatrix,
    fit_parameters_optimized,
    _compute_fd_derivatives_for_mode,
)
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture

WLS_FIT_PARAMS = {
    "epta_j0030_isolated": ["F0", "F1"],
    "epta_j1909_t2": ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2", "PMRA", "PMDEC"],
    "ppta_j1902_ell1h": ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2", "PMRA", "PMDEC"],
}

WLS_MAX_ITER = {
    "ppta_j1902_ell1h": 40,
}

WRMS_DELTA_TOL_NS = 15.0
PARAM_SIGMA_TOL = 0.05
FD1_WRMS_DELTA_TOL_NS = 15.0
FD1_PARAM_REL_TOL = 0.05


def _assert_param_close_to_tempo2(param: str, jug_value: float, ref_entry: dict[str, float]):
    ref_value = float(ref_entry["value"])
    ref_error = float(ref_entry.get("error") or 0.0)
    abs_floor = max(abs(ref_value) * 1.0e-12, 1.0e-24)
    allowed = max(PARAM_SIGMA_TOL * ref_error, abs_floor)
    delta = abs(jug_value - ref_value)
    assert delta < allowed, (
        f"{param}: JUG={jug_value:.17g}, Tempo2={ref_value:.17g}, "
        f"delta={delta:.3e}, error={ref_error:.3e}, allowed={allowed:.3e}"
    )


@pytest.mark.tempo2
def test_tempo2_sandbox_fit_smoke():
    fixture = get_tempo2_fixture("epta_j0030_isolated")
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"], dofit=True, fit_params=["F0"])

    assert ref.ntoa > 0
    assert np.all(np.isfinite(ref.residuals_us))
    assert np.isfinite(ref.wrms_us)
    assert ref.params


@pytest.mark.tempo2
@pytest.mark.parametrize(
    "fixture_id",
    [
        "epta_j0030_isolated",
        "ng5_j1600_tdb_equatorial",
        "ng5_j1600_tdb_ecliptic_cross_engine",
    ],
)
def test_jug_tempo2_fit_parity_f0_wls(fixture_id):
    fixture = get_tempo2_fixture(fixture_id)
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"], dofit=True, fit_params=["F0"])

    jug = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        ["F0"],
        max_iter=2,
        verbose=False,
        compatibility="tempo2",
    )

    assert abs(jug["final_rms"] - ref.wrms_us) < 0.005
    assert abs(float(jug["final_params"]["F0"]) - float(ref.params["F0"]["value"])) < 1e-13
    delta_ns = (np.asarray(jug["residuals_us"]) - ref.residuals_us) * 1000.0
    assert np.sqrt(np.mean(np.square(delta_ns))) < 100.0


@pytest.mark.tempo2
@pytest.mark.parametrize("fixture_id", ["epta_j0030_isolated", "epta_j1909_t2", "ppta_j1902_ell1h"])
def test_jug_tempo2_fit_parity_multi_parameter_wls(fixture_id):
    """Compare multi-parameter WLS fits against libstempo on supported fixtures."""
    fixture = get_tempo2_fixture(fixture_id)
    fit_params = WLS_FIT_PARAMS[fixture_id]

    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        dofit=True,
        fit_params=fit_params,
    )
    jug = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params,
        max_iter=WLS_MAX_ITER.get(fixture_id, 20),
        verbose=False,
        compatibility="tempo2",
    )

    assert jug["converged"], f"{fixture_id}: JUG WLS fit did not converge"
    wrms_delta_ns = abs(jug["final_rms"] - ref.wrms_us) * 1000.0
    assert wrms_delta_ns < WRMS_DELTA_TOL_NS, (
        f"{fixture_id}: JUG WRMS={jug['final_rms']:.9f} us, "
        f"Tempo2 WRMS={ref.wrms_us:.9f} us, delta={wrms_delta_ns:.3f} ns"
    )

    for param in fit_params:
        assert param in ref.params, f"{fixture_id}: {param} missing from Tempo2 snapshot"
        assert param in jug["final_params"], f"{fixture_id}: {param} missing from JUG fit result"
        _assert_param_close_to_tempo2(param, float(jug["final_params"][param]), ref.params[param])


@pytest.mark.tempo2
def test_jug_tempo2_fit_parity_fd1_wls():
    """FD1-only WLS vs libstempo on a fixture with frequency-dependent delays."""
    fixture = get_tempo2_fixture("ppta_j1902_ell1h")
    ref = tempo2_reference(
        fixture["par_path"],
        fixture["tim_path"],
        dofit=True,
        fit_params=["FD1"],
    )
    jug = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        ["FD1"],
        max_iter=10,
        verbose=False,
        compatibility="tempo2",
    )

    assert jug["converged"], "FD1 WLS fit did not converge"
    wrms_delta_ns = abs(jug["final_rms"] - ref.wrms_us) * 1000.0
    assert wrms_delta_ns < FD1_WRMS_DELTA_TOL_NS, (
        f"JUG WRMS={jug['final_rms']:.9f} us, Tempo2 WRMS={ref.wrms_us:.9f} us, "
        f"delta={wrms_delta_ns:.3f} ns"
    )
    _assert_param_close_to_tempo2(
        "FD1", float(jug["final_params"]["FD1"]), ref.params["FD1"]
    )


@pytest.mark.tempo2
def test_tempo2_designmatrix_fd1_matches_fitter_analytic():
    """Export API and fitter share the same analytic FD1 column."""
    from jug.fitting.optimized_fitter import _normalize_fd_column_mode
    from jug.io.par_reader import parse_par_file
    from jug.residuals.simple_calculator import compute_residuals_simple

    fixture = get_tempo2_fixture("ppta_j1902_ell1h")
    dm = compute_designmatrix(
        fixture["par_path"],
        fixture["tim_path"],
        ["FD1"],
        compatibility="tempo2",
        fd_column_mode="tempo2_delay",
    )
    base = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    params = parse_par_file(fixture["par_path"])
    resolved = _normalize_fd_column_mode("tempo2_delay", compatibility="tempo2")
    fit_fd = _compute_fd_derivatives_for_mode(
        params=params,
        freq_mhz=np.asarray(base["freq_bary_mhz"], dtype=np.float64),
        fit_params=["FD1"],
        tdb_mjd=np.asarray(base["tdb_mjd"], dtype=np.float64),
        fd_column_mode=resolved,
    )["FD1"]
    np.testing.assert_allclose(dm.matrix[:, 0], fit_fd, rtol=0.0, atol=1e-15)


@pytest.mark.tempo2
def test_tempo2_fit_fd_column_mode_override_and_validation():
    fixture = get_tempo2_fixture("ppta_j1902_ell1h")
    base = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        ["FD1"],
        max_iter=5,
        verbose=False,
        compatibility="tempo2",
    )
    explicit = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        ["FD1"],
        max_iter=5,
        verbose=False,
        compatibility="tempo2",
        fd_column_mode="tempo2_delay",
    )
    assert abs(float(base["final_params"]["FD1"]) - float(explicit["final_params"]["FD1"])) < 1e-12

    with pytest.raises(ValueError, match="fd_column_mode"):
        fit_parameters_optimized(
            fixture["par_path"],
            fixture["tim_path"],
            ["FD1"],
            max_iter=1,
            verbose=False,
            compatibility="tempo2",
            fd_column_mode="not_a_mode",
        )
