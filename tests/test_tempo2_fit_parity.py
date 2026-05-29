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

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture

WLS_FIT_PARAMS = {
    "epta_j0030_isolated": ["F0", "F1"],
    "epta_j1909_t2": ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2", "PMRA", "PMDEC"],
    "ppta_j1902_ell1h": ["F0", "F1", "PB", "A1", "TASC", "EPS1", "EPS2", "PMRA", "PMDEC"],
}

WRMS_DELTA_TOL_NS = 15.0
PARAM_SIGMA_TOL = 0.05


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
def test_jug_tempo2_fit_parity_f0_wls():
    fixture = get_tempo2_fixture("epta_j0030_isolated")
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
        max_iter=20,
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
