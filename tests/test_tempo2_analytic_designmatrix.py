"""Tight gates: tempo2 analytic design matrix vs simplified-model autodiff."""

from __future__ import annotations

import numpy as np
import pytest

from jug.fitting.designmatrix_assembly import assemble_analytic_designmatrix
from jug.fitting.jax_residual_delta import compute_simplified_autodiff_designmatrix_from_setup
from jug.fitting.optimized_fitter import _compute_designmatrix_from_setup
from jug.utils.constants import SECS_PER_DAY
from jug.utils.units import native_derivative_to_fit_column

pytestmark = pytest.mark.tempo2


def _mean_project(matrix: np.ndarray, weights: np.ndarray, *, compatibility: str) -> np.ndarray:
    if str(compatibility).lower() == "tempo2":
        return matrix - np.mean(matrix, axis=0, keepdims=True)
    w = np.asarray(weights, dtype=np.float64)
    return matrix - (w @ matrix) / w.sum()


def _assert_columns_match(
    analytic: np.ndarray,
    reference: np.ndarray,
    params: list[str],
    *,
    rtol_spin: float = 1e-6,
    rtol_delay: float = 1e-4,
):
    assert analytic.shape == reference.shape
    spin_params = {p for p in params if p.startswith("F") and p[1:].isdigit()}
    for idx, param in enumerate(params):
        rtol = rtol_spin if param in spin_params else rtol_delay
        np.testing.assert_allclose(
            analytic[:, idx],
            reference[:, idx],
            rtol=rtol,
            atol=1e-12,
            err_msg=f"column {param}",
        )


def test_analytic_matches_simplified_autodiff_wsrt167(wsrt167_fit_setup_factory):
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    setup = wsrt167_fit_setup_factory(fit_params, design_matrix_method="analytic")
    setup.residual_delta_jax_cache = None

    analytic = assemble_analytic_designmatrix(setup, fit_params, output_units="fit")
    reference = compute_simplified_autodiff_designmatrix_from_setup(setup, fit_params)

    weights = np.asarray(setup.weights, dtype=np.float64)
    analytic_p = _mean_project(analytic, weights, compatibility=setup.compatibility)
    reference_p = _mean_project(reference, weights, compatibility=setup.compatibility)

    _assert_columns_match(analytic_p, reference_p, fit_params)


@pytest.mark.parametrize("param", ["PB", "EPS1", "EPS2"])
def test_analytic_matches_simplified_autodiff_binary(param):
    from tempo2_test_helpers import build_fit_setup_for_fixture
    from tempo2_fixtures import get_tempo2_fixture

    fixture = get_tempo2_fixture("epta_j1909_t2")
    fit_params = [param]
    setup = build_fit_setup_for_fixture(
        fixture,
        fit_params,
        design_matrix_method="analytic",
    )
    setup.residual_delta_jax_cache = None

    analytic = assemble_analytic_designmatrix(setup, fit_params, output_units="fit")
    reference = compute_simplified_autodiff_designmatrix_from_setup(setup, fit_params)

    weights = np.asarray(setup.weights, dtype=np.float64)
    analytic_p = _mean_project(analytic, weights, compatibility=setup.compatibility)
    reference_p = _mean_project(reference, weights, compatibility=setup.compatibility)

    _assert_columns_match(analytic_p, reference_p, fit_params)


@pytest.mark.parametrize(
    "tempo2_native",
    ["staged_bclt", "fixed_state_bclt", "fixed_state_stripped", "full"],
)
def test_analytic_invariant_across_tempo2_native(wsrt167_fit_setup_factory, tempo2_native):
    """Analytic columns do not depend on tempo2_native graph mode."""
    fit_params = ["F0", "DM"]
    reference = assemble_analytic_designmatrix(
        wsrt167_fit_setup_factory(
            fit_params,
            tempo2_native="staged_bclt",
            design_matrix_method="analytic",
        ),
        fit_params,
        output_units="fit",
    )
    candidate = assemble_analytic_designmatrix(
        wsrt167_fit_setup_factory(
            fit_params,
            tempo2_native=tempo2_native,
            design_matrix_method="analytic",
        ),
        fit_params,
        output_units="fit",
    )
    np.testing.assert_allclose(candidate, reference, rtol=0.0, atol=0.0)


def test_wls_native_columns_match_export_fit_columns():
    from jug.fitting.optimized_fitter import GeneralFitSetup

    tdb_mjd = np.array([55000.0, 55000.25, 55000.5], dtype=float)
    freq_mhz = np.array([820.0, 900.0, 1100.0], dtype=float)
    errors_us = np.full(len(tdb_mjd), 1.0, dtype=float)
    params = {
        "F0": 200.0,
        "PEPOCH": 55000.0,
        "DM": 10.0,
        "DMEPOCH": 55000.0,
        "RAJ": 0.5,
        "DECJ": -0.25,
    }
    dt_sec = (tdb_mjd - params["PEPOCH"]) * SECS_PER_DAY
    ssb_obs = 1e-3 * np.arange(3 * len(tdb_mjd), dtype=float).reshape(len(tdb_mjd), 3)

    setup = GeneralFitSetup(
        params=params,
        fit_param_list=["F0", "RAJ", "DM"],
        compatibility="tempo2",
        fd_column_mode="delay_only",
        design_matrix_method="analytic",
        param_values_start=[200.0, 0.5, 10.0],
        toas_mjd=tdb_mjd,
        freq_mhz=freq_mhz,
        errors_us=errors_us,
        errors_sec=errors_us * 1e-6,
        weights=1.0 / (errors_us * 1e-6) ** 2,
        dt_sec_cached=dt_sec,
        dt_sec_ld=np.asarray(dt_sec, dtype=np.longdouble),
        tdb_mjd=tdb_mjd,
        initial_dm_delay=np.zeros(len(tdb_mjd)),
        dm_params=["DM"],
        spin_params=["F0"],
        binary_params=[],
        astrometry_params=["RAJ"],
        fd_params=[],
        sw_params=[],
        roemer_shapiro_sec=None,
        prebinary_delay_sec=None,
        initial_binary_delay=None,
        ssb_obs_pos_ls=ssb_obs,
        obs_sun_pos_ls=None,
        obs_planet_pos_ls=None,
        initial_astrometric_delay=np.zeros(len(tdb_mjd)),
        initial_fd_delay=None,
        initial_sw_delay=None,
        sw_geometry_pc=None,
        toa_flags=None,
        ecorr_whitener=None,
        red_noise_basis=None,
        red_noise_prior=None,
        dm_noise_basis=None,
        dm_noise_prior=None,
        chromatic_noise_basis=None,
        chromatic_noise_prior=None,
        ecorr_basis=None,
        ecorr_prior=None,
        band_noise_bases=None,
        band_noise_priors=None,
        band_noise_labels=None,
        group_noise_bases=None,
        group_noise_priors=None,
        group_noise_labels=None,
        dmx_design_matrix=None,
        dmx_labels=None,
        initial_dmx_delay=None,
        dmjump_design_matrix=None,
        dmjump_labels=None,
        jump_masks=None,
        fdjump_masks=None,
        fdjump_params=[],
        initial_fdjump_delay=None,
        jump_phase=None,
        tzr_phase=None,
        noise_config=None,
    )

    native = assemble_analytic_designmatrix(setup, ["F0", "RAJ", "DM"], output_units="native")
    export = _compute_designmatrix_from_setup(setup, ["F0", "RAJ", "DM"])

    for idx, param in enumerate(["F0", "RAJ", "DM"]):
        expected = native_derivative_to_fit_column(param, native[:, idx])
        np.testing.assert_allclose(export[:, idx], expected, rtol=0.0, atol=1e-15)