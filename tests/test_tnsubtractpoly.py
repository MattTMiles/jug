"""Regression tests for TNsubtractPoly final-state consistency."""

from pathlib import Path

import numpy as np

from jug.fitting.optimized_fitter import (
    _build_general_fit_setup_from_files,
    _compute_full_model_residuals,
    fit_parameters_optimized_cached,
)


DATA = Path(__file__).parent / "data_golden"
PAR = DATA / "J1909_proper.par"
TIM = DATA / "J1909_proper.tim"
FIT_PARAMS = ["F0", "F1", "DM1", "DM2"]


def _fit(tn_subtract_poly):
    setup = _build_general_fit_setup_from_files(
        PAR, TIM, FIT_PARAMS, None, verbose=False,
    )
    setup.params["TNSUBTRACTPOLY"] = int(tn_subtract_poly)
    result = fit_parameters_optimized_cached(
        setup, max_iter=8, verbose=False, solver_mode="exact",
    )
    return setup, result


def _cleaned_residuals(result, weights):
    residuals = np.asarray(result["residuals_us"], dtype=float)
    noise = np.zeros_like(residuals)
    for label, realization in result["noise_realizations"].items():
        if not label.endswith("_err") and label not in ("DMX", "DMJUMP"):
            noise += np.asarray(realization, dtype=float)
    cleaned = residuals - noise
    return cleaned - np.sum(weights * cleaned) / np.sum(weights)


def test_gls_result_residuals_match_final_parameters():
    setup, result = _fit(tn_subtract_poly=True)
    expected_sec, _, _, _ = _compute_full_model_residuals(setup.params, setup)

    np.testing.assert_allclose(
        result["residuals_us"], expected_sec * 1e6, rtol=0, atol=1e-9,
    )

    residuals_sec = np.asarray(result["residuals_us"]) * 1e-6
    expected_rms = np.sqrt(
        np.sum(setup.weights * residuals_sec**2) / np.sum(setup.weights)
    ) * 1e6
    np.testing.assert_allclose(
        result["final_rms"], expected_rms, rtol=0, atol=1e-12,
    )


def test_tnsubtractpoly_preserves_cleaned_residuals():
    setup_off, result_off = _fit(tn_subtract_poly=False)
    setup_on, result_on = _fit(tn_subtract_poly=True)

    clean_off = _cleaned_residuals(result_off, setup_off.weights)
    clean_on = _cleaned_residuals(result_on, setup_on.weights)
    np.testing.assert_allclose(clean_on, clean_off, rtol=0, atol=5e-5)

    np.testing.assert_allclose(
        result_on["noise_subtracted_rms"],
        result_off["noise_subtracted_rms"],
        rtol=0,
        atol=5e-5,
    )
