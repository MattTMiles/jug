"""Tests for true JAX autodiff design-matrix assembly."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jug.fitting.jax_residual_delta import (
    compute_autodiff_designmatrix_from_setup,
    make_residual_delta_jax_fn,
)
from jug.fitting.optimized_fitter import (
    _compute_designmatrix_from_setup,
    GeneralFitSetup,
)
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY


def _setup(fit_params, *, method="analytic"):
    tdb_mjd = np.array([55000.0, 55000.25, 55000.5, 55000.75, 55001.0], dtype=float)
    freq_mhz = np.array([820.0, 900.0, 1100.0, 1400.0, 1600.0], dtype=float)
    errors_us = np.full(len(tdb_mjd), 1.0, dtype=float)
    params = {
        "F0": 200.0,
        "F1": -1.0e-15,
        "PEPOCH": 55000.0,
        "DM": 10.0,
        "DMEPOCH": 55000.0,
    }
    dt_sec = (tdb_mjd - params["PEPOCH"]) * SECS_PER_DAY
    initial_dm_delay = K_DM_SEC * params["DM"] / (freq_mhz**2)
    return GeneralFitSetup(
        params=params,
        fit_param_list=list(fit_params),
        compatibility="pint",
        fd_column_mode="delay_only",
        design_matrix_method=method,
        param_values_start=[float(params.get(p, 0.0)) for p in fit_params],
        toas_mjd=tdb_mjd,
        freq_mhz=freq_mhz,
        errors_us=errors_us,
        errors_sec=errors_us * 1.0e-6,
        weights=1.0 / (errors_us * 1.0e-6) ** 2,
        dt_sec_cached=dt_sec,
        dt_sec_ld=np.asarray(dt_sec, dtype=np.longdouble),
        tdb_mjd=tdb_mjd,
        initial_dm_delay=initial_dm_delay,
        dm_params=[p for p in fit_params if p.startswith("DM")],
        spin_params=[p for p in fit_params if p.startswith("F") and p[1:].isdigit()],
        binary_params=[],
        astrometry_params=[],
        fd_params=[],
        sw_params=[],
        roemer_shapiro_sec=None,
        prebinary_delay_sec=None,
        initial_binary_delay=None,
        ssb_obs_pos_ls=None,
        obs_sun_pos_ls=None,
        obs_planet_pos_ls=None,
        initial_astrometric_delay=None,
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


def test_residual_delta_jax_zero_is_zero():
    fit_params = ["F0", "DM"]
    setup = _setup(fit_params, method="autodiff")
    residual_fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)

    delta = np.asarray(residual_fn(jnp.zeros(len(fit_params), dtype=jnp.float64)))

    np.testing.assert_allclose(delta, 0.0, atol=1.0e-14, rtol=0.0)


def test_autodiff_matrix_builder_does_not_call_full_model_residuals(monkeypatch):
    fit_params = ["F0", "DM"]
    setup = _setup(fit_params, method="autodiff")

    def fail_full_model_residuals(*args, **kwargs):
        raise AssertionError("autodiff design matrix must not call host residuals")

    monkeypatch.setattr(
        "jug.fitting.optimized_fitter._compute_full_model_residuals",
        fail_full_model_residuals,
    )

    matrix = _compute_designmatrix_from_setup(setup, fit_params)

    assert matrix.shape == (len(setup.tdb_mjd), len(fit_params))
    assert np.all(np.isfinite(matrix))


def test_autodiff_matches_mean_projected_analytic_spin_dm():
    fit_params = ["F0", "DM"]
    analytic_setup = _setup(fit_params, method="analytic")
    autodiff_setup = _setup(fit_params, method="autodiff")

    analytic = _compute_designmatrix_from_setup(analytic_setup, fit_params)
    autodiff = compute_autodiff_designmatrix_from_setup(autodiff_setup, fit_params)
    weights = np.asarray(analytic_setup.weights, dtype=np.float64)
    analytic = analytic - (weights @ analytic) / weights.sum()

    np.testing.assert_allclose(autodiff, analytic, rtol=2.0e-8, atol=1.0e-13)
