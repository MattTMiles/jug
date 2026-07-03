"""Tests for true JAX autodiff design-matrix assembly."""

from __future__ import annotations

import dataclasses
import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting.binary_registry import compute_binary_delay
from jug.fitting.forward_delay import compute_total_delay_change
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


BINARY_CASES = {
    "ELL1": dict(BINARY="ELL1", A1=10.0, PB=5.0, TASC=55000.0, EPS1=1e-3, EPS2=-2e-3),
    "T2_ELL1": dict(BINARY="T2", A1=10.0, PB=5.0, TASC=55000.0, EPS1=1e-3, EPS2=-2e-3),
    "ELL1H": dict(
        BINARY="ELL1H", A1=10.0, PB=5.0, TASC=55000.0, EPS1=1e-3, EPS2=-2e-3, H3=5e-8, STIG=0.7
    ),
    "DD": dict(BINARY="DD", A1=10.0, PB=5.0, T0=55000.0, ECC=0.1, OM=45.0),
    "DDK": dict(
        BINARY="DDK",
        A1=10.0,
        PB=5.0,
        T0=55000.0,
        ECC=0.1,
        OM=45.0,
        KIN=60.0,
        KOM=30.0,
        PX=1.0,
        PMRA=5.0,
        PMDEC=-3.0,
        RAJ=1.0,
        DECJ=-0.5,
    ),
    "DDK_PM": dict(
        BINARY="DDK",
        A1=10.0,
        PB=5.0,
        T0=55000.0,
        ECC=0.1,
        OM=45.0,
        KIN=60.0,
        KOM=30.0,
        PX=1.0,
        PMRA=5.0,
        PMDEC=-3.0,
        RAJ=1.0,
        DECJ=-0.5,
    ),
    "DDK_ECL_PM": dict(
        BINARY="DDK",
        A1=10.0,
        PB=5.0,
        T0=55000.0,
        ECC=0.1,
        OM=45.0,
        KIN=60.0,
        KOM=30.0,
        PX=1.0,
        _ecliptic_coords=True,
        _ecliptic_frame="IERS2010",
        _ecliptic_lon_deg=120.0,
        _ecliptic_lat_deg=-30.0,
        PMELONG=5.0,
        PMELAT=-3.0,
    ),
}
BINARY_FIT = {
    "ELL1": ["A1", "EPS1", "EPS2"],
    "T2_ELL1": ["A1", "EPS1", "EPS2"],
    "ELL1H": ["A1", "H3", "STIG"],
    "DD": ["A1", "ECC", "OM"],
    "DDK": ["A1", "KIN", "KOM"],
    "DDK_PM": ["A1", "PMRA", "PMDEC"],
    "DDK_ECL_PM": ["A1", "PMELONG", "PMELAT"],
}


def _binary_setup(case, method="autodiff"):
    fit_params = list(BINARY_FIT[case])
    base = _setup(["F0"], method=method)
    n = len(base.tdb_mjd)
    params = dict(base.params)
    params.update(BINARY_CASES[case])
    params.setdefault("PEPOCH", 55000.0)
    prebinary = np.zeros(n, dtype=float)
    obs_pos = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    toas_prebinary = base.tdb_mjd - prebinary / SECS_PER_DAY
    init_binary = np.asarray(
        compute_binary_delay(toas_prebinary, params, obs_pos_ls=obs_pos), dtype=float
    )
    return dataclasses.replace(
        base,
        params=params,
        fit_param_list=fit_params,
        param_values_start=[float(params.get(p, 0.0)) for p in fit_params],
        binary_params=["A1"],
        prebinary_delay_sec=prebinary,
        initial_binary_delay=init_binary,
        ssb_obs_pos_ls=obs_pos,
        dm_params=[],
        spin_params=[],
        initial_dm_delay=None,
        binary_plan=None,
    ), fit_params


@pytest.mark.parametrize("case", list(BINARY_CASES))
def test_residual_delta_jax_zero_is_zero_binary(case):
    setup, fit_params = _binary_setup(case)
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    delta = np.asarray(fn(jnp.zeros(len(fit_params))))
    np.testing.assert_allclose(delta, 0.0, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("case", list(BINARY_CASES))
def test_np_vs_jax_delay_change_parity(case):
    setup, fit_params = _binary_setup(case)
    params = dict(setup.params)
    for p in fit_params:
        params[p] = float(params.get(p, 0.0)) + 1e-6
    d_np = np.asarray(compute_total_delay_change(params, setup, xp=np))
    d_jx = np.asarray(compute_total_delay_change(params, setup, xp=jnp))
    np.testing.assert_allclose(d_jx, d_np, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("case", ["ELL1", "DD"])
def test_autodiff_binary_column_matches_analytic(case):
    setup, fit_params = _binary_setup(case, method="autodiff")
    analytic = _compute_designmatrix_from_setup(_binary_setup(case, "analytic")[0], fit_params)
    autodiff = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    w = np.asarray(setup.weights)
    analytic = analytic - (w @ analytic) / w.sum()
    np.testing.assert_allclose(autodiff, analytic, rtol=2e-2, atol=1e-9)


def test_ddk_proper_motion_is_traceable():
    setup, fit_params = _binary_setup("DDK_PM", method="autodiff")
    mtx = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    for name in ("PMRA", "PMDEC"):
        col = mtx[:, fit_params.index(name)]
        assert np.linalg.norm(col) > 0.0, f"{name} column is all-zero -> PM not traced"


def test_ddk_ecliptic_proper_motion_aliases_are_traceable():
    setup, fit_params = _binary_setup("DDK_ECL_PM", method="autodiff")
    mtx = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    for name in ("PMELONG", "PMELAT"):
        col = mtx[:, fit_params.index(name)]
        assert np.linalg.norm(col) > 0.0, f"{name} column is all-zero -> ecliptic PM alias not traced"


def test_dd_orthometric_fit_is_rejected():
    from jug.fitting.binary_delay_plan import resolve_binary_structure

    ref = dict(
        BINARY="DDH",
        A1=10.0,
        PB=5.0,
        T0=55000.0,
        ECC=0.1,
        OM=45.0,
        H3=5e-8,
        STIG=0.7,
    )
    with pytest.raises(NotImplementedError):
        resolve_binary_structure(ref, ["A1", "H3"])


def test_epoch_in_fit_params_raises():
    from jug.fitting.forward_delay import _assert_no_epoch_fit_params

    with pytest.raises(ValueError, match="DMEPOCH"):
        _assert_no_epoch_fit_params(["F0", "DMEPOCH"])
