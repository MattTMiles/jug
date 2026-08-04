"""Ecliptic-parameter JAX autodiff parity regression tests."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting.derivatives_astrometry import compute_astrometric_delay
from jug.fitting.jax_residual_delta import (
    _simplified_residual_jacobian_oracle,
    make_residual_delta_jax_fn,
)
from jug.utils.units import native_derivative_to_fit_column
from jug.fitting.optimized_fitter import (
    GeneralFitSetup,
    _compute_full_model_residuals,
    _update_param,
)
from jug.io.astrometry_state import reconvert_ecliptic_to_equatorial
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY

PICOSECOND = 1.0e-12


def _base_setup(*, fit_params: list[str]) -> GeneralFitSetup:
    tdb_mjd = np.array([55000.0, 55000.25, 55000.5, 55000.75, 55001.0], dtype=float)
    freq_mhz = np.array([820.0, 900.0, 1100.0, 1400.0, 1600.0], dtype=float)
    errors_us = np.full(len(tdb_mjd), 1.0, dtype=float)
    params = {"F0": 200.0, "PEPOCH": 55000.0}
    dt_sec = (tdb_mjd - params["PEPOCH"]) * SECS_PER_DAY
    return GeneralFitSetup(
        params=params,
        fit_param_list=list(fit_params),
        compatibility="pint",
        fd_column_mode="delay_only",
        param_values_start=[float(params.get(p, 0.0)) for p in fit_params],
        toas_mjd=tdb_mjd,
        freq_mhz=freq_mhz,
        errors_us=errors_us,
        errors_sec=errors_us * 1.0e-6,
        weights=1.0 / (errors_us * 1.0e-6) ** 2,
        dt_sec_cached=dt_sec,
        dt_sec_ld=np.asarray(dt_sec, dtype=np.longdouble),
        tdb_mjd=tdb_mjd,
        initial_dm_delay=None,
        dm_params=[],
        spin_params=["F0"],
        binary_params=[],
        astrometry_params=["ELONG"],
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


def _ecliptic_params(*, family: str) -> dict:
    base = {
        "_ecliptic_coords": True,
        "_ecliptic_frame": "IERS2010",
        "_ecliptic_lon_deg": 120.0,
        "_ecliptic_lat_deg": -30.0,
        "_ecliptic_pm_lon": 5.0,
        "_ecliptic_pm_lat": -3.0,
        "PX": 1.0,
        "POSEPOCH": 55000.0,
    }
    if family == "lambda":
        base.update(
            {
                "LAMBDA": 120.0,
                "BETA": -30.0,
                "PMLAMBDA": 5.0,
                "PMBETA": -3.0,
            }
        )
    else:
        base.update(
            {
                "ELONG": 120.0,
                "ELAT": -30.0,
                "PMELONG": 5.0,
                "PMELAT": -3.0,
            }
        )
    reconvert_ecliptic_to_equatorial(base)
    return base


def _ecliptic_astrometry_setup(*, family: str):
    if family == "lambda":
        fit_params = ["LAMBDA", "BETA", "PMLAMBDA", "PMBETA"]
        ref_theta = np.array([120.0, -30.0, 5.0, -3.0], dtype=float)
    else:
        fit_params = ["ELONG", "ELAT", "PMELONG", "PMELAT"]
        ref_theta = np.array([120.0, -30.0, 5.0, -3.0], dtype=float)

    base = _base_setup(fit_params=fit_params)
    params = dict(base.params)
    params.update(_ecliptic_params(family=family))
    n = len(base.tdb_mjd)
    obs_pos = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    init_astro = np.asarray(
        compute_astrometric_delay(params, base.tdb_mjd, obs_pos), dtype=float
    )
    setup = dataclasses.replace(
        base,
        params=params,
        fit_param_list=fit_params,
        param_values_start=list(ref_theta),
        initial_astrometric_delay=init_astro,
        ssb_obs_pos_ls=obs_pos,
    )
    return setup, fit_params, ref_theta, dict(params)


_LAMBDA_TO_ELONG = {
    "LAMBDA": "ELONG",
    "BETA": "ELAT",
    "PMLAMBDA": "PMELONG",
    "PMBETA": "PMELAT",
}


def _numpy_residual_delta(setup, fit_params, ref_params, ref_theta, delta):
    ref_residuals, _, _, _ = _compute_full_model_residuals(ref_params, setup)
    params = dict(ref_params)
    for idx, name in enumerate(fit_params):
        backend = _LAMBDA_TO_ELONG.get(name.upper(), name.upper())
        _update_param(params, backend, float(ref_theta[idx]) + float(delta[idx]))
    new_residuals, _, _, _ = _compute_full_model_residuals(params, setup)
    return np.asarray(new_residuals, dtype=float) - np.asarray(ref_residuals, dtype=float)


@pytest.mark.parametrize("family", ["elong", "lambda"])
def test_oracle_ecliptic_columns_nonzero(family):
    setup, fit_params, _, _ = _ecliptic_astrometry_setup(family=family)
    matrix = _simplified_residual_jacobian_oracle(setup, fit_params)
    for idx, name in enumerate(fit_params):
        norm = np.linalg.norm(matrix[:, idx])
        assert norm > 0.0, f"{name} residual-Jacobian column is all-zero"


@pytest.mark.parametrize("family", ["elong", "lambda"])
def test_jax_numpy_residual_delta_ecliptic_parity(family):
    setup, fit_params, ref_theta, ref_params = _ecliptic_astrometry_setup(family=family)
    residual_fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
    )
    steps = {
        "ELONG": 1.0e-4,
        "LAMBDA": 1.0e-4,
        "ELAT": 1.0e-4,
        "BETA": 1.0e-4,
        "PMELONG": 1.0e-3,
        "PMLAMBDA": 1.0e-3,
        "PMELAT": 1.0e-3,
        "PMBETA": 1.0e-3,
    }
    for idx, name in enumerate(fit_params):
        for sign in (-1.0, 1.0):
            delta = np.zeros(len(fit_params), dtype=float)
            delta[idx] = sign * steps[name]
            np_delta = _numpy_residual_delta(
                setup, fit_params, ref_params, ref_theta, delta
            )
            jax_delta = np.asarray(residual_fn(jnp.asarray(delta)))
            np.testing.assert_allclose(
                jax_delta,
                np_delta,
                atol=PICOSECOND,
                err_msg=f"{name} residual-delta parity failed",
            )


@pytest.mark.parametrize("family", ["elong"])
def test_jacfwd_matches_numpy_fd_ecliptic(family):
    setup, fit_params, ref_theta, ref_params = _ecliptic_astrometry_setup(family=family)
    residual_fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
    )
    jac_jax = np.asarray(jax.jacfwd(residual_fn)(jnp.zeros(len(fit_params))))
    steps = {
        "ELONG": 1.0e-4,
        "ELAT": 1.0e-4,
        "PMELONG": 1.0e-3,
        "PMELAT": 1.0e-3,
    }
    for idx, name in enumerate(fit_params):
        h = steps[name]
        plus = np.zeros(len(fit_params))
        minus = np.zeros(len(fit_params))
        plus[idx] = h
        minus[idx] = -h
        col_np = (
            _numpy_residual_delta(setup, fit_params, ref_params, ref_theta, plus)
            - _numpy_residual_delta(setup, fit_params, ref_params, ref_theta, minus)
        ) / (2.0 * h)
        col_jax = jac_jax[:, idx]
        atol = max(PICOSECOND / h, 1.0e-6 * np.max(np.abs(col_np)))
        np.testing.assert_allclose(col_jax, col_np, atol=atol, err_msg=f"{name} Jacobian")


@pytest.mark.parametrize("family", ["elong", "lambda"])
def test_oracle_matches_jacfwd_ecliptic(family):
    """Oracle J_fit columns must match jacfwd in fit units.

    Guards BUG 002: LAMBDA/BETA must not inherit RAJ/DECJ hourangle/radian
    scales when exporting residual-Jacobian columns.
    """
    setup, fit_params, ref_theta, ref_params = _ecliptic_astrometry_setup(family=family)
    matrix = _simplified_residual_jacobian_oracle(setup, fit_params)
    residual_fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
    )
    jac = np.asarray(jax.jacfwd(residual_fn)(jnp.zeros(len(fit_params))))
    for idx, name in enumerate(fit_params):
        exported = np.asarray(matrix[:, idx], dtype=float)
        raw = np.asarray(
            native_derivative_to_fit_column(name, jac[:, idx]), dtype=float
        )
        exported_ms = exported - np.mean(exported)
        raw_ms = raw - np.mean(raw)
        denom = np.linalg.norm(exported_ms)
        assert denom > 0.0, f"{name}: exported column vanished"
        ratio = np.linalg.norm(raw_ms) / denom
        np.testing.assert_allclose(
            ratio,
            1.0,
            rtol=1.0e-8,
            atol=1.0e-8,
            err_msg=f"{name}: J_fit/jac scale mismatch (got {ratio})",
        )
        np.testing.assert_allclose(
            exported,
            raw,
            rtol=1.0e-10,
            atol=1.0e-15,
            err_msg=f"{name}: oracle J_fit != jacfwd(fit units)",
        )


def test_analytic_derivatives_accept_lambda_aliases():
    from jug.fitting.derivatives_astrometry import compute_astrometry_derivatives

    n = 3
    params = {
        "_ecliptic_coords": True,
        "_ecliptic_frame": "IERS2010",
        "_ecliptic_lon_deg": 120.0,
        "_ecliptic_lat_deg": -30.0,
        "_ecliptic_pm_lon": 5.0,
        "_ecliptic_pm_lat": -3.0,
        "RAJ": 1.2,
        "DECJ": -0.5,
        "_raj_rad": 1.2,
        "_decj_rad": -0.5,
        "PMRA": 5.0,
        "PMDEC": -3.0,
        "POSEPOCH": 55000.0,
    }
    toas = np.array([55000.0, 55000.5, 55001.0], dtype=float)
    obs_pos = 1e-3 * np.ones((n, 3), dtype=float)
    derivs = compute_astrometry_derivatives(
        params, toas, obs_pos, ["LAMBDA", "BETA", "PMLAMBDA", "PMBETA"]
    )
    for name in ("LAMBDA", "BETA", "PMLAMBDA", "PMBETA"):
        assert name in derivs
        assert np.all(np.isfinite(derivs[name]))
        assert np.linalg.norm(np.asarray(derivs[name])) > 0.0


def test_jax_ecliptic_transform_matches_numpy_at_reference():
    from jug.fitting.derivatives_astrometry import ecliptic_deg_to_equatorial_rad
    from jug.io.par_reader import OBLIQUITY_ARCSEC

    params = _ecliptic_params(family="elong")
    frame = str(params["_ecliptic_frame"]).upper()
    obl_rad = float(
        OBLIQUITY_ARCSEC[frame] * np.pi / (180.0 * 3600.0)
    )
    ra_jax, dec_jax, pmra_jax, pmdec_jax = ecliptic_deg_to_equatorial_rad(
        params["_ecliptic_lon_deg"],
        params["_ecliptic_lat_deg"],
        params["_ecliptic_pm_lon"],
        params["_ecliptic_pm_lat"],
        obl_rad,
        xp=jnp,
    )
    np.testing.assert_allclose(float(ra_jax), params["_raj_rad"], rtol=0, atol=1.0e-15)
    np.testing.assert_allclose(float(dec_jax), params["_decj_rad"], rtol=0, atol=1.0e-15)
    np.testing.assert_allclose(float(pmra_jax), params["PMRA"], rtol=0, atol=1.0e-12)
    np.testing.assert_allclose(float(pmdec_jax), params["PMDEC"], rtol=0, atol=1.0e-12)


def test_jax_ecliptic_sync_preserves_pmra_when_ecliptic_pm_zero():
    """When ecliptic PM is zero, JAX sync must not overwrite existing PMRA/PMDEC."""
    from jug.fitting.jax_residual_delta import _build_params_from_delta
    from jug.io.par_reader import OBLIQUITY_ARCSEC

    ref_params = _ecliptic_params(family="elong")
    ref_params["_ecliptic_pm_lon"] = 0.0
    ref_params["_ecliptic_pm_lat"] = 0.0
    ref_params["PMELONG"] = 0.0
    ref_params["PMELAT"] = 0.0
    ref_params["PMRA"] = 12.34
    ref_params["PMDEC"] = -5.67
    reconvert_ecliptic_to_equatorial(ref_params)
    assert ref_params.get("PMRA") == 12.34
    assert ref_params.get("PMDEC") == -5.67

    obl_rad = float(OBLIQUITY_ARCSEC["IERS2010"] * np.pi / (180.0 * 3600.0))
    synced = _build_params_from_delta(
        ref_params,
        ["ELONG"],
        np.array([ref_params["_ecliptic_lon_deg"]], dtype=float),
        jnp.array([1.0e-4], dtype=float),
        ecliptic_coords=True,
        obl_rad=obl_rad,
        ecliptic_init=(
            ref_params["_ecliptic_lon_deg"],
            ref_params["_ecliptic_lat_deg"],
            0.0,
            0.0,
        ),
        native_family="elong",
    )
    assert float(synced["PMRA"]) == 12.34
    assert float(synced["PMDEC"]) == -5.67
