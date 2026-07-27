"""TEMPORARY parity tests: deprecated NumPy residual path vs JAX path.

These tests will be removed once residual_delta_np is deleted and JAX is the
sole residual evaluator. They are already deprecated — do not extend the NumPy
path or add new production dependencies on it.

Remove this entire module when: all groups pass AND residual_delta_np is gone.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting.derivatives_astrometry import compute_astrometric_delay
from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.fitting.optimized_fitter import (
    GeneralFitSetup,
    _compute_full_model_residuals,
    _update_param,
)
from jug.io.astrometry_state import reconvert_ecliptic_to_equatorial
from jug.io.par_reader import parse_dec, parse_ra
from jug.utils.constants import K_DM_SEC, SECS_PER_DAY

from test_designmatrix_autodiff import (
    BINARY_CASES,
    BINARY_FIT,
    _binary_setup,
    _setup,
)
from test_jax_ecliptic_parity import _ecliptic_astrometry_setup

pytestmark = pytest.mark.jug_numpy_jax_parity_deprecated

PICOSECOND = 1.0e-12

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
        if backend.upper() in {"RAJ", "DECJ"} and isinstance(
            params.get(backend.upper()), str
        ):
            current = (
                parse_ra(params[backend.upper()])
                if backend.upper() == "RAJ"
                else parse_dec(params[backend.upper()])
            )
            _update_param(params, backend, current + float(delta[idx]))
        else:
            _update_param(
                params, backend, float(ref_theta[idx]) + float(delta[idx])
            )
    new_residuals, _, _, _ = _compute_full_model_residuals(params, setup)
    return np.asarray(new_residuals, dtype=float) - np.asarray(ref_residuals, dtype=float)


def _assert_jax_numpy_parity_deprecated(
    setup: GeneralFitSetup,
    fit_params: list[str],
    *,
    ref_params: dict | None = None,
    ref_theta: np.ndarray | None = None,
    step_sizes: dict[str, float] | None = None,
):
    ref_params = dict(ref_params or setup.params)
    if ref_theta is None:
        ref_theta = np.array(
            [float(ref_params.get(p, 0.0)) for p in fit_params], dtype=float
        )
    residual_fn = make_residual_delta_jax_fn(
        setup=setup,
        fit_params=fit_params,
        ref_params=ref_params,
        ref_theta=ref_theta,
    )
    default_steps = {
        "F0": 1.0e-8,
        "F1": 1.0e-16,
        "DM": 1.0e-4,
        "RAJ": 1.0e-8,
        "DECJ": 1.0e-8,
        "PMRA": 1.0e-6,
        "PMDEC": 1.0e-6,
        "PX": 1.0e-6,
        "ELONG": 1.0e-4,
        "ELAT": 1.0e-4,
        "LAMBDA": 1.0e-4,
        "BETA": 1.0e-4,
        "PMELONG": 1.0e-3,
        "PMELAT": 1.0e-3,
        "PMLAMBDA": 1.0e-3,
        "PMBETA": 1.0e-3,
    }
    step_sizes = step_sizes or {}
    delta_zero = np.zeros(len(fit_params), dtype=float)
    np.testing.assert_allclose(
        np.asarray(residual_fn(jnp.asarray(delta_zero))),
        _numpy_residual_delta(setup, fit_params, ref_params, ref_theta, delta_zero),
        atol=PICOSECOND,
    )
    for idx, name in enumerate(fit_params):
        step = step_sizes.get(name, default_steps.get(name, 1.0e-6))
        for sign in (-1.0, 1.0):
            delta = np.zeros(len(fit_params), dtype=float)
            delta[idx] = sign * step
            np.testing.assert_allclose(
                np.asarray(residual_fn(jnp.asarray(delta))),
                _numpy_residual_delta(
                    setup, fit_params, ref_params, ref_theta, delta
                ),
                atol=PICOSECOND,
                err_msg=f"{name} parity failed",
            )


def _equatorial_astrometry_setup():
    fit_params = ["RAJ", "DECJ", "PMRA", "PMDEC", "PX"]
    base = _setup(["F0"])
    n = len(base.tdb_mjd)
    params = dict(base.params)
    params.update(
        {
            "RAJ": 1.2,
            "DECJ": -0.5,
            "PMRA": 5.0,
            "PMDEC": -3.0,
            "PX": 1.0,
            "POSEPOCH": 55000.0,
            "_raj_rad": 1.2,
            "_decj_rad": -0.5,
        }
    )
    obs_pos = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    init_astro = np.asarray(
        compute_astrometric_delay(params, base.tdb_mjd, obs_pos), dtype=float
    )
    ref_theta = np.array([1.2, -0.5, 5.0, -3.0, 1.0], dtype=float)
    setup = dataclasses.replace(
        base,
        params=params,
        fit_param_list=fit_params,
        param_values_start=list(ref_theta),
        astrometry_params=fit_params,
        initial_astrometric_delay=init_astro,
        ssb_obs_pos_ls=obs_pos,
        dm_params=[],
        spin_params=["F0"],
        initial_dm_delay=None,
    )
    return setup, fit_params, ref_theta, params


# TEMPORARY: remove with residual_delta_np
def test_deprecated_parity_spin():
    fit_params = ["F0", "F1"]
    setup = _setup(fit_params)
    _assert_jax_numpy_parity_deprecated(setup, fit_params)


# TEMPORARY: remove with residual_delta_np
def test_deprecated_parity_dm():
    fit_params = ["DM"]
    setup = _setup(fit_params)
    _assert_jax_numpy_parity_deprecated(setup, fit_params)


# TEMPORARY: remove with residual_delta_np
def test_deprecated_parity_astrometry_equatorial():
    setup, fit_params, ref_theta, ref_params = _equatorial_astrometry_setup()
    _assert_jax_numpy_parity_deprecated(
        setup, fit_params, ref_params=ref_params, ref_theta=ref_theta
    )


# TEMPORARY: remove with residual_delta_np
@pytest.mark.parametrize("family", ["elong", "lambda"])
def test_deprecated_parity_astrometry_ecliptic(family):
    setup, fit_params, ref_theta, ref_params = _ecliptic_astrometry_setup(
        family=family
    )
    _assert_jax_numpy_parity_deprecated(
        setup, fit_params, ref_params=ref_params, ref_theta=ref_theta
    )


# TEMPORARY: remove with residual_delta_np
@pytest.mark.parametrize("case", list(BINARY_CASES))
def test_deprecated_parity_binary(case):
    setup, fit_params = _binary_setup(case)
    _assert_jax_numpy_parity_deprecated(setup, fit_params)
