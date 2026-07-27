"""ECC/E alias parity for DD autodiff and residual deltas (Fix J2)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jug.fitting.binary_delay_plan import resolve_binary_structure
from jug.fitting.binary_registry import compute_binary_delay
from jug.fitting.jax_residual_delta import (
    _simplified_residual_jacobian_oracle,
    make_residual_delta_jax_fn,
)
from jug.utils.constants import SECS_PER_DAY


def _dd_params(ecc_key: str, ecc: float):
    params = {
        "BINARY": "DD",
        "A1": 10.0,
        "PB": 5.0,
        "T0": 55000.0,
        "OM": 45.0,
        "M2": 0.2,
        "SINI": 0.9,
        ecc_key: ecc,
    }
    return params


@pytest.mark.parametrize("ecc", [5e-4, 0.44])
def test_ecc_vs_e_plan_delay_and_column(ecc):
    from test_designmatrix_autodiff import _setup

    p_ecc = _dd_params("ECC", ecc)
    p_e = _dd_params("E", ecc)
    plan_ecc = resolve_binary_structure(p_ecc, ["ECC"])
    plan_e = resolve_binary_structure(p_e, ["ECC"])
    toas = jnp.linspace(55000.0, 55002.0, 24)
    d_ecc = np.asarray(plan_ecc.evaluate(toas, p_ecc, None, jnp))
    d_e = np.asarray(plan_e.evaluate(toas, p_e, None, jnp))
    np.testing.assert_allclose(d_ecc, d_e, rtol=0, atol=0)

    def delay_ecc(e):
        return plan_ecc.evaluate(toas, {**p_ecc, "ECC": e}, None, jnp)

    def delay_e(e):
        return plan_e.evaluate(toas, {**p_e, "ECC": e}, None, jnp)

    col_ecc = np.asarray(jax.jacfwd(delay_ecc)(ecc))
    col_e = np.asarray(jax.jacfwd(delay_e)(ecc))
    assert np.max(np.abs(col_ecc)) > 0
    np.testing.assert_allclose(col_ecc, col_e, rtol=0, atol=0)

    # Residual-delta path: same eccentricity perturbation under both spellings.
    fit_params = ["ECC"]
    base = _setup(["F0"])
    n = len(base.tdb_mjd)
    obs = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    prebinary = np.zeros(n, dtype=float)
    toas_pre = base.tdb_mjd - prebinary / SECS_PER_DAY

    def _mk(params):
        init = np.asarray(
            compute_binary_delay(toas_pre, params, obs_pos_ls=obs), dtype=float
        )
        setup = dataclasses.replace(
            base,
            params=dict(params, F0=200.0, PEPOCH=55000.0),
            fit_param_list=fit_params,
            param_values_start=[ecc],
            binary_params=["ECC"],
            prebinary_delay_sec=prebinary,
            initial_binary_delay=init,
            ssb_obs_pos_ls=obs,
            dm_params=[],
            spin_params=[],
            initial_dm_delay=None,
            binary_plan=None,
        )
        return setup

    setup_ecc = _mk(p_ecc)
    setup_e = _mk({**p_e, "ECC": ecc})  # canonical copy present for residual writes
    # Keep raw E as well for alias readers.
    setup_e = dataclasses.replace(
        setup_e, params={**setup_e.params, "E": ecc, "ECC": ecc}
    )
    fn_ecc = make_residual_delta_jax_fn(setup=setup_ecc, fit_params=fit_params)
    fn_e = make_residual_delta_jax_fn(setup=setup_e, fit_params=fit_params)
    delta = jnp.array([1e-6])
    r_ecc = np.asarray(fn_ecc(delta))
    r_e = np.asarray(fn_e(delta))
    np.testing.assert_allclose(r_ecc, r_e, atol=1e-12)
    assert np.max(np.abs(r_ecc)) > 0

    mtx_ecc = _simplified_residual_jacobian_oracle(setup_ecc, fit_params)
    mtx_e = _simplified_residual_jacobian_oracle(setup_e, fit_params)
    assert np.linalg.norm(mtx_ecc[:, 0]) > 0
    np.testing.assert_allclose(mtx_ecc, mtx_e, rtol=1e-10, atol=1e-14)
