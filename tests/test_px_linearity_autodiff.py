"""PX delay linearity at/through zero (Fix J1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from jug.fitting.derivatives_astrometry import (
    compute_astrometric_delay,
    d_delay_d_PX,
)
from jug.fitting.derivatives_dd import (
    _as_f64,
    _compute_kopeikin_corrections_traceable,
    resolve_kopeikin_flags,
)
from jug.fitting.jax_residual_delta import compute_autodiff_designmatrix_from_setup


MAS_TO_RAD = np.pi / 180.0 / 3600.0 / 1000.0


def test_astrometric_px_column_nonzero_at_zero():
    n = 16
    toas = jnp.linspace(55000.0, 55100.0, n)
    ssb = jnp.ones((n, 3)) * 100.0
    base = {
        "RAJ": 1.0,
        "DECJ": 0.5,
        "PMRA": 0.0,
        "PMDEC": 0.0,
        "PX": 0.0,
        "POSEPOCH": 55000.0,
    }

    def delay_vec(px):
        return compute_astrometric_delay({**base, "PX": px}, toas, ssb)

    analytic = np.asarray(
        d_delay_d_PX(
            base["RAJ"],
            base["DECJ"],
            ssb,
            toas_mjd=np.asarray(toas),
            posepoch_mjd=base["POSEPOCH"],
        )
    ) * MAS_TO_RAD

    cols = []
    for px in (-1.0, 0.0, 1.0):
        col = np.asarray(jax.jacfwd(delay_vec)(px))
        assert np.all(np.isfinite(col))
        assert np.linalg.norm(col) > 0.0
        np.testing.assert_allclose(col, analytic, rtol=1e-10, atol=1e-15)
        cols.append(col)
    np.testing.assert_allclose(cols[0], cols[1], rtol=0, atol=1e-15)
    np.testing.assert_allclose(cols[2], cols[1], rtol=0, atol=1e-15)


def test_autodiff_design_matrix_px_zero_nonzero_column():
    from test_designmatrix_autodiff import _setup

    fit_params = ["PX"]
    setup = _setup(["F0"], method="autodiff")
    n = len(setup.tdb_mjd)
    params = dict(setup.params)
    params.update(
        {
            "RAJ": 1.0,
            "DECJ": 0.5,
            "PMRA": 0.0,
            "PMDEC": 0.0,
            "PX": 0.0,
            "POSEPOCH": 55000.0,
        }
    )
    obs = 1e-3 * (1.0 + np.arange(3 * n, dtype=float)).reshape(n, 3)
    setup = dataclasses.replace(
        setup,
        params=params,
        fit_param_list=fit_params,
        param_values_start=[0.0],
        astrometry_params=["PX"],
        ssb_obs_pos_ls=obs,
        initial_astrometric_delay=np.zeros(n),
        dm_params=[],
        spin_params=[],
        initial_dm_delay=None,
    )
    matrix = compute_autodiff_designmatrix_from_setup(setup, fit_params)
    assert np.linalg.norm(matrix[:, fit_params.index("PX")]) > 0.0


@pytest.mark.parametrize("px_ref", [-1.0, 0.0, 1.0])
def test_kopeikin_px_linear_signed_through_zero(px_ref):
    """has_parallax is structural (KIN), not value-gated on reference PX."""
    params = {
        "BINARY": "DDK",
        "A1": 10.0,
        "T0": 55000.0,
        "KIN": 60.0,
        "KOM": 30.0,
        "PX": px_ref,
        "RAJ": 1.0,
        "DECJ": -0.5,
        "PMRA": 0.0,
        "PMDEC": 0.0,
    }
    struct = resolve_kopeikin_flags(params)
    assert struct.has_parallax is True
    toas = jnp.linspace(55000.0, 55100.0, 8)
    obs = jnp.ones((8, 3))

    def a1_corr(px):
        d_a1, d_om, _ = _compute_kopeikin_corrections_traceable(
            toas, 10.0, 55000.0, 60.0, 30.0, px, 0.0, 0.0, obs, struct
        )
        return d_a1.sum()

    def om_corr(px):
        d_a1, d_om, _ = _compute_kopeikin_corrections_traceable(
            toas, 10.0, 55000.0, 60.0, 30.0, px, 0.0, 0.0, obs, struct
        )
        return d_om.sum()

    for getter in (
        lambda px: _compute_kopeikin_corrections_traceable(
            toas, 10.0, 55000.0, 60.0, 30.0, px, 0.0, 0.0, obs, struct
        )[0],
        lambda px: _compute_kopeikin_corrections_traceable(
            toas, 10.0, 55000.0, 60.0, 30.0, px, 0.0, 0.0, obs, struct
        )[1],
    ):
        vals = {px: np.asarray(getter(px)) for px in (-1.0, 0.0, 1.0)}
        np.testing.assert_allclose(vals[0.0], 0.0, atol=0)
        np.testing.assert_allclose(vals[-1.0], -vals[1.0], rtol=0, atol=1e-15)

    for fn in (a1_corr, om_corr):
        grads = [float(jax.jacfwd(fn)(px)) for px in (-1.0, 0.0, 1.0)]
        assert all(np.isfinite(g) and abs(g) > 0 for g in grads)
        np.testing.assert_allclose(grads[0], grads[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(grads[2], grads[1], rtol=0, atol=1e-12)


def test_kopeikin_has_parallax_false_without_kin():
    struct = resolve_kopeikin_flags(
        {
            "BINARY": "DDK",
            "A1": 10.0,
            "T0": 55000.0,
            "KIN": 0.0,
            "KOM": 30.0,
            "PX": 1.0,
            "RAJ": 1.0,
            "DECJ": -0.5,
        }
    )
    assert struct.has_parallax is False


def test_kopeikin_longdouble_inputs_match_float64():
    params = {
        "BINARY": "DDK",
        "A1": 10.0,
        "T0": 55000.0,
        "KIN": 60.0,
        "KOM": 30.0,
        "PX": 1.0,
        "RAJ": 1.0,
        "DECJ": -0.5,
        "PMRA": 0.0,
        "PMDEC": 0.0,
    }
    struct = resolve_kopeikin_flags(params)
    toas = jnp.linspace(55000.0, 55100.0, 8)
    obs = jnp.ones((8, 3))
    out_f64 = _compute_kopeikin_corrections_traceable(
        toas, 10.0, 55000.0, 60.0, 30.0, 1.0, 0.0, 0.0, obs, struct
    )
    out_ld = _compute_kopeikin_corrections_traceable(
        toas,
        np.longdouble(10.0),
        np.longdouble(55000.0),
        np.longdouble(60.0),
        np.longdouble(30.0),
        np.longdouble(1.0),
        np.longdouble(0.0),
        np.longdouble(0.0),
        obs,
        struct,
    )
    for a, b in zip(out_f64, out_ld):
        arr = np.asarray(b)
        assert arr.dtype == np.float64 or jnp.asarray(b).dtype == jnp.float64
        assert np.all(np.isfinite(arr))
        np.testing.assert_allclose(np.asarray(a), arr, rtol=0, atol=0)
    # Sanitizer itself accepts longdouble.
    assert float(_as_f64(np.longdouble("55000.123456789012345"))) == pytest.approx(
        float(np.longdouble("55000.123456789012345"))
    )
