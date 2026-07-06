"""DEV ORACLE — native residual_delta uses full JAX chain, not Taylor fallback."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax
import jax.numpy as jnp

from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.fitting.optimized_fitter import _build_general_fit_setup_from_files
from tempo2_native_test_helpers import load_wsrt167_fixture


@pytest.fixture
def wsrt167_setup(monkeypatch):
    monkeypatch.setattr(
        "jug.residuals.tempo2_native_quarantine.USE_JAX_TEMPO2_NATIVE_CHAIN",
        True,
    )
    monkeypatch.setattr(
        "jug.fitting.optimized_fitter.USE_JAX_TEMPO2_NATIVE_CHAIN",
        True,
    )
    fixture = load_wsrt167_fixture()
    return _build_general_fit_setup_from_files(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=["F0"],
        clock_dir=None,
        verbose=False,
        compatibility="tempo2",
        design_matrix_method="autodiff",
    )


def test_native_residual_delta_uses_full_chain_not_taylor(wsrt167_setup, monkeypatch):
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable (USE_JAX_TEMPO2_NATIVE_CHAIN off?)")
    if setup.native_tempo2_terms is None:
        pytest.skip("native_tempo2_terms unavailable")

    calls = {"taylor": 0}

    def _taylor(*args, **kwargs):
        calls["taylor"] += 1
        raise AssertionError("_phase_residual_delta_jax must not run in native mode")

    monkeypatch.setattr(
        "jug.fitting.jax_residual_delta._phase_residual_delta_jax",
        _taylor,
    )
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0"])
    delta = fn(jnp.zeros(1, dtype=jnp.float64))
    assert delta.shape[0] == setup.toas_mjd.shape[0]
    assert calls["taylor"] == 0


def test_native_f0_jacfwd_finite_difference_spot_check(wsrt167_setup):
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0"])
    jac = np.asarray(jax.jacfwd(fn)(jnp.zeros(1, dtype=jnp.float64)), dtype=np.float64).reshape(-1)
    eps = 1e-8
    fd = (np.asarray(fn(jnp.asarray([eps]))) - np.asarray(fn(jnp.asarray([-eps])))) / (2 * eps)
    fd = fd.reshape(-1)
    scale = max(float(np.max(np.abs(fd))), 1.0)
    assert float(np.max(np.abs(jac - fd))) / scale < 0.05


@pytest.mark.parametrize(
    "param,eps",
    [
        ("RAJ", 1e-10),
        ("DECJ", 1e-10),
        ("PX", 1e-4),
        ("DM", 1e-5),
    ],
)
def test_native_delta_recomputes_delay_terms(wsrt167_setup, param, eps):
    """Phase 5 gate: delay terms must move when astrometry/DM are perturbed."""
    setup = wsrt167_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=[param])
    delta = np.asarray(fn(jnp.asarray([eps], dtype=jnp.float64)))
    assert np.max(np.abs(delta)) > 0.0, (
        f"{param} produced zero native residual_delta; delay terms are frozen"
    )
