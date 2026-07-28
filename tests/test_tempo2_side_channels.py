"""Phase 0: tempo2 JAX autodiff side channels (FD, JUMP, NE_SW, …)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

import jax.numpy as jnp

from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.fitting.optimized_fitter import _build_general_fit_setup_from_files
from tempo2_fixtures import get_tempo2_fixture
from tempo2_test_helpers import residual_jacobian_fit_from_setup

pytestmark = [pytest.mark.tempo2]


def _tempo2_setup(fixture_id: str, fit_params: list[str], *, tempo2_native: str = "staged_bclt"):
    fixture = get_tempo2_fixture(fixture_id)
    return _build_general_fit_setup_from_files(
        fixture["par_path"],
        fixture["tim_path"],
        fit_params=fit_params,
        clock_dir=None,
        verbose=False,
        compatibility="tempo2",
        tempo2_native=tempo2_native,
    )


def _assert_column_nonzero(setup, fit_params: list[str], param: str):
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    matrix = residual_jacobian_fit_from_setup(
        setup, fit_params, delay_model="native"
    )
    col = matrix[:, fit_params.index(param)]
    assert np.all(np.isfinite(col)), f"{param} autodiff column is not finite"
    assert float(np.max(np.abs(col))) > 0.0, f"{param} autodiff column is all zero"


def _assert_perturbation_moves_residual(setup, param: str, eps: float):
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=[param])
    delta = np.asarray(fn(jnp.asarray([eps], dtype=jnp.float64)))
    assert float(np.max(np.abs(delta))) > 0.0, (
        f"{param} produced zero tempo2 residual_delta via side channel"
    )


@pytest.mark.parametrize("tempo2_native", ["staged_bclt", "fixed_state_bclt"])
def test_tempo2_fd1_autodiff_column_nonzero(tempo2_native):
    setup = _tempo2_setup("ppta_j1902_ell1h", ["FD1"], tempo2_native=tempo2_native)
    _assert_column_nonzero(setup, ["FD1"], "FD1")
    _assert_perturbation_moves_residual(setup, "FD1", eps=1e-8)


@pytest.mark.parametrize("tempo2_native", ["staged_bclt", "fixed_state_bclt"])
def test_tempo2_jump_autodiff_column_nonzero(tempo2_native):
    # JUMP1 has no matching TOAs on this fixture; JUMP2 (-medusa_58925_jump) does.
    setup = _tempo2_setup("ppta_j1902_ell1h", ["JUMP2"], tempo2_native=tempo2_native)
    if "JUMP2" not in (setup.fit_param_list or []):
        pytest.skip("JUMP2 dropped during setup (empty mask)")
    _assert_column_nonzero(setup, ["JUMP2"], "JUMP2")
    _assert_perturbation_moves_residual(setup, "JUMP2", eps=1e-9)


@pytest.mark.parametrize("tempo2_native", ["staged_bclt", "fixed_state_bclt"])
def test_tempo2_ne_sw_autodiff_column_nonzero(tempo2_native):
    setup = _tempo2_setup("epta_j0030_isolated", ["NE_SW"], tempo2_native=tempo2_native)
    if setup.sw_geometry_pc is None:
        pytest.skip("sw_geometry_pc unavailable on fixture")
    _assert_column_nonzero(setup, ["NE_SW"], "NE_SW")
    _assert_perturbation_moves_residual(setup, "NE_SW", eps=1e-4)


def test_tempo2_side_channels_use_modular_path_not_pint_total(monkeypatch):
    """Tempo2 autodiff must call side channels, not full PINT delay routing."""
    setup = _tempo2_setup("ppta_j1902_ell1h", ["F0", "FD1"])
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")

    calls = {"bbat": 0, "side": 0, "pint_total": 0}

    def _pint_total(*args, **kwargs):
        calls["pint_total"] += 1
        raise AssertionError("compute_total_delay_change must not run on tempo2 path")

    def _bbat(*args, **kwargs):
        calls["bbat"] += 1
        from jug.residuals.tempo2.terms import compute_bbat_delay_change_sec_jax

        return compute_bbat_delay_change_sec_jax(*args, **kwargs)

    def _side(*args, **kwargs):
        calls["side"] += 1
        from jug.fitting.forward_delay import compute_side_delay_change

        return compute_side_delay_change(*args, **kwargs)

    monkeypatch.setattr(
        "jug.fitting.forward_delay.compute_total_delay_change",
        _pint_total,
    )
    monkeypatch.setattr(
        "jug.fitting.jax_residual_delta.compute_bbat_delay_change_sec_jax",
        _bbat,
    )
    monkeypatch.setattr(
        "jug.fitting.jax_residual_delta.compute_side_delay_change",
        _side,
    )

    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0", "FD1"])
    fn(jnp.zeros(2, dtype=jnp.float64))
    assert calls["bbat"] == 1
    assert calls["side"] == 1
    assert calls["pint_total"] == 0