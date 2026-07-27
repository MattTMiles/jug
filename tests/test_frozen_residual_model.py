"""Tests for the frozen residual-model export API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting import residual_model
from jug.fitting.residual_model import (
    FrozenResidualModel,
    NativeChainStatus,
    export_frozen_residual_model,
)


def _model(**overrides) -> FrozenResidualModel:
    kwargs = dict(
        fit_params=("F0",),
        param_mapping=(),
        reference_theta_native=np.array([100.0], dtype=np.float64),
        reference_residuals_sec=np.array([0.0], dtype=np.float64),
        subtract_tzr=True,
        compatibility="pint",
        mean_mode="weighted",
        row_tokens=("000000|56000|0.000000000000000",),
        _mean_weights=np.array([1.0], dtype=np.float64),
        _residual_delta_jax_fn=lambda delta: delta,
        _residual_jacobian_native_fn=lambda delta: jnp.ones((1, 1)),
        _native_chain_status=NativeChainStatus(False, False),
    )
    kwargs.update(overrides)
    for key in ("reference_theta_native", "reference_residuals_sec", "_mean_weights"):
        if kwargs[key] is not None:
            arr = np.array(kwargs[key], dtype=np.float64, copy=True)
            arr.setflags(write=False)
            kwargs[key] = arr
    return FrozenResidualModel(**kwargs)


def test_public_surface_reachable_via_jug_timing():
    import jug.timing

    assert jug.timing.FrozenResidualModel is FrozenResidualModel
    assert jug.timing.export_frozen_residual_model is export_frozen_residual_model


def test_export_requires_fit_params():
    with pytest.raises(ValueError, match="fit_params must be non-empty"):
        export_frozen_residual_model(object(), fit_params=())


def test_transform_linear_residual_contract():
    w = np.array([0.25, 0.75], dtype=np.float64)
    w.setflags(write=False)
    model = _model(
        row_tokens=("000000|1|0.0", "000001|1|0.1"),
        _mean_weights=w,
        mean_mode="weighted",
        compatibility="pint",
    )
    v = jnp.asarray([1.0, 3.0], dtype=jnp.float64)
    out = model.transform_linear_residual(v)
    expected = v - jnp.sum(v * jnp.asarray(w))
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))
    zeros = model.transform_linear_residual(jnp.zeros(2))
    np.testing.assert_allclose(np.asarray(zeros), 0.0)

    unweighted = _model(
        row_tokens=("000000|1|0.0", "000001|1|0.1"),
        _mean_weights=None,
        mean_mode="unweighted",
        compatibility="tempo2",
    )
    out_u = unweighted.transform_linear_residual(v)
    np.testing.assert_allclose(np.asarray(out_u), np.asarray(v - jnp.mean(v)))

    with pytest.raises(ValueError, match="1-D residual vector"):
        model.transform_linear_residual(jnp.ones((2, 2)))


def test_verify_native_chain_gates():
    pint = _model(compatibility="pint")
    pint.verify_native_chain()  # no-op

    missing = _model(
        compatibility="tempo2",
        _native_chain_status=NativeChainStatus(False, False),
    )
    with pytest.raises(RuntimeError, match="no native_chain_static"):
        missing.verify_native_chain()

    no_obs = _model(
        compatibility="tempo2",
        _native_chain_status=NativeChainStatus(True, False),
    )
    with pytest.raises(RuntimeError, match="tempo2_obs_state"):
        no_obs.verify_native_chain()

    ok = _model(
        compatibility="tempo2",
        _native_chain_status=NativeChainStatus(True, True),
    )
    ok.verify_native_chain()


def _write_dd_e_par(path, ecc=5e-4):
    path.write_text(
        "\n".join(
            [
                "PSRJ J0000+0000",
                "F0 200.0 1",
                "F1 -1e-15 1",
                "PEPOCH 56000",
                "POSEPOCH 56000",
                "DM 10.0 1",
                "DMEPOCH 56000",
                "RAJ 00:00:00",
                "DECJ +00:00:00",
                "UNITS TDB",
                "EPHEM DE440",
                "CLK TT(BIPM2021)",
                "BINARY DD",
                "PB 5.0 1",
                "T0 56000.0 1",
                "A1 10.0 1",
                "OM 45.0 1",
                f"E {ecc} 1",
                "M2 0.2 1",
                "SINI 0.9 1",
                "TZRMJD 56000",
                "TZRFRQ 1400",
                "TZRSITE ao",
                "",
            ]
        )
    )


def _write_minimal_tim(path, n=8):
    lines = ["FORMAT 1"]
    for i in range(n):
        lines.append(f"test 1400.0 {56000.0 + 0.1 * i} 1.0 ao")
    path.write_text("\n".join(lines) + "\n")


def test_export_runtime_fit_params_ecc_alias(tmp_path):
    from jug.engine.session import TimingSession

    par = tmp_path / "dd_e.par"
    tim = tmp_path / "dd_e.tim"
    _write_dd_e_par(par)
    _write_minimal_tim(tim)
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="pint")
    session.compute_residuals(subtract_tzr=True)
    state = export_frozen_residual_model(
        session,
        fit_params=["ECC"],
        param_mapping={"ECC": "E"},
    )
    assert ("ECC", "E") in state.param_mapping
    assert not state.reference_theta_native.flags.writeable
    assert not state.reference_residuals_sec.flags.writeable
    assert len(state.row_tokens) == len(session.toas_data)

    zeros = jnp.zeros(1, dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(state.residual_delta_jax(zeros)), 0.0, atol=1e-12)

    jac = state.residual_jacobian_native()
    jac_fwd = np.asarray(jax.jacfwd(state.residual_delta_jax)(zeros))
    np.testing.assert_allclose(jac, jac_fwd, rtol=1e-10, atol=1e-14)

    eps = 1e-7
    fd = (
        np.asarray(state.residual_delta_jax(jnp.asarray([eps])))
        - np.asarray(state.residual_delta_jax(jnp.asarray([-eps])))
    ) / (2 * eps)
    np.testing.assert_allclose(jac[:, 0], fd, rtol=2e-2, atol=1e-10)

    delta = np.array([1e-6])
    r_jax = np.asarray(state.residual_delta_jax(delta))
    assert np.max(np.abs(r_jax)) > 0


def test_sim_ddk_t0_residual_delta_float64():
    from pathlib import Path

    from jug.engine.session import TimingSession
    from tempo2_fixtures import get_tempo2_fixture

    try:
        fixture = get_tempo2_fixture("sim_ddk_tcb")
    except KeyError:
        pytest.skip("sim_ddk_tcb fixture not registered")
    par = Path(fixture["par_path"])
    tim = Path(fixture["tim_path"])
    if not par.exists() or not tim.exists():
        pytest.skip("sim_ddk_tcb data missing")
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="tempo2")
    session.compute_residuals(subtract_tzr=True)
    state = export_frozen_residual_model(session, fit_params=["T0"])
    zeros = np.zeros(1, dtype=np.float64)
    r_jax = np.asarray(state.residual_delta_jax(zeros), dtype=np.float64)
    assert r_jax.dtype == np.float64
    assert np.all(np.isfinite(r_jax))
    np.testing.assert_allclose(r_jax, 0.0, atol=5e-12)
    state.verify_native_chain()
