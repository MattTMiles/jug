"""Tests for the frozen residual-model export API."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jug.fitting.residual_model import (
    FrozenResidualModel,
    NativeChainStatus,
    export_frozen_residual_model,
)
from jug.residuals.gauge import ReferenceGauge


def _model(**overrides) -> FrozenResidualModel:
    kwargs = dict(
        fit_params=("F0",),
        param_mapping=(),
        reference_theta_native=np.array([100.0], dtype=np.float64),
        reference_residuals_sec=np.array([0.0], dtype=np.float64),
        subtract_tzr=True,
        compatibility="pint",
        reference_gauge=ReferenceGauge(mode="mean", weights=np.array([1.0])),
        row_tokens=("000000|56000|0.000000000000000",),
        _residual_delta_jax_fn=lambda delta: delta,
        _residual_jacobian_native_fn=lambda delta: jnp.ones((1, 1)),
        _native_chain_status=NativeChainStatus(False, False),
    )
    kwargs.update(overrides)
    for key in ("reference_theta_native", "reference_residuals_sec"):
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


def test_centering_transform_api_is_gone():
    model = _model()
    assert not hasattr(model, "mean_mode")
    assert not hasattr(model, "_mean_weights")
    assert not hasattr(model, "transform_linear_residual")
    assert isinstance(model.reference_gauge, ReferenceGauge)
    assert model.reference_gauge.mode == "mean"


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
        reference_gauge=ReferenceGauge(mode="mean", weights=None),
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
    assert state.reference_gauge.mode == "mean"
    assert state.reference_gauge.weights is not None
    assert not state.reference_gauge.weights.flags.writeable

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


def test_residual_jacobian_is_gauge_free_graph(tmp_path):
    from jug.engine.session import TimingSession

    par = tmp_path / "dd_e.par"
    tim = tmp_path / "dd_e.tim"
    _write_dd_e_par(par)
    _write_minimal_tim(tim)
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="pint")
    session.compute_residuals(subtract_tzr=True)
    state = export_frozen_residual_model(session, fit_params=["F0", "DM"])
    jac = state.residual_jacobian_native()
    # Gauge-free Jacobian columns need not be mean-zero.
    col_means = np.mean(jac, axis=0)
    assert np.any(np.abs(col_means) > 1e-20)


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
    assert state.reference_gauge.mode == "mean"
    assert state.reference_gauge.weights is None  # tempo2 → unweighted
    delta = np.array([1e-8], dtype=np.float64)
    r = np.asarray(state.residual_delta_jax(delta), dtype=np.float64)
    assert r.dtype == np.float64
    assert np.max(np.abs(r)) > 0


def test_tempo2_reference_gauge_matches_host_unweighted_mean(tmp_path):
    """reference_gauge must describe the gauge actually applied to reference_residuals_sec."""
    from jug.engine.session import TimingSession

    par = tmp_path / "t2.par"
    tim = tmp_path / "t2.tim"
    par.write_text(
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
                "TZRMJD 56000",
                "TZRFRQ 1400",
                "TZRSITE ao",
                "",
            ]
        )
    )
    # Non-uniform TOA errors so weighted vs unweighted means disagree.
    lines = ["FORMAT 1"]
    errors = [0.1, 0.1, 5.0, 5.0, 0.2, 0.2, 8.0, 0.15]
    for i, err in enumerate(errors):
        lines.append(f"test 1400.0 {56000.0 + 0.1 * i} {err} ao")
    tim.write_text("\n".join(lines) + "\n")

    session = TimingSession(str(par), str(tim), verbose=False, compatibility="tempo2")
    session.compute_residuals(subtract_tzr=True)
    state = export_frozen_residual_model(session, fit_params=["F0", "DM"])
    assert state.reference_gauge.mode == "mean"
    assert state.reference_gauge.weights is None
    ref = np.asarray(state.reference_residuals_sec, dtype=np.float64)
    assert abs(float(np.mean(ref))) < 1e-14


def test_tempo2_refphs_tzr_records_constant_gauge_and_refuses_abs(tmp_path):
    from jug.engine.session import TimingSession

    par = tmp_path / "tzr.par"
    tim = tmp_path / "tzr.tim"
    par.write_text(
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
                "TZRMJD 56000",
                "TZRFRQ 1400",
                "TZRSITE ao",
                "REFPHS TZR",
                "",
            ]
        )
    )
    tim.write_text(
        "FORMAT 1\n" + "\n".join(f"test 1400.0 {56000.0 + 0.1 * i} 1.0 ao" for i in range(8)) + "\n"
    )
    session = TimingSession(str(par), str(tim), verbose=False, compatibility="tempo2")
    session.compute_residuals(subtract_tzr=True)
    cached = session._cached_result_by_mode[True]
    assert cached.get("tzr_apply_mode") == "post_wrap"
    assert cached.get("tzr_residual_sec") is not None

    state = export_frozen_residual_model(session, fit_params=["F0", "DM"])
    assert state.reference_gauge.mode == "constant"
    assert state.reference_gauge.offset_sec == pytest.approx(float(cached["tzr_residual_sec"]))
    with pytest.raises(NotImplementedError, match="REFPHS TZR"):
        state.absolute_residuals_sec(np.array([1e-8, 1e-3], dtype=np.float64))
