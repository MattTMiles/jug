"""Tests for the frozen JAX timing-state export API."""

import numpy as np
import pytest

from jug.fitting import jax_timing_state
from jug.fitting.jax_timing_state import JaxTimingState
from jug.io.par_reader import get_longdouble


def _state(**overrides) -> JaxTimingState:
    kwargs = dict(
        fit_params=("F0",),
        param_mapping=(),
        runtime_fit_params=("F0",),
        ref_params={"F0": 100.0},
        ref_theta=np.array([100.0]),
        reference_residuals_sec=np.array([0.0]),
        subtract_tzr=True,
        compatibility="pint",
        phase_mean_mode="weighted",
        isort=None,
        design_matrix=np.empty((1, 1)),
        column_units=("Hz",),
        setup=object(),
        _residual_delta_jax_fn=lambda delta: delta,
    )
    kwargs.update(overrides)
    if "fit_params" in overrides and "runtime_fit_params" not in overrides:
        kwargs["runtime_fit_params"] = tuple(overrides["fit_params"])
    return JaxTimingState(**kwargs)


def test_public_surface_reachable_via_jug_timing():
    import jug.timing

    assert jug.timing.JaxTimingState is JaxTimingState
    assert jug.timing.export_jax_timing_state is (
        jax_timing_state.export_jax_timing_state
    )


def test_residual_delta_np_preserves_high_precision_f0(monkeypatch):
    base_f0 = np.longdouble("326.60056708749672367")
    ref_params = {
        "F0": float(base_f0),
        "_high_precision": {"F0": "326.60056708749672367"},
    }

    def compute_residuals(params, setup):
        f0 = get_longdouble(params, "F0")
        return (
            np.array([float((f0 - base_f0) * np.longdouble("1e6"))]),
            None,
            None,
            None,
        )

    monkeypatch.setattr(
        jax_timing_state, "_compute_full_model_residuals", compute_residuals
    )
    state = _state(ref_params=ref_params, ref_theta=np.array([float(base_f0)]))

    with pytest.deprecated_call():
        np.testing.assert_allclose(
            state.residual_delta_np(np.zeros(1)), [0.0], atol=1e-18
        )


def test_linearized_residual_delta_is_plain_matmul():
    design = np.array([[1.0, 0.5], [2.0, -0.5], [0.0, 1.0]])
    state = _state(
        fit_params=("F0", "F1"),
        ref_params={"F0": 100.0, "F1": -1e-15},
        ref_theta=np.array([100.0, -1e-15]),
        reference_residuals_sec=np.zeros(3),
        design_matrix=design,
        column_units=("Hz", "Hz/s"),
    )
    delta = np.array([0.25, -0.5])
    expected = design @ delta
    np.testing.assert_allclose(state.linearized_residual_delta_np(delta), expected)
    np.testing.assert_allclose(
        np.asarray(state.linearized_residual_delta_jax(delta)), expected
    )


def test_export_requires_fit_params():
    with pytest.raises(ValueError, match="fit_params must be non-empty"):
        jax_timing_state.export_jax_timing_state(object(), fit_params=())


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
    # FORMAT 1: filename freq mjd error site [flags...]
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
    state = jax_timing_state.export_jax_timing_state(
        session,
        fit_params=["ECC"],
        param_mapping={"ECC": "E"},
        design_matrix_method="autodiff",
    )
    assert state.runtime_fit_params == ("ECC",)
    assert ("ECC", "E") in state.param_mapping
    assert np.linalg.norm(state.design_matrix[:, 0]) > 0.0
    delta = np.array([1e-6])
    with pytest.deprecated_call():
        r_np = state.residual_delta_np(delta)
    r_jax = np.asarray(state.residual_delta_jax(delta))
    with pytest.deprecated_call():
        r0 = state.residual_delta_np(np.zeros(1))
    assert np.max(np.abs(r_np - r0)) > 0
    np.testing.assert_allclose(r_np, r_jax, atol=1e-10)


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
    state = jax_timing_state.export_jax_timing_state(
        session,
        fit_params=["T0"],
        design_matrix_method="autodiff",
    )
    zeros = np.zeros(1, dtype=np.float64)
    with pytest.deprecated_call():
        r_np = np.asarray(state.residual_delta_np(zeros), dtype=np.float64)
    assert r_np.dtype == np.float64
    assert np.all(np.isfinite(r_np))
    np.testing.assert_allclose(r_np, 0.0, atol=5e-12)
    r_jax = np.asarray(state.residual_delta_jax(zeros), dtype=np.float64)
    assert r_jax.dtype == np.float64
    assert np.all(np.isfinite(r_jax))
    np.testing.assert_allclose(r_jax, r_np, atol=5e-12)
