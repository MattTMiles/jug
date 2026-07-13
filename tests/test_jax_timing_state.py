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
