"""Unit tests for jug.residuals.gauge (phase gauge)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jug.residuals.gauge import (
    ReferenceGauge,
    apply_phase_gauge,
    gauge_offset_sec,
    reconstruct_absolute_residuals,
)


def test_mode_none_is_identity():
    r = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    out = apply_phase_gauge(r, ReferenceGauge(mode="none"))
    assert out is r
    assert gauge_offset_sec(r, ReferenceGauge(mode="none")) == 0.0


def test_mode_mean_unweighted():
    r = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gauged = apply_phase_gauge(r, ReferenceGauge(mode="mean", weights=None))
    np.testing.assert_allclose(gauged, r - np.mean(r))
    np.testing.assert_allclose(np.mean(gauged), 0.0, atol=1e-15)


def test_mode_mean_weighted():
    r = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    w = np.array([0.25, 0.25, 0.5], dtype=np.float64)
    gauged = apply_phase_gauge(r, ReferenceGauge(mode="mean", weights=w))
    expected_c = float(np.sum(r * w) / np.sum(w))
    np.testing.assert_allclose(gauged, r - expected_c)
    np.testing.assert_allclose(float(np.sum(gauged * w)), 0.0, atol=1e-15)


def test_mode_constant():
    r = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gauged = apply_phase_gauge(r, ReferenceGauge(mode="constant", offset_sec=0.5))
    np.testing.assert_allclose(gauged, r - 0.5)


def test_uniform_weights_agree_with_unweighted():
    r = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float64)
    w = np.full(r.shape, 0.25, dtype=np.float64)
    u = apply_phase_gauge(r, ReferenceGauge(mode="mean", weights=None))
    v = apply_phase_gauge(r, ReferenceGauge(mode="mean", weights=w))
    np.testing.assert_allclose(u, v)


def test_reference_gauge_weights_are_readonly_copies():
    w = np.array([0.5, 0.5], dtype=np.float64)
    gauge = ReferenceGauge(mode="mean", weights=w)
    assert gauge.weights is not w
    assert not gauge.weights.flags.writeable
    w[0] = 0.0
    np.testing.assert_allclose(gauge.weights, [0.5, 0.5])


def test_tracer_safe_under_jit():
    from jug.residuals.gauge import _gauge_offset_values

    @jax.jit
    def centered(r, w):
        return r - _gauge_offset_values(r, mode="mean", weights=w, xp=jnp)

    r = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)
    w = jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float64)
    out = centered(r, w)
    expected = r - jnp.sum(r * w) / jnp.sum(w)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))


def test_constant_algebra_not_linear_not_idempotent():
    r = np.array([1.0, -1.0, 0.5], dtype=np.float64)
    c = 0.25
    gauge = ReferenceGauge(mode="constant", offset_sec=c)
    g0 = apply_phase_gauge(np.zeros_like(r), gauge)
    np.testing.assert_allclose(g0, -c * np.ones_like(r))
    assert not np.allclose(g0, 0.0)

    once = apply_phase_gauge(r, gauge)
    twice = apply_phase_gauge(once, gauge)
    assert not np.allclose(twice, once)
    np.testing.assert_allclose(twice, r - 2 * c)


def test_mean_algebra_idempotent_and_distributes():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    b = np.array([-0.5, 0.0, 1.5], dtype=np.float64)
    gauge = ReferenceGauge(mode="mean", weights=None)
    pa = apply_phase_gauge(a, gauge)
    pb = apply_phase_gauge(b, gauge)
    np.testing.assert_allclose(apply_phase_gauge(pa, gauge), pa)
    np.testing.assert_allclose(
        apply_phase_gauge(a + b, gauge),
        pa + pb,
    )
    np.testing.assert_allclose(apply_phase_gauge(np.zeros_like(a), gauge), 0.0)


def test_validation_bad_mode():
    with pytest.raises(ValueError, match="none.*mean.*constant"):
        gauge_offset_sec(np.ones(2), ReferenceGauge(mode="bogus"))  # type: ignore[arg-type]


def test_validation_payload_incompatibilities():
    r = np.ones(3)
    with pytest.raises(ValueError, match="offset_sec"):
        gauge_offset_sec(r, ReferenceGauge(mode="constant"))
    with pytest.raises(ValueError, match="weights"):
        gauge_offset_sec(
            r, ReferenceGauge(mode="constant", offset_sec=1.0, weights=np.ones(3))
        )
    with pytest.raises(ValueError, match="offset_sec"):
        gauge_offset_sec(r, ReferenceGauge(mode="mean", offset_sec=1.0))
    with pytest.raises(ValueError, match="weights or offset_sec"):
        gauge_offset_sec(r, ReferenceGauge(mode="none", offset_sec=1.0))
    with pytest.raises(ValueError, match="shape"):
        gauge_offset_sec(r, ReferenceGauge(mode="mean", weights=np.ones(2)))


def test_validation_empty_short_circuit_skips_zero_sum_weights():
    empty = np.asarray([], dtype=np.float64)
    w_empty = np.asarray([], dtype=np.float64)
    assert gauge_offset_sec(empty, ReferenceGauge(mode="mean", weights=w_empty)) == 0.0
    out = apply_phase_gauge(empty, ReferenceGauge(mode="mean", weights=w_empty))
    assert len(np.asarray(out)) == 0


def test_validation_weight_contents_when_n_positive():
    r = np.ones(3)
    with pytest.raises(ValueError, match="finite"):
        gauge_offset_sec(
            r, ReferenceGauge(mode="mean", weights=np.array([1.0, np.nan, 1.0]))
        )
    with pytest.raises(ValueError, match="non-negative"):
        gauge_offset_sec(
            r, ReferenceGauge(mode="mean", weights=np.array([1.0, -0.1, 1.0]))
        )
    with pytest.raises(ValueError, match="positive finite sum"):
        gauge_offset_sec(
            r, ReferenceGauge(mode="mean", weights=np.zeros(3))
        )


def test_reconstruct_constant_raises_naming_refphs_tzr():
    r = np.ones(3)
    d = np.zeros(3)
    with pytest.raises(NotImplementedError, match="REFPHS TZR"):
        reconstruct_absolute_residuals(
            r, d, ReferenceGauge(mode="constant", offset_sec=1e-6)
        )
