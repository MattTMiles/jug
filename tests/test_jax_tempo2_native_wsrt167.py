"""DEV ORACLE — full native chain vs libstempo on wsrt167 (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.tempo2_native_quarantine import USE_JAX_TEMPO2_NATIVE_CHAIN
from jug.testing.tempo2_reference import tempo2_reference
from tempo2_fixtures import get_tempo2_fixture


@pytest.mark.skipif(
    not USE_JAX_TEMPO2_NATIVE_CHAIN,
    reason="Native chain disabled until Phase 4 gates pass",
)
def test_jax_native_wsrt167_residuals():
    from jug.residuals.simple_calculator import compute_residuals_simple

    fx = get_tempo2_fixture("wsrt167")
    jug = compute_residuals_simple(
        fx["par_path"], fx["tim_path"], verbose=False, compatibility="tempo2"
    )
    ref = tempo2_reference(fx["par_path"], fx["tim_path"])
    delta_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    delta_ns = delta_ns - np.mean(delta_ns)
    assert np.sqrt(np.mean(delta_ns**2)) < 5.0


def test_production_wsrt167_baseline_documented():
    """Production path remains ~16 ns until USE_JAX_TEMPO2_NATIVE_CHAIN flip."""
    from jug.residuals.simple_calculator import compute_residuals_simple

    fx = get_tempo2_fixture("wsrt167")
    jug = compute_residuals_simple(
        fx["par_path"], fx["tim_path"], verbose=False, compatibility="tempo2"
    )
    ref = tempo2_reference(fx["par_path"], fx["tim_path"])
    delta_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    delta_ns = delta_ns - np.mean(delta_ns)
    rms = float(np.sqrt(np.mean(delta_ns**2)))
    assert 14.0 < rms < 20.0
