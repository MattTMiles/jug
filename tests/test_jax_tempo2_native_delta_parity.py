"""DEV ORACLE — native chain residual delta smoke (Phase 5b)."""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.tempo2_native_quarantine import USE_JAX_TEMPO2_NATIVE_CHAIN


def test_native_delta_module_importable():
    from jug.fitting import jax_residual_delta

    assert hasattr(jax_residual_delta, "make_residual_delta_jax_fn")


@pytest.mark.skipif(
    not USE_JAX_TEMPO2_NATIVE_CHAIN,
    reason="Native delta path enabled with USE_JAX_TEMPO2_NATIVE_CHAIN",
)
def test_native_delta_wsrt167_smoke():
    pytest.skip("Native chain delta parity pending Phase 5b wiring")
