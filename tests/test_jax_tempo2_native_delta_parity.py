"""DEV ORACLE — native chain residual_delta smoke (Phase 5)."""

from __future__ import annotations

import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]


def test_native_delta_module_importable():
    from jug.fitting import jax_residual_delta

    assert hasattr(jax_residual_delta, "make_residual_delta_jax_fn")
    assert hasattr(jax_residual_delta, "compute_autodiff_designmatrix_from_setup")


def test_native_delta_phase5_gates_documented():
    """Full Phase 5 gates: tests/test_tempo2_native_residual_delta_jax.py."""
    from pathlib import Path

    gate_path = Path(__file__).with_name("test_tempo2_native_residual_delta_jax.py")
    text = gate_path.read_text(encoding="utf-8")
    assert "test_native_autodiff_designmatrix_f0_matches_libstempo" in text
    assert "test_native_residual_delta_uses_full_chain_not_taylor" in text
