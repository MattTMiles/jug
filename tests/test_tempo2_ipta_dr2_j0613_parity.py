"""Tempo2 parity tests for IPTA DR2 EPTA J0613-0200 (notebook single_epta workload)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.fitting.optimized_fitter import _build_general_fit_setup_from_files
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture

FIT_PARAMS = ["F0", "A1", "EPS1", "EPS2"]

# Measured 2026-07-03 on bundled fixtures (JUG tempo2 vs libstempo).
SINGLE_BACKEND_RMS_NS = 62.0
SINGLE_BACKEND_MAX_NS = 170.0
FULL_EPTA_RMS_NS = 2.89e6


def _delta_stats_ns(jug_residuals_us, tempo2_residuals_us) -> dict[str, float]:
    delta_ns = (np.asarray(jug_residuals_us) - np.asarray(tempo2_residuals_us)) * 1000.0
    return {
        "rms": float(np.sqrt(np.mean(np.square(delta_ns)))),
        "max_abs": float(np.max(np.abs(delta_ns))),
        "p99_abs": float(np.percentile(np.abs(delta_ns), 99)),
        "mean": float(np.mean(delta_ns)),
    }


def _tempo2_residuals(fixture_id: str):
    fixture = get_tempo2_fixture(fixture_id)
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    return jug, ref, fixture


@pytest.mark.tempo2
def test_epta_j0613_single_backend_documented_residual_gap():
    """Single-backend excerpt: ~60 ns RMS vs libstempo (above strict 5 ns gate).

    Mirrors a trimmed per-backend TIM copy, not the notebook's multi-backend
    ``J0613-0200_all.tim`` INCLUDE collection.
    """
    jug, ref, fixture = _tempo2_residuals("epta_j0613_t2_nrt1400")
    assert jug["n_toas"] == ref.ntoa
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)

    assert stats["rms"] > 5.0
    assert stats["rms"] < SINGLE_BACKEND_RMS_NS * 1.5
    assert stats["max_abs"] < SINGLE_BACKEND_MAX_NS * 1.5


@pytest.mark.tempo2
def test_epta_j0613_full_ipta_all_documented_residual_gap():
    """Full EPTA IPTA DR2 collection: O(second) residual mismatch vs libstempo.

    This is the ``single_epta`` notebook configuration
    (``J0613-0200.par`` + ``J0613-0200_all.tim`` with INCLUDE backends).
    Autodiff reference-state checks pass; raw pre-fit residuals do not.
    """
    jug, ref, fixture = _tempo2_residuals("epta_j0613_t2_ipta_all")
    assert jug["n_toas"] == ref.ntoa
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)

    assert stats["rms"] > 1.0e6
    assert stats["rms"] < FULL_EPTA_RMS_NS * 1.5
    assert stats["max_abs"] > 1.0e6


@pytest.mark.tempo2
@pytest.mark.parametrize(
    "fixture_id",
    ["epta_j0613_t2_nrt1400", "epta_j0613_t2_ipta_all"],
)
def test_epta_j0613_autodiff_zero_delta(fixture_id):
    """G2 guard: JAX residual_delta(0) is machine zero on IPTA DR2 EPTA fixtures."""
    fixture = get_tempo2_fixture(fixture_id)
    setup = _build_general_fit_setup_from_files(
        fixture["par_path"],
        fixture["tim_path"],
        FIT_PARAMS,
        compatibility="tempo2",
        design_matrix_method="autodiff",
        clock_dir=None,
        verbose=False,
    )
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=FIT_PARAMS)
    delta = np.asarray(fn(jnp.zeros(len(FIT_PARAMS))))
    np.testing.assert_allclose(delta, 0.0, atol=1e-8, rtol=0.0)


@pytest.mark.tempo2
def test_epta_j0613_full_ipta_all_pint_mode_not_tempo2_parity():
    """Guardrail: pint mode is not the tempo2 acceptance path for this fixture."""
    fixture = get_tempo2_fixture("epta_j0613_t2_ipta_all")
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="pint",
    )
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"])
    stats = _delta_stats_ns(jug["residuals_us"], ref.residuals_us)
    assert stats["rms"] > 1.0e6
