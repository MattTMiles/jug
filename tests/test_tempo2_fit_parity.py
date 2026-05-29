"""Fit parity tests for Tempo2-compatible mode."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.fitting.optimized_fitter import fit_parameters_optimized
from jug.testing.tempo2_reference import tempo2_reference

from tempo2_fixtures import get_tempo2_fixture


@pytest.mark.tempo2
def test_tempo2_sandbox_fit_smoke():
    fixture = get_tempo2_fixture("epta_j0030_isolated")
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"], dofit=True, fit_params=["F0"])

    assert ref.ntoa > 0
    assert np.all(np.isfinite(ref.residuals_us))
    assert np.isfinite(ref.wrms_us)
    assert ref.params


@pytest.mark.tempo2
def test_jug_tempo2_fit_parity_f0_wls():
    fixture = get_tempo2_fixture("epta_j0030_isolated")
    ref = tempo2_reference(fixture["par_path"], fixture["tim_path"], dofit=True, fit_params=["F0"])

    jug = fit_parameters_optimized(
        fixture["par_path"],
        fixture["tim_path"],
        ["F0"],
        max_iter=2,
        verbose=False,
        compatibility="tempo2",
    )

    assert abs(jug["final_rms"] - ref.wrms_us) < 0.005
    assert abs(float(jug["final_params"]["F0"]) - float(ref.params["F0"]["value"])) < 1e-13
    delta_ns = (np.asarray(jug["residuals_us"]) - ref.residuals_us) * 1000.0
    assert np.sqrt(np.mean(np.square(delta_ns))) < 100.0
