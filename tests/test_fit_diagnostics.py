"""Tests for ill-conditioning diagnostics in optimized fitter."""

from __future__ import annotations

import warnings

import numpy as np

from jug.fitting import optimized_fitter as opt
from tempo2_fixtures import get_tempo2_fixture


def test_compute_condition_diagnostics_flags_ill_conditioned():
    x = np.linspace(0.0, 1.0, 128)
    # Near-collinear columns -> large condition number.
    m = np.column_stack([x, x + 1.0e-12 * x**2])
    diag = opt._compute_condition_diagnostics(m, ["P1", "P2"], threshold=1e10)
    assert diag["n_params"] == 2
    assert diag["labels"] == ["P1", "P2"]
    assert diag["ill_conditioned"]
    assert diag["condition_number"] > 1e10


def test_fit_reports_diagnostics_without_dropping_params(monkeypatch):
    fixture = get_tempo2_fixture("epta_j0030_isolated")
    fit_params = ["F0", "F1"]

    def _forced_ill(_matrix, labels, threshold=1e12):
        return {
            "n_params": len(labels),
            "labels": list(labels),
            "condition_number": 1.0e20,
            "max_abs_correlation": 0.999999,
            "threshold": float(threshold),
            "ill_conditioned": True,
        }

    monkeypatch.setattr(opt, "_compute_condition_diagnostics", _forced_ill)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        result = opt.fit_parameters_optimized(
            fixture["par_path"],
            fixture["tim_path"],
            fit_params,
            max_iter=1,
            verbose=False,
            compatibility="tempo2",
        )

    assert any("Ill-conditioned multi-parameter fit detected" in str(w.message) for w in caught)
    assert result["fit_diagnostics"]["ill_conditioned"] is True
    assert result["fit_diagnostics"]["requested_fit_params"] == fit_params
    assert set(fit_params).issubset(result["final_params"].keys())

