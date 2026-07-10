"""Tests for ill-conditioning diagnostics in optimized fitter."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from jug.fitting import optimized_fitter as opt

GOLDEN_DIR = Path(__file__).parent / "data_golden"
PAR = GOLDEN_DIR / "J1909_proper.par"
TIM = GOLDEN_DIR / "J1909_proper.tim"


def test_compute_condition_diagnostics_flags_ill_conditioned():
    x = np.linspace(0.0, 1.0, 128)
    m = np.column_stack([x, x + 1.0e-12 * x**2])
    diag = opt._compute_condition_diagnostics(m, ["P1", "P2"], threshold=1e10)
    assert diag["n_params"] == 2
    assert diag["labels"] == ["P1", "P2"]
    assert diag["ill_conditioned"]
    assert diag["condition_number"] > 1e10


def test_fit_reports_diagnostics_without_dropping_params(monkeypatch):
    if not PAR.exists() or not TIM.exists():
        pytest.skip("golden J1909 dataset not found")
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
            PAR,
            TIM,
            fit_params,
            max_iter=1,
            verbose=False,
            compatibility="pint",
        )

    assert any("Ill-conditioned multi-parameter fit detected" in str(w.message) for w in caught)
    assert result["fit_diagnostics"]["ill_conditioned"] is True
    assert result["fit_diagnostics"]["requested_fit_params"] == fit_params
    assert set(fit_params).issubset(result["final_params"].keys())