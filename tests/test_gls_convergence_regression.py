"""Regression guard for scale-aware GLS convergence on J1747-4036."""

import os
from pathlib import Path

import numpy as np
import pytest

from jug.engine.session import TimingSession

ROOT = Path(__file__).parents[1]
PAR = ROOT / "data/pulsars/NG_data/NG_15yr_partim/J1747-4036_PINT_20220302.nb.par"
TIM = ROOT / "data/pulsars/NG_data/NG_15yr_partim/J1747-4036_PINT_20220302.nb.tim"
CLOCK = ROOT / "data/clock"

pytestmark = pytest.mark.skipif(
    not (PAR.exists() and TIM.exists()), reason="J1747 NG15 data unavailable",
)

def _fit():
    session = TimingSession(str(PAR), str(TIM), clock_dir=str(CLOCK))
    return session.fit_parameters(max_iter=5)

def test_default_gls_reaches_strict_reference(monkeypatch):
    monkeypatch.delenv("JUG_GLS_DTOL", raising=False)
    default = _fit()
    monkeypatch.setenv("JUG_GLS_DTOL", "0")
    strict = _fit()

    d = np.asarray(default["residuals_us"]) - np.asarray(strict["residuals_us"])
    d -= np.mean(d)
    assert np.std(d) * 1e3 < 0.02

    for component in ("RedNoise", "ECORR"):
        noise_d = (np.asarray(default["noise_realizations"][component])
                   - np.asarray(strict["noise_realizations"][component]))
        noise_d -= np.mean(noise_d)
        assert np.std(noise_d) * 1e3 < 0.01


def test_rejected_gls_trials_retain_current_state(monkeypatch):
    monkeypatch.setenv("JUG_GLS_OBJ_ATOL", "-1e100")
    session = TimingSession(str(PAR), str(TIM), clock_dir=str(CLOCK))
    hp = session.params.get("_high_precision", {})
    initial = {
        param: np.longdouble(hp.get(param, session.params[param]))
        for param in ("F0", "F1")
    }

    result = session.fit_parameters(max_iter=5)

    assert result["step_failed"] is True
    assert result["converged"] is False
    assert result["iterations"] == 1
    for param, value in initial.items():
        assert result["final_params_ld"][param] == value
