"""Smoke tests for the libstempo sandbox oracle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("libstempo")

from jug.testing.tempo2_reference import tempo2_reference


DATA_TEMPO2 = Path(__file__).parent / "data_tempo2"


@pytest.mark.tempo2
def test_tempo2_sandbox_smoke_epta_isolated():
    par = DATA_TEMPO2 / "epta_j0030_isolated" / "epta_j0030_isolated.par"
    tim = DATA_TEMPO2 / "epta_j0030_isolated" / "epta_j0030_isolated.tim"

    ref = tempo2_reference(par, tim)

    assert ref.ntoa == 10
    assert ref.residuals_us.shape == ref.errors_us.shape
    assert np.all(np.isfinite(ref.residuals_us))
    assert np.all(np.isfinite(ref.errors_us))
    assert np.isfinite(ref.wrms_us)
