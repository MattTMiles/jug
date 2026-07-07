"""DEV ORACLE — tempo2 native graph absolute eval vs libstempo on wsrt167."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_reference import tempo2_reference
from tempo2_fixtures import get_tempo2_fixture


def test_jax_native_wsrt167_residuals():
    """Tempo2 absolute eval uses native graph (default staged_bclt); ~18 ns vs libstempo."""
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
