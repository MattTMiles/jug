"""DEV ORACLE — EPTA J0613 native chain extension gates (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.residuals.simple_calculator import compute_residuals_simple
from jug.testing.tempo2_reference import tempo2_reference
from tempo2_fixtures import get_tempo2_fixture


@pytest.mark.parametrize(
    "fixture_id,max_rms_ns",
    [
        ("epta_j0613_t2_nrt1400", 10.0),
    ],
)
def test_production_epta_j0613_interim(fixture_id, max_rms_ns):
    fx = get_tempo2_fixture(fixture_id)
    jug = compute_residuals_simple(
        fx["par_path"], fx["tim_path"], verbose=False, compatibility="tempo2"
    )
    ref = tempo2_reference(fx["par_path"], fx["tim_path"])
    delta_ns = (
        np.asarray(jug["residuals_us"], dtype=np.float64)
        - np.asarray(ref.residuals_us, dtype=np.float64)
    ) * 1e3
    delta_ns = delta_ns - np.mean(delta_ns)
    assert np.sqrt(np.mean(delta_ns**2)) < max_rms_ns
