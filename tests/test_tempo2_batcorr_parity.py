"""DEV ORACLE — native formBats batCorr parity (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from jug.testing.tempo2_reference import tempo2_reference
from tempo2_test_helpers import delta_ns


def test_native_batcorr_wsrt167_matches_pytempo_bat_corr_days(wsrt167_fixture):
    pytest.importorskip("pytempo")
    ref = tempo2_reference(
        wsrt167_fixture["par_path"], wsrt167_fixture["tim_path"], include_batcorr=True
    )
    oracle = load_pytempo_native_oracle(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        fixture_id="wsrt167",
    )
    lib_delta = delta_ns(ref.bat_corr_days, oracle.fields["bat_corr_days"], is_mjd=True)
    assert np.sqrt(np.mean(lib_delta**2)) < 1.0
