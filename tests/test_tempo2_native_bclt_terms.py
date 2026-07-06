"""DEV ORACLE — BCLT term split vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    delta_ns,
    load_wsrt167_fixture,
)


def test_native_roemer_wsrt167_vs_pytempo():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    roemer = np.asarray(jax.device_get(native.roemer_sec), dtype=np.float64)
    delta = delta_ns(roemer, oracle.fields["roemer_sec"])
    # BCLT roemer uses fixed posPulsar + explicit PM terms (tempo2 calculate_bclt.C).
    assert np.sqrt(np.mean(delta**2)) < 5.0
