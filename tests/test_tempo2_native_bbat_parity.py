"""DEV ORACLE — native bbat parity vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    compute_native_terms_model_epoch,
    delta_ns,
    load_wsrt167_fixture,
)


def test_native_bbat_strict_formbats_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    bbat = np.asarray(jax.device_get(native.bbat_mjd), dtype=np.float64)
    delta = delta_ns(bbat, oracle.fields["bbat_mjd"], is_mjd=True)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0


def test_native_bbat_model_epoch_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_model_epoch(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    bbat = np.asarray(jax.device_get(native.bbat_mjd), dtype=np.float64)
    delta = delta_ns(bbat, oracle.fields["bbat_mjd"], is_mjd=True)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 500.0
