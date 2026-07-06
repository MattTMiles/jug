"""DEV ORACLE — native formBats batCorr parity (Phase 1)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("libstempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from jug.testing.tempo2_reference import tempo2_reference
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    compute_native_terms_model_epoch,
    delta_ns,
    load_wsrt167_fixture,
    native_batcorr_days,
)


def test_native_batcorr_strict_formbats_wsrt167():
    """Strict formBats path on wsrt167 (~60 us batCorr vs pytempo)."""
    pytest.importorskip("pytempo")
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    delta = delta_ns(native_batcorr_days(native), oracle.fields["bat_corr_days"], is_mjd=True)
    rms = float(np.sqrt(np.mean(delta**2)))
    assert rms < 1.0


def test_native_batcorr_model_epoch_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_model_epoch(fixture)
    ref = tempo2_reference(
        fixture["par_path"], fixture["tim_path"], include_batcorr=True
    )
    delta = delta_ns(native_batcorr_days(native), ref.bat_corr_days, is_mjd=True)
    assert np.sqrt(np.mean(delta**2)) < 300.0
    assert np.max(np.abs(delta)) < 700.0


def test_native_batcorr_wsrt167_matches_pytempo_bat_corr_days():
    pytest.importorskip("pytempo")
    fixture = load_wsrt167_fixture()
    ref = tempo2_reference(
        fixture["par_path"], fixture["tim_path"], include_batcorr=True
    )
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    lib_delta = delta_ns(ref.bat_corr_days, oracle.fields["bat_corr_days"], is_mjd=True)
    assert np.sqrt(np.mean(lib_delta**2)) < 1.0
