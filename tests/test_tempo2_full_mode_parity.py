"""DEV ORACLE — full in-graph tempo2 JAX chain vs pytempo (wsrt167)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax

from tempo2_test_helpers import (
    compute_native_terms_for_fixture,
    delta_ns,
    load_wsrt167_fixture,
    native_batcorr_days,
    rms_ns,
)


@pytest.fixture(scope="module")
def wsrt167_full_native_terms():
    """Amortize full-mode JAX compile across module."""
    fixture = load_wsrt167_fixture()
    return compute_native_terms_for_fixture(fixture, tempo2_native="full")


def test_full_mode_roemer_vs_pytempo(wsrt167_full_native_terms, wsrt167_pytempo_oracle):
    roemer = np.asarray(jax.device_get(wsrt167_full_native_terms.roemer_sec), dtype=np.float64)
    assert rms_ns(roemer, wsrt167_pytempo_oracle.fields["roemer_sec"]) < 1.0


def test_full_mode_tdis2_vs_pytempo(wsrt167_full_native_terms, wsrt167_pytempo_oracle):
    tdis2 = np.asarray(jax.device_get(wsrt167_full_native_terms.tdis2_sec), dtype=np.float64)
    assert rms_ns(tdis2, wsrt167_pytempo_oracle.fields["tdis2_sec"]) < 1.0


def test_full_mode_dt_ssb_vs_pytempo(wsrt167_full_native_terms, wsrt167_pytempo_oracle):
    dt_ssb = np.asarray(jax.device_get(wsrt167_full_native_terms.dt_ssb_sec), dtype=np.float64)
    assert rms_ns(dt_ssb, wsrt167_pytempo_oracle.fields["dt_ssb_sec"]) < 1.0


def test_full_mode_batcorr_vs_pytempo(wsrt167_full_native_terms, wsrt167_pytempo_oracle):
    bat = native_batcorr_days(wsrt167_full_native_terms)
    assert rms_ns(bat, wsrt167_pytempo_oracle.fields["bat_corr_days"], is_mjd=True) < 1.0


def test_full_mode_bbat_vs_pytempo_components(wsrt167_full_native_terms, wsrt167_pytempo_oracle):
    bbat = np.asarray(jax.device_get(wsrt167_full_native_terms.bbat_mjd), dtype=np.float64)
    oracle = wsrt167_pytempo_oracle.fields.get("bbat_from_components_mjd")
    if oracle is None:
        oracle = wsrt167_pytempo_oracle.fields["bbat_mjd"]
    delta = delta_ns(bbat, oracle, is_mjd=True)
    assert float(np.sqrt(np.mean(delta**2))) < 1.0