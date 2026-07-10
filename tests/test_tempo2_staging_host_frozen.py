"""DEV ORACLE — host-frozen staging JAX tail vs pytempo (no full in-graph compile)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result
from tempo2_test_helpers import load_wsrt167_fixture, native_batcorr_days, rms_ns


def _staging_terms(fixture):
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=True,
        tempo2_native="staged_bclt",
    )
    return prepare_tempo2_chain_from_simple_result(jug, params, toas)


def test_staging_roemer_matches_pytempo_wsrt167(wsrt167_pytempo_oracle):
    fixture = load_wsrt167_fixture()
    terms = _staging_terms(fixture)
    roemer = np.asarray(jax.device_get(terms.roemer_sec), dtype=np.float64)
    assert rms_ns(roemer, wsrt167_pytempo_oracle.fields["roemer_sec"]) < 1.0


def test_staging_tdis2_matches_pytempo_wsrt167(wsrt167_pytempo_oracle):
    fixture = load_wsrt167_fixture()
    terms = _staging_terms(fixture)
    tdis2 = np.asarray(jax.device_get(terms.tdis2_sec), dtype=np.float64)
    assert rms_ns(tdis2, wsrt167_pytempo_oracle.fields["tdis2_sec"]) < 1.0


def test_staging_dt_ssb_matches_pytempo_wsrt167(wsrt167_pytempo_oracle):
    fixture = load_wsrt167_fixture()
    terms = _staging_terms(fixture)
    dt_ssb = np.asarray(jax.device_get(terms.dt_ssb_sec), dtype=np.float64)
    assert rms_ns(dt_ssb, wsrt167_pytempo_oracle.fields["dt_ssb_sec"]) < 1.0


def test_staging_batcorr_matches_pytempo_wsrt167(wsrt167_pytempo_oracle):
    fixture = load_wsrt167_fixture()
    terms = _staging_terms(fixture)
    bat = native_batcorr_days(terms)
    assert rms_ns(bat, wsrt167_pytempo_oracle.fields["bat_corr_days"], is_mjd=True) < 1.0
