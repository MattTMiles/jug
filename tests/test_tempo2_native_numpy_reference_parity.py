"""NumPy host-frozen reference vs pytempo (hybrid parity gates)."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

from jug.residuals.tempo2_native.chain_numpy import (
    compute_tempo2_native_terms_numpy_from_simple_result,
)
from tempo2_native_test_helpers import load_wsrt167_fixture, rms_ns


@pytest.fixture(autouse=True)
def _enable_numpy_native_chain(monkeypatch):
    monkeypatch.setenv("JUG_DEV_NUMPY_TEMPO2_CHAIN", "1")


def test_numpy_host_frozen_reference_matches_pytempo_wsrt167(wsrt167_pytempo_oracle):
    fixture = load_wsrt167_fixture()
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.residuals.simple_calculator import compute_residuals_simple

    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        np_terms = compute_tempo2_native_terms_numpy_from_simple_result(
            jug, params, toas
        )
    pt = wsrt167_pytempo_oracle.fields

    assert rms_ns(np_terms["correction_tt_sec"], pt["correction_tt_sec"]) < 1.0
    assert rms_ns(np_terms["correction_tt_tb_sec"], pt["correction_tt_tb_sec"]) < 1.0
    assert rms_ns(np_terms["roemer_sec"], pt["roemer_sec"]) < 1.0
    assert rms_ns(np_terms["tdis1_sec"], pt["tdis1_sec"]) < 1.0
    assert rms_ns(np_terms["tdis2_sec"], pt["tdis2_sec"]) < 1.0
    assert rms_ns(np_terms["dt_ssb_sec"], pt["dt_ssb_sec"]) < 1.0
    assert (
        rms_ns(
            np_terms["bat_corr_day"] + np_terms["bat_corr_day_residual"],
            pt["bat_corr_days"],
            is_mjd=True,
        )
        < 1.0
    )
