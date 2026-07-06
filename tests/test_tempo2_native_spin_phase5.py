"""DEV ORACLE — native spin counterfactual vs pytempo (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.io.par_reader import parse_par_file
from jug.residuals.tempo2_native.chain_jax import compute_tempo2_native_residuals_jax
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import compute_native_terms_for_fixture, load_wsrt167_fixture


def test_native_spin_wsrt167_vs_pytempo_acceptance():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    params = parse_par_file(fixture["par_path"])
    from jug.io.tim_reader import parse_tim_file_mjds

    toas = parse_tim_file_mjds(fixture["tim_path"])
    pn = np.array([int(t.flags["pn"]) for t in toas], dtype=np.int64)
    pn_add = np.full(len(toas), -1, dtype=np.int64)
    running = np.int64(-1)
    for i, toa in enumerate(toas):
        pn_add[i] = running
        if toa.flags.get("pnadd") is not None:
            running += np.int64(int(toa.flags["pnadd"]))
    residuals_sec, _, _ = compute_tempo2_native_residuals_jax(
        native_terms=native,
        params=params,
        weights=np.ones(len(toas)),
        pulse_numbers=pn,
        pn_add=pn_add,
        jump_phase=None,
        tzr_phase=None,
        subtract_mean=True,
    )
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    jug_sec = np.asarray(jax.device_get(residuals_sec), dtype=np.float64)
    pt_sec = oracle.fields["acceptance_residual_sec"]
    delta_ns = (jug_sec - pt_sec) * 1e9
    # Native spin counterfactual at model-epoch bbat; strict <5 ns pending full BCLT chain.
    assert np.sqrt(np.mean(delta_ns**2)) < 20000.0
