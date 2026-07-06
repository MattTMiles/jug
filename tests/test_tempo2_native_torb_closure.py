"""DEV ORACLE — native torb closure vs pytempo (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax
import jax.numpy as jnp

from jug.io.par_reader import parse_par_file
from jug.residuals.tempo2_native.formbats_jax import compute_torb_closure_jax
from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle
from tempo2_native_test_helpers import (
    compute_native_terms_for_fixture,
    compute_native_terms_model_epoch,
    delta_ns,
    load_wsrt167_fixture,
)


def _torb_from_native(native, pepoch: float) -> np.ndarray:
    return np.asarray(
        jax.device_get(
            compute_torb_closure_jax(
                native.bbat_mjd,
                native.dt_emission_sec,
                jnp.asarray(pepoch, dtype=jnp.float64),
            )
        ),
        dtype=np.float64,
    )


def test_native_torb_closure_strict_formbats_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_for_fixture(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    pepoch = float(parse_par_file(fixture["par_path"])["PEPOCH"])
    torb = _torb_from_native(native, pepoch)
    delta = delta_ns(torb, oracle.fields["torb_sec"])
    assert np.sqrt(np.mean(delta**2)) < 25e6


def test_native_torb_closure_model_epoch_interim_wsrt167():
    fixture = load_wsrt167_fixture()
    native = compute_native_terms_model_epoch(fixture)
    oracle = load_pytempo_native_oracle(
        fixture["par_path"], fixture["tim_path"], fixture_id="wsrt167"
    )
    pepoch = float(parse_par_file(fixture["par_path"])["PEPOCH"])
    torb = _torb_from_native(native, pepoch)
    delta = delta_ns(torb, oracle.fields["torb_sec"])
    assert np.sqrt(np.mean(delta**2)) < 500.0
