"""DEV ORACLE — production native chain avoids host round-trips in the JIT graph."""

from __future__ import annotations

import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2]

import jax

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2_native.chain_jax import prepare_native_chain_from_simple_result
from tempo2_native_test_helpers import compute_native_terms_for_fixture, load_wsrt167_fixture


def test_native_chain_no_device_get_mid_bclt(monkeypatch):
    fixture = load_wsrt167_fixture()
    calls = []

    orig = jax.device_get

    def tracked(x, *args, **kwargs):
        calls.append(1)
        return orig(x, *args, **kwargs)

    monkeypatch.setattr(jax, "device_get", tracked)
    compute_native_terms_for_fixture(fixture)
    assert len(calls) == 0


def test_native_chain_does_not_call_legacy_bclt_numpy_dm(monkeypatch):
    fixture = load_wsrt167_fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "_dm_vals_numpy must not run on the unified JIT path"
        )

    monkeypatch.setattr(
        "jug.residuals.tempo2_native.model_jax._dm_vals_numpy",
        forbidden,
    )
    compute_native_terms_for_fixture(fixture)


def test_unified_model_rejects_host_shortcuts(monkeypatch):
    """JIT wrapper must not re-enter host IFTE/ephemeris/DM staging."""
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"], fixture["tim_path"], verbose=False, compatibility="tempo2"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("host shortcut used in native production path")

    monkeypatch.setattr("jug.utils.ifteph.ifte_delta_t_mjd", forbidden)
    monkeypatch.setattr(
        "jug.residuals.tempo2_native.model_jax.prepare_ephemeris_inputs_jax",
        forbidden,
    )
    monkeypatch.setattr(
        "jug.residuals.tempo2_native.model_jax._dm_vals_numpy",
        forbidden,
    )
    prepare_native_chain_from_simple_result(jug, params, toas)
