"""DEV ORACLE — native chain JIT boundary vs host staging inputs."""

from __future__ import annotations

import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.dev_oracle, pytest.mark.tempo2, pytest.mark.slow]

import jax

from jug.io.par_reader import parse_par_file
from jug.io.tim_reader import parse_tim_file_mjds
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2.fit_setup import prepare_native_chain_from_simple_result
from tempo2_native_test_helpers import load_wsrt167_fixture


@pytest.fixture(scope="module")
def wsrt167_native_inputs():
    """Parse once; skip legacy overlay so simple_calculator stays lightweight."""
    fixture = load_wsrt167_fixture()
    params = parse_par_file(fixture["par_path"])
    toas = parse_tim_file_mjds(fixture["tim_path"])
    jug = compute_residuals_simple(
        fixture["par_path"],
        fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
        skip_native_bclt_overlay=True,
    )
    return fixture, params, toas, jug


@pytest.fixture(scope="module", autouse=True)
def _warm_native_jit(wsrt167_native_inputs):
    """Compile the unified native chain once per module (amortize JAX JIT)."""
    _fixture, params, toas, jug = wsrt167_native_inputs
    prepare_native_chain_from_simple_result(jug, params, toas)


def test_native_chain_no_device_get_mid_bclt(monkeypatch, wsrt167_native_inputs):
    calls = []

    orig = jax.device_get

    def tracked(x, *args, **kwargs):
        calls.append(1)
        return orig(x, *args, **kwargs)

    monkeypatch.setattr(jax, "device_get", tracked)
    _fixture, params, toas, jug = wsrt167_native_inputs
    prepare_native_chain_from_simple_result(jug, params, toas)
    assert len(calls) == 0


def test_native_chain_does_not_call_legacy_bclt_numpy_dm(monkeypatch, wsrt167_native_inputs):
    def forbidden(*args, **kwargs):
        raise AssertionError(
            "_dm_vals_numpy must not run on the unified JIT path"
        )

    monkeypatch.setattr(
        "jug.residuals.tempo2.model.static._dm_vals_numpy",
        forbidden,
    )
    _fixture, params, toas, jug = wsrt167_native_inputs
    prepare_native_chain_from_simple_result(jug, params, toas)


def test_unified_native_path_has_no_host_ifte_geometry(monkeypatch, wsrt167_native_inputs):
    """Host provides only static tables; geometry derived in-graph."""
    _fixture, params, toas, jug = wsrt167_native_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("host ephemeris shortcut used in unified JIT path")

    monkeypatch.setattr(
        "jug.residuals.tempo2.model.static.prepare_ephemeris_inputs_jax",
        forbidden,
    )
    monkeypatch.setattr(
        "jug.delays.tempo2_ephemeris.compute_tempo2_observatory_state",
        forbidden,
    )
    prepare_native_chain_from_simple_result(jug, params, toas)


def test_public_native_path_does_not_call_host_ifte_delta(monkeypatch, wsrt167_native_inputs):
    """IFTE ``IF_deltaT`` runs inside JIT; host ``ifte_delta_t_mjd`` is not called."""
    _fixture, params, toas, jug = wsrt167_native_inputs
    calls = {"ifte": 0}
    from jug.utils import ifteph

    _orig_ifte = ifteph.ifte_delta_t_mjd

    def tracked_ifte(mjd):
        calls["ifte"] += 1
        return _orig_ifte(mjd)

    monkeypatch.setattr(ifteph, "ifte_delta_t_mjd", tracked_ifte)
    prepare_native_chain_from_simple_result(jug, params, toas)
    assert calls["ifte"] == 0, "native chain must evaluate IFTE inside JAX JIT"


def test_unified_native_path_has_no_host_einstein_rate(monkeypatch, wsrt167_native_inputs):
    """``einsteinRate`` is evaluated inside JIT, not pre-baked on the host."""
    _fixture, params, toas, jug = wsrt167_native_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("host einstein_rate precompute used on unified path")

    monkeypatch.setattr(
        "jug.residuals.tempo2.model.static.tempo2_einstein_rate_host",
        forbidden,
    )
    prepare_native_chain_from_simple_result(jug, params, toas)


def test_unified_native_path_has_no_host_troposphere(monkeypatch, wsrt167_native_inputs):
    """Production tropo uses Tempo2-native JAX kernel, not ``troposphere.py``."""
    _fixture, params, toas, jug = wsrt167_native_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("host troposphere helper used on unified path")

    monkeypatch.setattr("jug.delays.troposphere.compute_tropospheric_delay", forbidden)
    prepare_native_chain_from_simple_result(jug, params, toas)
