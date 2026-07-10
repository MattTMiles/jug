"""``fixed_state_stripped`` tempo2 JAX graph mode gates."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytempo")

import jax
import jax.numpy as jnp

from jug.fitting.jax_residual_delta import make_residual_delta_jax_fn
from jug.residuals.tempo2.compensated import mjd_view_from_daysec
from jug.residuals.tempo2.delta_pack import build_delta_pack_for_setup
from jug.residuals.tempo2.graph_config import TEMPO2_GRAPH_FIXED_STATE_STRIPPED
from jug.residuals.tempo2.model.bbat_lite import bbat_lite_daysec_from_pack
from tempo2_test_helpers import delta_ns

pytestmark = [pytest.mark.tempo2, pytest.mark.slow]


def _wsrt167_setup_from_cache(wsrt167_fixture, wsrt167_session_cache, *, tempo2_native: str):
    from jug.io.par_reader import parse_par_file
    from tempo2_test_helpers import build_fit_setup_from_jug_cache

    params = parse_par_file(wsrt167_fixture["par_path"])
    return build_fit_setup_from_jug_cache(
        params=params,
        session_cached_data=wsrt167_session_cache,
        fit_params=["RAJ", "DECJ", "F0", "DM"],
        tempo2_native=tempo2_native,
    )


@pytest.fixture(scope="module")
def wsrt167_stripped_setup(wsrt167_fixture, wsrt167_session_cache):
    return _wsrt167_setup_from_cache(
        wsrt167_fixture, wsrt167_session_cache, tempo2_native="fixed_state_stripped"
    )


@pytest.fixture(scope="module")
def wsrt167_bclt_setup(wsrt167_fixture, wsrt167_session_cache):
    return _wsrt167_setup_from_cache(
        wsrt167_fixture, wsrt167_session_cache, tempo2_native="fixed_state_bclt"
    )


def test_stripped_delta_pack_mode(wsrt167_stripped_setup):
    setup = wsrt167_stripped_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    pack = build_delta_pack_for_setup(setup)
    assert pack is not None
    assert pack.mode == TEMPO2_GRAPH_FIXED_STATE_STRIPPED
    assert pack.bbat_ref_int_day is not None
    assert pack.bbat_ref_sec_in_day is not None


def test_stripped_lite_bbat_matches_bclt_tail(wsrt167_stripped_setup, wsrt167_bclt_setup):
    """BBAT lite must match ``fixed_state_bclt`` tail at reference parameters."""
    from jug.residuals.tempo2.delta_pack import _native_fixed_state_terms_from_pack

    stripped = wsrt167_stripped_setup
    bclt = wsrt167_bclt_setup
    if stripped.native_chain_static is None or bclt.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    pack = build_delta_pack_for_setup(stripped)
    assert pack is not None
    bclt_pack = build_delta_pack_for_setup(bclt)
    assert bclt_pack is not None

    lite_int, lite_sec = bbat_lite_daysec_from_pack(stripped.params, pack)
    lite_mjd = np.asarray(
        jax.device_get(mjd_view_from_daysec(lite_int, lite_sec)), dtype=np.float64
    )
    bclt_terms, _ = _native_fixed_state_terms_from_pack(stripped.params, bclt_pack)
    bclt_mjd = np.asarray(jax.device_get(bclt_terms.bbat_mjd), dtype=np.float64)
    delta = delta_ns(lite_mjd, bclt_mjd, is_mjd=True)
    assert float(np.sqrt(np.mean(delta**2))) < 1.0


def test_stripped_zero_perturbation_near_zero_delta(wsrt167_stripped_setup):
    setup = wsrt167_stripped_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fn = make_residual_delta_jax_fn(setup=setup, fit_params=["F0"])
    delta = np.asarray(fn(jnp.zeros(1, dtype=jnp.float64)), dtype=np.float64)
    assert float(np.max(np.abs(delta))) < 1e-9


def test_stripped_lite_bbat_oracle_wsrt167(wsrt167_stripped_setup, wsrt167_pytempo_oracle):
    """BBAT lite compensated daysec vs pytempo component-assembled BBAT."""
    setup = wsrt167_stripped_setup
    if setup.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    pack = build_delta_pack_for_setup(setup)
    assert pack is not None
    lite_int, lite_sec = bbat_lite_daysec_from_pack(setup.params, pack)
    lite_mjd = np.asarray(
        jax.device_get(mjd_view_from_daysec(lite_int, lite_sec)), dtype=np.float64
    )
    oracle = wsrt167_pytempo_oracle.fields.get("bbat_from_components_mjd")
    if oracle is None:
        oracle = wsrt167_pytempo_oracle.fields["bbat_mjd"]
    delta = delta_ns(lite_mjd, oracle, is_mjd=True)
    assert float(np.sqrt(np.mean(delta**2))) < 1.0


def test_stripped_envelope_vs_fixed_state_bclt(
    wsrt167_stripped_setup,
    wsrt167_bclt_setup,
):
    stripped = wsrt167_stripped_setup
    bclt = wsrt167_bclt_setup
    if stripped.native_chain_static is None or bclt.native_chain_static is None:
        pytest.skip("native_chain_static unavailable")
    fit_params = ["RAJ", "DECJ", "F0", "DM"]
    eps = jnp.asarray([1e-10, 1e-10, 1e-10, 1e-5], dtype=jnp.float64)
    stripped_fn = make_residual_delta_jax_fn(setup=stripped, fit_params=fit_params)
    bclt_fn = make_residual_delta_jax_fn(setup=bclt, fit_params=fit_params)
    stripped_delta = np.asarray(stripped_fn(eps), dtype=np.float64)
    bclt_delta = np.asarray(bclt_fn(eps), dtype=np.float64)
    diff = stripped_delta - bclt_delta
    rms_ns = float(np.sqrt(np.mean((diff * 1e9) ** 2)))
    assert rms_ns < 1.0