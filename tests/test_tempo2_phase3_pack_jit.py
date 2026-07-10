"""Phase 3 — host-frozen pack build, dt_ssb cache, shared JIT closure."""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pytempo")
pytestmark = [pytest.mark.tempo2, pytest.mark.dev_oracle]

from jug.fitting.jax_residual_delta import (
    _prepare_residual_delta_jax,
    compute_autodiff_designmatrix_from_setup,
    make_residual_delta_jax_fn,
)
from jug.residuals.simple_calculator import compute_residuals_simple
from jug.residuals.tempo2.delta_pack import (
    build_delta_pack_for_setup,
    build_fixed_state_bclt_delta_pack,
)
from tempo2_test_helpers import load_wsrt167_fixture


@pytest.mark.slow
def test_bclt_dt_ssb_cached_in_term_diagnostics(wsrt167_fixture_paths):
    """Host residuals export reference BCLT dt_ssb for pack-build fast path."""
    par_path, tim_path = wsrt167_fixture_paths
    jug = compute_residuals_simple(
        par_path, tim_path, verbose=False, compatibility="tempo2"
    )
    td = jug["term_diagnostics"]
    assert "bclt_dt_ssb_sec" in td
    assert "dt_ssb_sec" in td
    cached = np.asarray(td["bclt_dt_ssb_sec"], dtype=np.float64)
    assert cached.shape == (jug["n_toas"],)
    assert np.all(np.isfinite(cached))


def test_host_frozen_pack_build_skips_spk_load(wsrt167_setup_multiparam):
    """Host-frozen delta packs must not load SPK/EOP tables at build time."""
    setup = wsrt167_setup_multiparam
    setup.tempo2_native = "fixed_state_bclt"

    def _fail_spk(*_args, **_kwargs):
        raise AssertionError("pack_tempo2_spk_jax must not run for host-frozen packs")

    with patch(
        "jug.delays.tempo2_spk_jax.pack_tempo2_spk_jax",
        side_effect=_fail_spk,
    ):
        pack = build_fixed_state_bclt_delta_pack(setup)

    assert pack is not None
    assert pack.mode == "fixed_state_bclt"
    assert pack.dt_ssb_ref_sec is not None


def test_shared_jit_residual_and_jacfwd_agree(wsrt167_setup):
    """Residual fn and jacfwd design matrix share one XLA core."""
    setup = wsrt167_setup
    fit_params = ("F0",)
    core, residual_fn, jac_fn = _prepare_residual_delta_jax(
        setup=setup, fit_params=fit_params
    )
    zero = jnp.zeros((1,), dtype=jnp.float64)
    r0 = np.asarray(residual_fn(zero), dtype=np.float64)
    jac = np.asarray(jac_fn(zero), dtype=np.float64)
    assert r0.shape == (len(setup.tdb_mjd),)
    assert jac.shape == (len(setup.tdb_mjd), 1)

    fn = make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
    r1 = np.asarray(fn(zero), dtype=np.float64)
    assert np.allclose(r0, r1, rtol=0, atol=0)

    dm = compute_autodiff_designmatrix_from_setup(setup=setup, fit_params=fit_params)
    assert dm.shape == (len(setup.tdb_mjd), 1)
    assert np.allclose(-jac[:, 0], dm[:, 0], rtol=0, atol=0)


def test_prepare_residual_delta_jax_session_cache(wsrt167_setup):
    """WLS-style path: residual fn + design matrix share one pack build per setup."""
    setup = wsrt167_setup
    setup.residual_delta_jax_cache = None
    fit_params = ("F0",)
    pack_calls: list[int] = []
    original = build_delta_pack_for_setup

    def counting(setup_arg):
        pack_calls.append(1)
        return original(setup_arg)

    with patch(
        "jug.fitting.jax_residual_delta.build_delta_pack_for_setup",
        side_effect=counting,
    ):
        make_residual_delta_jax_fn(setup=setup, fit_params=fit_params)
        compute_autodiff_designmatrix_from_setup(setup=setup, fit_params=fit_params)

    assert len(pack_calls) == 1
    assert setup.residual_delta_jax_cache is not None
    assert len(setup.residual_delta_jax_cache) == 1


@pytest.mark.slow
def test_wsrt167_graph_timing_f0(
    wsrt167_fixture, wsrt167_jug, wsrt167_params, wsrt167_toas, capsys
):
    """Record wsrt167 F0 timings by graph mode (no hard thresholds)."""
    from jug.testing.tempo2_graph_timing import benchmark_wsrt167_graph_modes

    report = benchmark_wsrt167_graph_modes(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        ["F0"],
        fixture_id="wsrt167_f0",
        jug_result=wsrt167_jug,
        params=wsrt167_params,
        toas=wsrt167_toas,
    )
    print("\n".join(report.summary_lines()))
    for m in report.modes.values():
        for key, value in m.as_dict().items():
            if key.endswith("_sec") and isinstance(value, (int, float)):
                assert value > 0.0, f"{m.mode}.{key} must be positive"
                assert np.isfinite(value), f"{m.mode}.{key} must be finite"
        assert m.pack_build_calls_wls_path == 1


@pytest.mark.slow
def test_wsrt167_graph_timing_multiparam(
    wsrt167_fixture, wsrt167_jug, wsrt167_params, wsrt167_toas, capsys
):
    """Record wsrt167 4-param timings — design-note compile benchmark."""
    from jug.testing.tempo2_graph_timing import (
        TEMPO2_GRAPH_FIXED_STATE_BCLT,
        TEMPO2_GRAPH_FIXED_STATE_STRIPPED,
        benchmark_wsrt167_graph_modes,
    )

    report = benchmark_wsrt167_graph_modes(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        ["RAJ", "DECJ", "F0", "DM"],
        fixture_id="wsrt167_4param",
        jug_result=wsrt167_jug,
        params=wsrt167_params,
        toas=wsrt167_toas,
    )
    print("\n".join(report.summary_lines()))
    stripped = report.modes[TEMPO2_GRAPH_FIXED_STATE_STRIPPED]
    bclt = report.modes[TEMPO2_GRAPH_FIXED_STATE_BCLT]
    total_s = stripped.residual_first_jit_sec + stripped.jac_first_jit_sec
    total_b = bclt.residual_first_jit_sec + bclt.jac_first_jit_sec
    print(
        f"stripped sum_jit={total_s:.1f}s fixed_state_bclt sum_jit={total_b:.1f}s "
        f"speedup={total_b / total_s:.2f}x"
    )
    assert total_s > 0 and total_b > 0


def test_stripped_prepare_trace_shklovskii_from_fit_params(wsrt167_setup_multiparam):
    """Stripped dev-oracle path mirrors production Shklovskii routing."""
    from jug.residuals.tempo2.fit_setup import prepare_tempo2_chain_from_simple_result

    setup = wsrt167_setup_multiparam
    static = setup.native_chain_static
    assert static is not None
    params = setup.params
    toas = static["toas"]
    jug = {
        "term_diagnostics": static["term_diagnostics"],
        "dt_sec": static["dt_sec"],
        "freq_bary_mhz": static["freq_bary_mhz"],
        "compatibility": setup.compatibility,
        "tempo2_native": "fixed_state_stripped",
    }
    captured: dict = {}
    from jug.residuals.tempo2 import delta_pack as dp

    original = dp._build_fixed_state_pack_from_host

    def recording_build(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    with patch.object(dp, "_build_fixed_state_pack_from_host", side_effect=recording_build):
        prepare_tempo2_chain_from_simple_result(
            jug,
            params,
            toas,
            fit_params=["PMRA", "F0"],
        )

    assert captured.get("trace_shklovskii") is True
    assert captured.get("mode") == "fixed_state_stripped"
    assert captured.get("stripped_fields") is not None


@pytest.fixture
def wsrt167_fixture_paths():
    fixture = load_wsrt167_fixture()
    return fixture["par_path"], fixture["tim_path"]


@pytest.fixture
def wsrt167_setup(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(
        ["F0"], tempo2_native="fixed_state_bclt"
    )


@pytest.fixture
def wsrt167_setup_multiparam(wsrt167_fit_setup_factory):
    return wsrt167_fit_setup_factory(
        ["RAJ", "DECJ", "F0", "DM"], tempo2_native="fixed_state_bclt"
    )