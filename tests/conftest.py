"""JUG pytest configuration."""

import pytest

import numpy as np

pytest_plugins = []

# Module name prefixes for tests that call libstempo/tempo2/pytempo oracles.
DEV_ORACLE_TEST_PREFIXES = ("test_dev_oracle_", "test_tempo2_")


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "dev_oracle: external libstempo/tempo2/pytempo oracle — delete with jug/testing/ harness",
    )
    config.addinivalue_line(
        "markers",
        "jug_numpy_jax_parity_deprecated: TEMPORARY — compares deprecated NumPy "
        "residual path to JAX; remove after JAX migration",
    )
    config.addinivalue_line(
        "markers",
        "probe: diagnostic-only test (writes reports, no assertions); skip in normal CI",
    )
    config.addinivalue_line(
        "markers",
        "slow: oracle-heavy or full-fixture test; skip in fast inner-loop runs",
    )


def pytest_collection_modifyitems(config, items):
    del config
    skip_probe = pytest.mark.skip(
        reason="probe: run with `pytest -m probe` when diagnostics are needed"
    )
    for item in items:
        mod_name = item.module.__name__.rsplit(".", 1)[-1]
        if mod_name.startswith(DEV_ORACLE_TEST_PREFIXES):
            item.add_marker(pytest.mark.dev_oracle)
        if "probe" in item.keywords:
            item.add_marker(skip_probe)


@pytest.fixture(scope="session")
def wsrt167_fixture():
    from tempo2_native_test_helpers import load_wsrt167_fixture

    return load_wsrt167_fixture()


@pytest.fixture(scope="session")
def wsrt167_pytempo_oracle(wsrt167_fixture):
    pytest.importorskip("pytempo")
    from jug.testing.tempo2_pytempo_oracle import load_pytempo_native_oracle

    return load_pytempo_native_oracle(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        fixture_id="wsrt167",
    )


@pytest.fixture(scope="session")
def wsrt167_native_terms(wsrt167_fixture):
    """Amortize one wsrt167 JAX native-chain compile across a module."""
    from jug.utils.jax_setup import ensure_jax_x64
    from tempo2_native_test_helpers import compute_native_terms_for_fixture

    ensure_jax_x64()
    return compute_native_terms_for_fixture(wsrt167_fixture)


@pytest.fixture(scope="session")
def wsrt167_jug(wsrt167_fixture):
    from jug.residuals.simple_calculator import compute_residuals_simple

    return compute_residuals_simple(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        verbose=False,
        compatibility="tempo2",
    )


@pytest.fixture(scope="session")
def wsrt167_libstempo(wsrt167_fixture):
    pytest.importorskip("libstempo")
    from jug.testing.tempo2_reference import tempo2_reference

    return tempo2_reference(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
    )


@pytest.fixture(scope="session")
def wsrt167_clock_inputs(wsrt167_fixture, wsrt167_jug):
    from jug.io.par_reader import parse_par_file
    from jug.io.tim_reader import parse_tim_file_mjds
    from jug.residuals.tempo2_native.chain_jax import _load_model_static_for_native_chain

    params = parse_par_file(wsrt167_fixture["par_path"])
    toas = parse_tim_file_mjds(wsrt167_fixture["tim_path"])
    sat = np.asarray(wsrt167_jug["term_diagnostics"]["sat_mjd"], dtype=np.float64)
    static = _load_model_static_for_native_chain(params, toas, wsrt167_jug)
    return wsrt167_fixture, sat, static


@pytest.fixture(scope="session")
def wsrt167_formbats_report(wsrt167_fixture):
    """Single overlay-path formBats ranking per module (expensive)."""
    from jug.testing.tempo2_formbats_closure import compare_formbats_components

    return compare_formbats_components(
        wsrt167_fixture["par_path"],
        wsrt167_fixture["tim_path"],
        fixture_id="wsrt167",
    )
