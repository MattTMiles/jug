"""JUG pytest configuration."""

import pytest

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


def pytest_collection_modifyitems(config, items):
    del config
    for item in items:
        mod_name = item.module.__name__.rsplit(".", 1)[-1]
        if mod_name.startswith(DEV_ORACLE_TEST_PREFIXES):
            item.add_marker(pytest.mark.dev_oracle)
