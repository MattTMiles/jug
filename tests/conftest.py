"""JUG pytest configuration."""

pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "jug_numpy_jax_parity_deprecated: TEMPORARY — compares deprecated NumPy "
        "residual path to JAX; remove after JAX migration",
    )
